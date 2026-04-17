import { Router, type IRouter, Request, Response } from "express";
import { openai } from "@workspace/integrations-openai-ai-server";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { proxyAuth } from "../../middlewares/proxyAuth.js";

const router: IRouter = Router();

router.use(proxyAuth);

const OPENAI_MODELS = [
  { id: "gpt-5.4", object: "model", owned_by: "openai" },
  { id: "gpt-5.2", object: "model", owned_by: "openai" },
  { id: "gpt-5.1", object: "model", owned_by: "openai" },
  { id: "gpt-5", object: "model", owned_by: "openai" },
  { id: "gpt-5-mini", object: "model", owned_by: "openai" },
  { id: "gpt-5-nano", object: "model", owned_by: "openai" },
  { id: "o4-mini", object: "model", owned_by: "openai" },
  { id: "o3", object: "model", owned_by: "openai" },
];

const ANTHROPIC_MODELS = [
  { id: "claude-opus-4-7", object: "model", owned_by: "anthropic" },
  { id: "claude-opus-4-6", object: "model", owned_by: "anthropic" },
  { id: "claude-opus-4-5", object: "model", owned_by: "anthropic" },
  { id: "claude-sonnet-4-6", object: "model", owned_by: "anthropic" },
  { id: "claude-sonnet-4-5", object: "model", owned_by: "anthropic" },
  { id: "claude-haiku-4-5", object: "model", owned_by: "anthropic" },
];

router.get("/models", (_req: Request, res: Response) => {
  res.json({ object: "list", data: [...OPENAI_MODELS, ...ANTHROPIC_MODELS] });
});

type OpenAIMessage = { role: string; content: string | null };
type ReasoningEffort = "low" | "medium" | "high";

function normalizeModel(model: string): string {
  return model.replace(/-\d{8}$/, "");
}

function isClaudeModel(model: string): boolean {
  return model.startsWith("claude");
}

function isReasoningModel(model: string): boolean {
  return model.startsWith("o3") || model.startsWith("o4") || model === "gpt-5.4" || model.startsWith("gpt-5.4-");
}

function parseBetaHeaders(req: Request): string[] {
  const raw = req.headers["anthropic-beta"];
  if (!raw) return [];
  const value = Array.isArray(raw) ? raw.join(",") : raw;
  return value.split(",").map((s) => s.trim()).filter(Boolean);
}

function isTransientError(err: unknown): boolean {
  const e = err as Record<string, unknown>;
  const msg = (e["message"] as string) ?? "";
  const status = e["status"] as number | undefined;
  if (msg.includes("auth_unavailable")) return false;
  if (status === 429) return true;
  if (msg.includes("i/o timeout") || msg.includes("dial tcp")) return true;
  return false;
}

function retryDelay(err: unknown, attempt: number): number {
  const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
  if (msg.includes("i/o timeout") || msg.includes("dial tcp")) return 1000;
  return 1500 * attempt;
}

function maxRetryAttempts(err: unknown, defaultMax: number): number {
  const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
  if (msg.includes("i/o timeout") || msg.includes("dial tcp")) return 2;
  return defaultMax;
}

async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const max = maxRetryAttempts(err, attempts);
      if (!isTransientError(err) || attempt >= max) throw err;
      await new Promise((r) => setTimeout(r, retryDelay(err, attempt)));
    }
  }
  throw lastErr;
}

async function handleOpenAIRoute(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as OpenAIMessage[]) ?? [];
  const stream = body["stream"] === true;
  const temperature = body["temperature"] as number | undefined;
  const maxTokens = (body["max_completion_tokens"] ?? body["max_tokens"]) as number | undefined;
  const reasoningEffort = body["reasoning_effort"] as ReasoningEffort | undefined;

  const chatMessages = messages.map((m) => ({
    role: m.role as "user" | "assistant" | "system",
    content: m.content ?? "",
  }));

  const isReasoning = isReasoningModel(model);

  const extraParams = {
    ...(temperature !== undefined && !isReasoning ? { temperature } : {}),
    ...(maxTokens !== undefined ? { max_completion_tokens: maxTokens } : {}),
    ...(reasoningEffort !== undefined && isReasoning ? { reasoning_effort: reasoningEffort } : {}),
  };

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const completionStream = await withRetry(() =>
        openai.chat.completions.create({
          model,
          messages: chatMessages,
          stream: true,
          stream_options: { include_usage: true },
          ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0])
      );

      for await (const chunk of completionStream) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const completion = await withRetry(() =>
        openai.chat.completions.create({
          model,
          messages: chatMessages,
          stream: false,
          ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0])
      );
      res.json(completion);
    }
  } catch (err: unknown) {
    req.log.error({ err }, "OpenAI completion error");
    if (!res.headersSent) {
      const errObj = err as Record<string, unknown>;
      const status = (errObj["status"] as number) ?? 500;
      const upstreamMsg = (errObj["message"] as string) ?? "Upstream OpenAI error";
      res.status(status).json({ error: { message: upstreamMsg, type: "api_error" } });
    } else {
      res.end();
    }
  }
}

type AnthropicThinking = { type: "enabled"; budget_tokens: number } | { type: "disabled" };

async function handleClaudeViaOpenAIFormat(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as OpenAIMessage[]) ?? [];
  const stream = body["stream"] === true;
  const temperature = body["temperature"] as number | undefined;
  const maxTokens = (body["max_completion_tokens"] ?? body["max_tokens"]) as number | undefined;
  const thinking = body["thinking"] as AnthropicThinking | undefined;
  const betas = parseBetaHeaders(req);
  const requestOptions = betas.length > 0 ? { headers: { "anthropic-beta": betas.join(",") } } : {};

  const systemMessages = messages.filter((m) => m.role === "system");
  const systemPrompt = systemMessages.map((m) => m.content ?? "").join("\n");

  // Build Anthropic-format messages from the non-system turns, stripping any
  // thinking blocks that may be in assistant history (cross-node signature issue).
  let anthropicMessages = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content ?? "",
    }));

  // Strip trailing assistant messages (Vertex AI does not support prefill)
  while (anthropicMessages.length > 0 && anthropicMessages[anthropicMessages.length - 1]?.role === "assistant") {
    anthropicMessages = anthropicMessages.slice(0, -1);
  }

  const requestId = `chatcmpl-proxy-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  const baseParams = {
    model,
    max_tokens: maxTokens ?? 16000,
    messages: anthropicMessages,
    ...(systemPrompt ? { system: systemPrompt } : {}),
    ...(thinking ? { thinking } : temperature !== undefined ? { temperature } : {}),
  };

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const anthropicStream = await withRetry(() =>
        anthropic.messages.create(
          { ...baseParams, stream: true } as Parameters<typeof anthropic.messages.create>[0],
          requestOptions,
        )
      );

      let thinkingBuffer = "";
      let inputTokens = 0;
      let outputTokens = 0;

      for await (const event of (anthropicStream as AsyncIterable<Record<string, unknown>>)) {
        const type = event["type"] as string;

        if (type === "message_start") {
          const msg = event["message"] as Record<string, unknown>;
          const usage = msg["usage"] as Record<string, unknown> | undefined;
          inputTokens = (usage?.["input_tokens"] as number) ?? 0;
        } else if (type === "content_block_delta") {
          const delta = event["delta"] as Record<string, unknown>;
          const deltaType = delta["type"] as string;

          if (deltaType === "thinking_delta") {
            const thinking = delta["thinking"] as string;
            thinkingBuffer += thinking;
            const chunk = {
              id: requestId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: { reasoning_content: thinking }, finish_reason: null }],
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          } else if (deltaType === "text_delta") {
            const chunk = {
              id: requestId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: { content: delta["text"] as string }, finish_reason: null }],
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        } else if (type === "message_delta") {
          const delta = event["delta"] as Record<string, unknown>;
          const usage = event["usage"] as Record<string, unknown> | undefined;
          outputTokens = (usage?.["output_tokens"] as number) ?? outputTokens;
          if (delta["stop_reason"]) {
            const stopReason = delta["stop_reason"] === "end_turn" ? "stop" : delta["stop_reason"] as string;
            const chunk = {
              id: requestId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: {}, finish_reason: stopReason }],
              usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        }
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const message = await withRetry(() =>
        anthropic.messages.create(
          baseParams as Parameters<typeof anthropic.messages.create>[0],
          requestOptions,
        )
      );

      const thinkingBlock = message.content.find((b) => b.type === "thinking");
      const textBlock = message.content.find((b) => b.type === "text");
      const thinkingText = thinkingBlock && thinkingBlock.type === "thinking" ? thinkingBlock.thinking : null;
      const text = textBlock && textBlock.type === "text" ? textBlock.text : "";

      res.json({
        id: requestId,
        object: "chat.completion",
        created,
        model,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: text,
            ...(thinkingText ? { reasoning_content: thinkingText } : {}),
          },
          finish_reason: message.stop_reason === "end_turn" ? "stop" : message.stop_reason,
        }],
        usage: {
          prompt_tokens: message.usage.input_tokens,
          completion_tokens: message.usage.output_tokens,
          total_tokens: message.usage.input_tokens + message.usage.output_tokens,
        },
      });
    }
  } catch (err: unknown) {
    req.log.error({ err }, "Anthropic via OpenAI-format error");
    if (!res.headersSent) {
      const errObj = err as Record<string, unknown>;
      const status = (errObj["status"] as number) ?? 500;
      const upstreamMsg = (errObj["message"] as string) ?? "Upstream Anthropic error";
      res.status(status).json({ error: { message: upstreamMsg, type: "api_error" } });
    } else {
      res.end();
    }
  }
}

router.post("/chat/completions", async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const originalModel = (body["model"] as string) || "gpt-5.2";
  // Normalize only for routing decision — the original model name (with any
  // date suffix) is forwarded to the upstream API.
  const routingModel = normalizeModel(originalModel);

  if (isClaudeModel(routingModel)) {
    await handleClaudeViaOpenAIFormat(originalModel, body, req, res);
  } else {
    await handleOpenAIRoute(originalModel, body, req, res);
  }
});

export default router;
