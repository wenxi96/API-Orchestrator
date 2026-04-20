import { Router, type IRouter, Request, Response } from "express";
import { openai } from "@workspace/integrations-openai-ai-server";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { proxyAuth } from "../../middlewares/proxyAuth.js";
import {
  normalizeModel,
  parseBetaHeaders,
  generateId,
  withRetry,
  mapAnthropicStopToOpenAI,
  abortOnClientClose,
} from "../../lib/upstream.js";

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

function isClaudeModel(model: string): boolean {
  return model.startsWith("claude");
}

function isReasoningModel(model: string): boolean {
  return model.startsWith("o3") || model.startsWith("o4")
    || model === "gpt-5.4" || model.startsWith("gpt-5.4-");
}

async function handleOpenAIRoute(
  model: string, body: Record<string, unknown>, req: Request, res: Response,
): Promise<void> {
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

  const abort = abortOnClientClose(req);
  const reqOpts = { signal: abort.signal };

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const completionStream = await withRetry(() =>
        openai.chat.completions.create({
          model, messages: chatMessages, stream: true,
          stream_options: { include_usage: true },
          ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0], reqOpts)
      );

      for await (const chunk of completionStream) {
        if (abort.signal.aborted) break;
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const completion = await withRetry(() =>
        openai.chat.completions.create({
          model, messages: chatMessages, stream: false, ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0], reqOpts)
      );
      res.json(completion);
    }
  } catch (err: unknown) {
    if (abort.signal.aborted) {
      req.log.info({ model }, "Client disconnected — upstream aborted");
      return;
    }
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

async function handleClaudeViaOpenAIFormat(
  model: string, body: Record<string, unknown>, req: Request, res: Response,
): Promise<void> {
  const messages = (body["messages"] as OpenAIMessage[]) ?? [];
  const stream = body["stream"] === true;
  const temperature = body["temperature"] as number | undefined;
  const maxTokens = (body["max_completion_tokens"] ?? body["max_tokens"]) as number | undefined;
  const thinking = body["thinking"] as AnthropicThinking | undefined;
  const betas = parseBetaHeaders(req);

  const systemMessages = messages.filter((m) => m.role === "system");
  const systemPrompt = systemMessages.map((m) => m.content ?? "").join("\n\n");

  let anthropicMessages = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content ?? "",
    }));

  // Vertex AI does not support assistant prefill
  while (anthropicMessages.length > 0 && anthropicMessages[anthropicMessages.length - 1]?.role === "assistant") {
    anthropicMessages = anthropicMessages.slice(0, -1);
  }

  const requestId = generateId("chatcmpl");
  const created = Math.floor(Date.now() / 1000);

  const baseParams = {
    model, max_tokens: maxTokens ?? 16000,
    messages: anthropicMessages,
    ...(systemPrompt ? { system: systemPrompt } : {}),
    ...(thinking ? { thinking } : temperature !== undefined ? { temperature } : {}),
  };

  const abort = abortOnClientClose(req);
  const requestOptions: Record<string, unknown> = { signal: abort.signal };
  if (betas.length > 0) requestOptions["headers"] = { "anthropic-beta": betas.join(",") };

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

      let inputTokens = 0;
      let outputTokens = 0;

      for await (const event of (anthropicStream as AsyncIterable<Record<string, unknown>>)) {
        if (abort.signal.aborted) break;
        const type = event["type"] as string;

        if (type === "message_start") {
          const msg = event["message"] as Record<string, unknown>;
          const usage = msg["usage"] as Record<string, unknown> | undefined;
          inputTokens = (usage?.["input_tokens"] as number) ?? 0;
        } else if (type === "content_block_delta") {
          const delta = event["delta"] as Record<string, unknown>;
          const deltaType = delta["type"] as string;

          if (deltaType === "thinking_delta") {
            const chunk = {
              id: requestId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: { reasoning_content: delta["thinking"] as string }, finish_reason: null }],
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
            const chunk = {
              id: requestId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: {}, finish_reason: mapAnthropicStopToOpenAI(delta["stop_reason"] as string) }],
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
        id: requestId, object: "chat.completion", created, model,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: text,
            ...(thinkingText ? { reasoning_content: thinkingText } : {}),
          },
          finish_reason: mapAnthropicStopToOpenAI(message.stop_reason),
        }],
        usage: {
          prompt_tokens: message.usage.input_tokens,
          completion_tokens: message.usage.output_tokens,
          total_tokens: message.usage.input_tokens + message.usage.output_tokens,
        },
      });
    }
  } catch (err: unknown) {
    if (abort.signal.aborted) {
      req.log.info({ model }, "Client disconnected — upstream aborted");
      return;
    }
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
  const routingModel = normalizeModel(originalModel);

  if (isClaudeModel(routingModel)) {
    await handleClaudeViaOpenAIFormat(originalModel, body, req, res);
  } else {
    await handleOpenAIRoute(originalModel, body, req, res);
  }
});

export default router;
