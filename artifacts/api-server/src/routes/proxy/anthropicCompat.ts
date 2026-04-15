import { Router, type IRouter, Request, Response } from "express";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { openai } from "@workspace/integrations-openai-ai-server";
import { proxyAuth } from "../../middlewares/proxyAuth.js";

const router: IRouter = Router();

router.use(proxyAuth);

type AnthropicMessage = { role: "user" | "assistant"; content: unknown };
type AnthropicThinking = { type: "enabled"; budget_tokens: number } | { type: "disabled" };
type ReasoningEffort = "low" | "medium" | "high";

function normalizeModel(model: string): string {
  return model.replace(/-\d{8}$/, "");
}

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt") || model.startsWith("o3") || model.startsWith("o4");
}

function isReasoningModel(model: string): boolean {
  return model.startsWith("o3") || model.startsWith("o4");
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
  // auth_unavailable is a Replit AI Integration cooldown that lasts several minutes.
  // Retrying here is futile — pass it through immediately so the client's own
  // backoff (e.g. Claude Code's 10-attempt retry) can handle it properly.
  if (msg.includes("auth_unavailable")) return false;
  return status === 429;
}

async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 3, delayMs = 1500): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (!isTransientError(err) || attempt === maxAttempts) throw err;
      await new Promise((r) => setTimeout(r, delayMs * attempt));
    }
  }
  throw lastErr;
}

async function handleAnthropicRoute(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as AnthropicMessage[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 16000;
  const system = body["system"] as unknown;
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;
  const thinking = body["thinking"] as AnthropicThinking | undefined;
  const betas = parseBetaHeaders(req);

  const baseParams: Record<string, unknown> = {
    model,
    max_tokens: maxTokens,
    messages,
    ...(system !== undefined ? { system } : {}),
    ...(thinking ? { thinking } : temperature !== undefined ? { temperature } : {}),
    ...(body["stop_sequences"] !== undefined ? { stop_sequences: body["stop_sequences"] } : {}),
    ...(body["top_p"] !== undefined ? { top_p: body["top_p"] } : {}),
    ...(body["top_k"] !== undefined ? { top_k: body["top_k"] } : {}),
    ...(body["metadata"] !== undefined ? { metadata: body["metadata"] } : {}),
    ...(body["tools"] !== undefined ? { tools: body["tools"] } : {}),
    ...(body["tool_choice"] !== undefined ? { tool_choice: body["tool_choice"] } : {}),
  };

  const requestOptions = betas.length > 0 ? { headers: { "anthropic-beta": betas.join(",") } } : {};

  try {
    if (stream) {
      // Use create({stream:true}) — the Replit AI Integration's stream helper
      // objects don't expose initialMessage(), so we use the create() API instead.
      // Wrapping just the create() call in withRetry means we can still retry
      // transient auth/503/429 errors before any response headers are sent.
      const eventStream = await withRetry(() =>
        anthropic.messages.create(
          { ...baseParams, stream: true } as Parameters<typeof anthropic.messages.create>[0],
          requestOptions,
        )
      );

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      for await (const event of (eventStream as AsyncIterable<{ type: string }>)) {
        res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
      res.end();
    } else {
      const message = await withRetry(() =>
        anthropic.messages.create(
          baseParams as Parameters<typeof anthropic.messages.create>[0],
          requestOptions,
        )
      );
      res.json(message);
    }
  } catch (err: unknown) {
    req.log.error({ err }, "Anthropic messages error");
    if (!res.headersSent) {
      const errObj = err as Record<string, unknown>;
      const status = (errObj["status"] as number) ?? 500;
      const upstreamMsg = (errObj["message"] as string) ?? "Upstream Anthropic error";
      res.status(status).json({ type: "error", error: { type: "api_error", message: upstreamMsg } });
    } else {
      res.end();
    }
  }
}

async function handleOpenAIViaAnthropicFormat(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as { role: "user" | "assistant"; content: string }[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 16000;
  const system = body["system"] as string | undefined;
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;
  const reasoningEffort = body["reasoning_effort"] as ReasoningEffort | undefined;

  const openaiMessages: { role: "system" | "user" | "assistant"; content: string }[] = [];
  if (system) openaiMessages.push({ role: "system", content: system });
  for (const m of messages) openaiMessages.push({ role: m.role, content: m.content });

  const messageId = `msg_proxy_${Date.now()}`;
  const isReasoning = isReasoningModel(model);

  const extraParams = {
    ...(temperature !== undefined && !isReasoning ? { temperature } : {}),
    max_completion_tokens: maxTokens,
    ...(reasoningEffort !== undefined && isReasoning ? { reasoning_effort: reasoningEffort } : {}),
  };

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const startEvent = {
        type: "message_start",
        message: {
          id: messageId, type: "message", role: "assistant", content: [],
          model, stop_reason: null, stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      };
      res.write(`event: message_start\ndata: ${JSON.stringify(startEvent)}\n\n`);
      res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`);
      res.write(`event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`);

      const completionStream = await openai.chat.completions.create({
        model, messages: openaiMessages, stream: true,
        stream_options: { include_usage: true },
        ...extraParams,
      } as Parameters<typeof openai.chat.completions.create>[0]);

      let outputTokens = 0;
      let stopReason = "end_turn";
      let reasoningBuffer = "";
      let reasoningBlockStarted = false;

      for await (const chunk of completionStream) {
        const delta = chunk.choices[0]?.delta as Record<string, unknown> | undefined;
        const reasoningContent = delta?.["reasoning_content"] as string | undefined;
        if (reasoningContent) {
          if (!reasoningBlockStarted) {
            reasoningBlockStarted = true;
            res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } })}\n\n`);
          }
          reasoningBuffer += reasoningContent;
          res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: reasoningContent } })}\n\n`);
        }
        const content = delta?.["content"] as string | undefined;
        if (content) {
          if (reasoningBlockStarted) {
            res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
            res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 1, content_block: { type: "text", text: "" } })}\n\n`);
            reasoningBlockStarted = false;
          }
          outputTokens += 1;
          res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: reasoningBuffer ? 1 : 0, delta: { type: "text_delta", text: content } })}\n\n`);
        }
        const finishReason = chunk.choices[0]?.finish_reason;
        if (finishReason) stopReason = finishReason === "stop" ? "end_turn" : finishReason;
        if (chunk.usage) outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      }

      res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: reasoningBuffer ? 1 : 0 })}\n\n`);
      res.write(`event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`);
      res.write(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
      res.end();
    } else {
      const completion = await openai.chat.completions.create({
        model, messages: openaiMessages, stream: false, ...extraParams,
      } as Parameters<typeof openai.chat.completions.create>[0]);

      const msg = completion.choices[0]?.message as Record<string, unknown> | undefined;
      const text = (msg?.["content"] as string) ?? "";
      const reasoningContent = msg?.["reasoning_content"] as string | undefined;
      const finishReason = completion.choices[0]?.finish_reason ?? "stop";

      const contentBlocks: unknown[] = [];
      if (reasoningContent) contentBlocks.push({ type: "thinking", thinking: reasoningContent });
      contentBlocks.push({ type: "text", text });

      res.json({
        id: messageId, type: "message", role: "assistant",
        content: contentBlocks, model,
        stop_reason: finishReason === "stop" ? "end_turn" : finishReason,
        stop_sequence: null,
        usage: {
          input_tokens: completion.usage?.prompt_tokens ?? 0,
          output_tokens: completion.usage?.completion_tokens ?? 0,
        },
      });
    }
  } catch (err: unknown) {
    req.log.error({ err }, "OpenAI via Anthropic-format error");
    if (!res.headersSent) {
      res.status(500).json({ type: "error", error: { type: "api_error", message: "Upstream OpenAI error" } });
    } else {
      res.end();
    }
  }
}

router.post("/messages", async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const model = normalizeModel((body["model"] as string) || "claude-sonnet-4-6");

  if (isOpenAIModel(model)) {
    await handleOpenAIViaAnthropicFormat(model, body, req, res);
  } else {
    await handleAnthropicRoute(model, body, req, res);
  }
});

export default router;
