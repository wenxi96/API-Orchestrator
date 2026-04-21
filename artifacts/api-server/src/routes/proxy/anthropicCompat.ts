import { Router, type IRouter, Request, Response } from "express";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { openai } from "@workspace/integrations-openai-ai-server";
import { proxyAuth } from "../../middlewares/proxyAuth.js";
import {
  normalizeModel,
  parseBetaHeaders,
  generateId,
  withRetry,
  withModelFallback,
  getFallbackChain,
  isAuthUnavailableError,
  formatAuthUnavailableMessage,
  mapOpenAIFinishToAnthropic,
  abortOnClientClose,
} from "../../lib/upstream.js";

const router: IRouter = Router();

router.use(proxyAuth);

type AnthropicMessage = { role: "user" | "assistant"; content: unknown };
type AnthropicThinking = { type: "enabled"; budget_tokens: number } | { type: "disabled" };
type ReasoningEffort = "low" | "medium" | "high";

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt") || model.startsWith("o3") || model.startsWith("o4");
}

function isReasoningModel(model: string): boolean {
  return model.startsWith("o3") || model.startsWith("o4")
    || model === "gpt-5.4" || model.startsWith("gpt-5.4-");
}

function isStreamingRequiredError(err: unknown): boolean {
  const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
  return msg.includes("Streaming is required");
}

// Beta headers known to produce long-running generations (minutes).
// Non-streaming requests with these headers tend to hit intermediate proxy
// idle-connection timeouts (~2-3 min) before the response arrives. We detect
// them early and always use streaming internally so the upstream connection
// stays active throughout generation.
const LONG_RUNNING_BETAS = ["interleaved-thinking", "context-1m"];

function isLongRunningRequest(betas: string[]): boolean {
  return betas.some((b) => LONG_RUNNING_BETAS.some((lr) => b.includes(lr)));
}

// Collect a streaming response and reassemble it into a complete Message object.
async function collectStreamAsMessage(
  params: Record<string, unknown>,
  requestOptions: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const eventStream = await anthropic.messages.create(
    { ...params, stream: true } as Parameters<typeof anthropic.messages.create>[0],
    requestOptions,
  );

  type ContentBlock = { type: string; text?: string; thinking?: string; signature?: string; input?: string; id?: string; name?: string };
  let messageBase: Record<string, unknown> = {};
  const contentBlocks: ContentBlock[] = [];

  for await (const event of (eventStream as AsyncIterable<Record<string, unknown>>)) {
    const type = event["type"] as string;

    if (type === "message_start") {
      messageBase = { ...(event["message"] as Record<string, unknown>) };
      messageBase["content"] = [];
    } else if (type === "content_block_start") {
      const idx = event["index"] as number;
      contentBlocks[idx] = { ...(event["content_block"] as ContentBlock) };
    } else if (type === "content_block_delta") {
      const idx = event["index"] as number;
      const delta = event["delta"] as Record<string, unknown>;
      const block = contentBlocks[idx];
      if (!block) continue;
      const deltaType = delta["type"] as string;
      if (deltaType === "text_delta") {
        block.text = (block.text ?? "") + (delta["text"] as string);
      } else if (deltaType === "thinking_delta") {
        block.thinking = (block.thinking ?? "") + (delta["thinking"] as string);
      } else if (deltaType === "signature_delta") {
        block.signature = delta["signature"] as string;
      } else if (deltaType === "input_json_delta") {
        block.input = (block.input ?? "") + (delta["partial_json"] as string);
      }
    } else if (type === "message_delta") {
      const delta = event["delta"] as Record<string, unknown>;
      if (delta["stop_reason"]) messageBase["stop_reason"] = delta["stop_reason"];
      if (delta["stop_sequence"] !== undefined) messageBase["stop_sequence"] = delta["stop_sequence"];
      const usage = event["usage"] as Record<string, unknown> | undefined;
      if (usage) {
        messageBase["usage"] = {
          ...(messageBase["usage"] as Record<string, unknown> ?? {}),
          ...usage,
        };
      }
    }
  }

  for (const block of contentBlocks) {
    if (block.type === "tool_use" && typeof block.input === "string") {
      try { block.input = JSON.parse(block.input); } catch { /* leave as-is */ }
    }
  }
  messageBase["content"] = contentBlocks.filter(Boolean);
  return messageBase;
}

// Normalize the `system` field: string or array of [{type:"text", text:"..."}].
function extractSystemText(raw: unknown): string | undefined {
  if (!raw) return undefined;
  if (typeof raw === "string") return raw || undefined;
  if (Array.isArray(raw)) {
    const text = (raw as Array<{ type: string; text?: string }>)
      .filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("\n");
    return text || undefined;
  }
  return undefined;
}

async function handleAnthropicRoute(
  model: string, body: Record<string, unknown>, req: Request, res: Response,
): Promise<void> {
  let messages = (body["messages"] as AnthropicMessage[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 16000;
  const system = body["system"] as unknown;
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;
  const thinking = body["thinking"] as AnthropicThinking | undefined;
  const betas = parseBetaHeaders(req);

  // Vertex AI does not support assistant-prefill; trim trailing assistant turns.
  while (messages.length > 0 && messages[messages.length - 1]?.role === "assistant") {
    req.log.warn({ model }, "Stripped trailing assistant message (no prefill on Vertex AI)");
    messages = messages.slice(0, -1);
  }

  // Strip thinking blocks from history — signatures are bound to the originating
  // Vertex AI orchestrator node and can't transfer across nodes (wendcc1 vs base).
  messages = messages.map((msg) => {
    if (msg.role !== "assistant") return msg;
    const content = msg.content;
    if (!Array.isArray(content)) return msg;
    const filtered = content.filter(
      (block: unknown) => (block as Record<string, unknown>)["type"] !== "thinking"
    );
    return filtered.length === content.length ? msg : { ...msg, content: filtered };
  });

  // Build params with a substitutable model — required so model fallback can
  // swap in a different model without rebuilding the entire param object.
  const buildParams = (m: string): Record<string, unknown> => ({
    model: m, max_tokens: maxTokens, messages,
    ...(system !== undefined ? { system } : {}),
    ...(thinking ? { thinking } : temperature !== undefined ? { temperature } : {}),
    ...(body["stop_sequences"] !== undefined ? { stop_sequences: body["stop_sequences"] } : {}),
    ...(body["top_p"] !== undefined ? { top_p: body["top_p"] } : {}),
    ...(body["top_k"] !== undefined ? { top_k: body["top_k"] } : {}),
    ...(body["metadata"] !== undefined ? { metadata: body["metadata"] } : {}),
    ...(body["tools"] !== undefined ? { tools: body["tools"] } : {}),
    ...(body["tool_choice"] !== undefined ? { tool_choice: body["tool_choice"] } : {}),
  });

  // Cancel upstream when client disconnects (saves tokens on broken streams).
  const abort = abortOnClientClose(req);
  const requestOptions: Record<string, unknown> = { signal: abort.signal };
  if (betas.length > 0) requestOptions["headers"] = { "anthropic-beta": betas.join(",") };

  const onFallback = (failed: string, next: string): void => {
    req.log.warn({ originalModel: model, failed, next },
      "auth_unavailable — auto-falling back to next model in chain");
  };

  try {
    if (stream) {
      // Set headers BEFORE attempting upstream — so on fallback exhaustion
      // we can still write a friendly error event without violating SSE.
      // Actually keep headers unset until we have a successful upstream call,
      // because fallback may take multiple attempts and we want to be able to
      // return JSON 503 if all fail.
      const { result: eventStream, usedModel } = await withModelFallback(model,
        (m) => withRetry(() =>
          anthropic.messages.create(
            { ...buildParams(m), stream: true } as Parameters<typeof anthropic.messages.create>[0],
            requestOptions,
          )
        ),
        onFallback,
      );

      if (usedModel !== model) {
        req.log.info({ originalModel: model, usedModel }, "Stream served by fallback model");
      }

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      // Custom header so clients can tell when a fallback was used.
      if (usedModel !== model) res.setHeader("X-Proxy-Fallback-Model", usedModel);

      for await (const event of (eventStream as AsyncIterable<{ type: string }>)) {
        if (abort.signal.aborted) break;
        res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
      res.end();
    } else {
      const callUpstream = async (m: string): Promise<Record<string, unknown>> => {
        const params = buildParams(m);
        if (isLongRunningRequest(betas)) {
          req.log.info({ model: m, betas }, "Long-running betas — using stream-and-collect");
          return await withRetry(() => collectStreamAsMessage(params, requestOptions));
        }
        try {
          return await withRetry(() =>
            anthropic.messages.create(
              params as Parameters<typeof anthropic.messages.create>[0],
              requestOptions,
            )
          ) as unknown as Record<string, unknown>;
        } catch (firstErr: unknown) {
          if (!isStreamingRequiredError(firstErr)) throw firstErr;
          req.log.warn({ model: m }, "Upstream requires streaming — fallback to stream-and-collect");
          return await withRetry(() => collectStreamAsMessage(params, requestOptions));
        }
      };

      const { result: message, usedModel } = await withModelFallback(model, callUpstream, onFallback);
      if (usedModel !== model) {
        req.log.info({ originalModel: model, usedModel }, "Request served by fallback model");
        res.setHeader("X-Proxy-Fallback-Model", usedModel);
      }
      res.json(message);
    }
  } catch (err: unknown) {
    if (abort.signal.aborted) {
      req.log.info({ model }, "Client disconnected — upstream aborted");
      return;
    }
    req.log.error({ err, originalModel: model }, "Anthropic messages error");
    if (!res.headersSent) {
      const errObj = err as Record<string, unknown>;

      // Friendly message when the entire fallback chain exhausted on auth_unavailable.
      if (isAuthUnavailableError(err)) {
        const tried = [model, ...getFallbackChain(model)];
        const friendly = formatAuthUnavailableMessage(model, tried);
        res.status(503).json({
          type: "error",
          error: {
            type: "overloaded_error",
            message: friendly,
            code: "upstream_auth_cooldown",
            original_model: model,
            tried_models: tried,
          },
        });
        return;
      }

      const status = (errObj["status"] as number) ?? 500;
      const upstreamMsg = (errObj["message"] as string) ?? "Upstream Anthropic error";
      res.status(status).json({ type: "error", error: { type: "api_error", message: upstreamMsg } });
    } else {
      res.end();
    }
  }
}

async function handleOpenAIViaAnthropicFormat(
  model: string, body: Record<string, unknown>, req: Request, res: Response,
): Promise<void> {
  const messages = (body["messages"] as { role: "user" | "assistant"; content: string }[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 16000;
  const systemText = extractSystemText(body["system"]);
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;
  const reasoningEffort = body["reasoning_effort"] as ReasoningEffort | undefined;

  const openaiMessages: { role: "system" | "user" | "assistant"; content: string }[] = [];
  if (systemText) openaiMessages.push({ role: "system", content: systemText });
  for (const m of messages) openaiMessages.push({ role: m.role, content: m.content });

  const messageId = generateId("msg");
  const isReasoning = isReasoningModel(model);

  const extraParams = {
    ...(temperature !== undefined && !isReasoning ? { temperature } : {}),
    max_completion_tokens: maxTokens,
    ...(reasoningEffort !== undefined && isReasoning ? { reasoning_effort: reasoningEffort } : {}),
  };

  const abort = abortOnClientClose(req);
  const reqOpts = { signal: abort.signal };

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
      res.write(`event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`);

      const completionStream = await withRetry(() =>
        openai.chat.completions.create({
          model, messages: openaiMessages, stream: true,
          stream_options: { include_usage: true },
          ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0], reqOpts)
      );

      let outputTokens = 0;
      let openaiFinishReason: string | null = null;
      let thinkingBlockOpen = false;
      let textBlockOpen = false;
      let textBlockIndex = 0;
      let hasThinking = false;

      for await (const chunk of completionStream) {
        if (abort.signal.aborted) break;
        const delta = chunk.choices[0]?.delta as Record<string, unknown> | undefined;
        const reasoningContent = delta?.["reasoning_content"] as string | undefined;
        const content = delta?.["content"] as string | undefined;

        if (reasoningContent) {
          if (!thinkingBlockOpen) {
            thinkingBlockOpen = true;
            hasThinking = true;
            res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } })}\n\n`);
          }
          res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: reasoningContent } })}\n\n`);
        }

        if (content) {
          if (thinkingBlockOpen) {
            res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
            thinkingBlockOpen = false;
          }
          if (!textBlockOpen) {
            textBlockIndex = hasThinking ? 1 : 0;
            textBlockOpen = true;
            res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: textBlockIndex, content_block: { type: "text", text: "" } })}\n\n`);
          }
          res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: textBlockIndex, delta: { type: "text_delta", text: content } })}\n\n`);
        }

        const finishReason = chunk.choices[0]?.finish_reason;
        if (finishReason) openaiFinishReason = finishReason;
        if (chunk.usage) outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      }

      if (thinkingBlockOpen) {
        res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
      }
      if (textBlockOpen) {
        res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: textBlockIndex })}\n\n`);
      } else {
        const emptyIdx = hasThinking ? 1 : 0;
        res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: emptyIdx, content_block: { type: "text", text: "" } })}\n\n`);
        res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: emptyIdx })}\n\n`);
      }

      const stopReason = mapOpenAIFinishToAnthropic(openaiFinishReason);
      res.write(`event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`);
      res.write(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
      res.end();
    } else {
      const completion = await withRetry(() =>
        openai.chat.completions.create({
          model, messages: openaiMessages, stream: false, ...extraParams,
        } as Parameters<typeof openai.chat.completions.create>[0], reqOpts)
      );

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
        stop_reason: mapOpenAIFinishToAnthropic(finishReason),
        stop_sequence: null,
        usage: {
          input_tokens: completion.usage?.prompt_tokens ?? 0,
          output_tokens: completion.usage?.completion_tokens ?? 0,
        },
      });
    }
  } catch (err: unknown) {
    if (abort.signal.aborted) {
      req.log.info({ model }, "Client disconnected — upstream aborted");
      return;
    }
    req.log.error({ err }, "OpenAI via Anthropic-format error");
    if (!res.headersSent) {
      const errObj = err as Record<string, unknown>;
      const status = (errObj["status"] as number) ?? 500;
      const upstreamMsg = (errObj["message"] as string) ?? "Upstream OpenAI error";
      res.status(status).json({ type: "error", error: { type: "api_error", message: upstreamMsg } });
    } else {
      res.end();
    }
  }
}

router.post("/messages", async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const originalModel = (body["model"] as string) || "claude-sonnet-4-6";
  const routingModel = normalizeModel(originalModel);

  if (isOpenAIModel(routingModel)) {
    await handleOpenAIViaAnthropicFormat(originalModel, body, req, res);
  } else {
    await handleAnthropicRoute(originalModel, body, req, res);
  }
});

export default router;
