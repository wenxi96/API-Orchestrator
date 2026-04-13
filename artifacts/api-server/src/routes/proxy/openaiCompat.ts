import { Router, type IRouter, Request, Response } from "express";
import { openai } from "@workspace/integrations-openai-ai-server";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { proxyAuth } from "../../middlewares/proxyAuth.js";

const router: IRouter = Router();

router.use(proxyAuth);

const OPENAI_MODELS = [
  { id: "gpt-5.2", object: "model", owned_by: "openai" },
  { id: "gpt-5.1", object: "model", owned_by: "openai" },
  { id: "gpt-5", object: "model", owned_by: "openai" },
  { id: "gpt-5-mini", object: "model", owned_by: "openai" },
  { id: "gpt-5-nano", object: "model", owned_by: "openai" },
  { id: "o4-mini", object: "model", owned_by: "openai" },
  { id: "o3", object: "model", owned_by: "openai" },
];

const ANTHROPIC_MODELS = [
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
  return model.startsWith("o3") || model.startsWith("o4");
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

      const completionStream = await openai.chat.completions.create({
        model,
        messages: chatMessages,
        stream: true,
        stream_options: { include_usage: true },
        ...extraParams,
      } as Parameters<typeof openai.chat.completions.create>[0]);

      for await (const chunk of completionStream) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const completion = await openai.chat.completions.create({
        model,
        messages: chatMessages,
        stream: false,
        ...extraParams,
      } as Parameters<typeof openai.chat.completions.create>[0]);
      res.json(completion);
    }
  } catch (err: unknown) {
    req.log.error({ err }, "OpenAI completion error");
    if (!res.headersSent) {
      res.status(500).json({ error: { message: "Upstream OpenAI error", type: "api_error" } });
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

  const systemMessages = messages.filter((m) => m.role === "system");
  const systemPrompt = systemMessages.map((m) => m.content ?? "").join("\n");
  const anthropicMessages = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content ?? "",
    }));

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

      const anthropicStream = anthropic.messages.stream(
        baseParams as Parameters<typeof anthropic.messages.stream>[0]
      );

      let thinkingBuffer = "";

      for await (const event of anthropicStream) {
        if (event.type === "content_block_delta") {
          if (event.delta.type === "thinking_delta") {
            thinkingBuffer += event.delta.thinking;
            const chunk = {
              id: requestId,
              object: "chat.completion.chunk",
              created,
              model,
              choices: [{
                index: 0,
                delta: { reasoning_content: event.delta.thinking },
                finish_reason: null,
              }],
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          } else if (event.delta.type === "text_delta") {
            const chunk = {
              id: requestId,
              object: "chat.completion.chunk",
              created,
              model,
              choices: [{
                index: 0,
                delta: { content: event.delta.text },
                finish_reason: null,
              }],
            };
            res.write(`data: ${JSON.stringify(chunk)}\n\n`);
          }
        } else if (event.type === "message_delta" && event.delta.stop_reason) {
          const chunk = {
            id: requestId,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [{
              index: 0,
              delta: {},
              finish_reason: event.delta.stop_reason === "end_turn" ? "stop" : event.delta.stop_reason,
            }],
          };
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } else {
      const message = await anthropic.messages.create(
        baseParams as Parameters<typeof anthropic.messages.create>[0]
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
      res.status(500).json({ error: { message: "Upstream Anthropic error", type: "api_error" } });
    } else {
      res.end();
    }
  }
}

router.post("/chat/completions", async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;
  const model = normalizeModel((body["model"] as string) || "gpt-5.2");

  if (isClaudeModel(model)) {
    await handleClaudeViaOpenAIFormat(model, body, req, res);
  } else {
    await handleOpenAIRoute(model, body, req, res);
  }
});

export default router;
