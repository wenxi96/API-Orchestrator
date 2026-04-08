import { Router, type IRouter, Request, Response } from "express";
import { anthropic } from "@workspace/integrations-anthropic-ai";
import { openai } from "@workspace/integrations-openai-ai-server";
import { proxyAuth } from "../../middlewares/proxyAuth.js";

const router: IRouter = Router();

router.use(proxyAuth);

type AnthropicMessage = { role: "user" | "assistant"; content: string };

function isOpenAIModel(model: string): boolean {
  return model.startsWith("gpt") || model.startsWith("o3") || model.startsWith("o4");
}

async function handleAnthropicRoute(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as AnthropicMessage[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 8192;
  const system = body["system"] as string | undefined;
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const anthropicStream = anthropic.messages.stream({
        model,
        max_tokens: maxTokens,
        messages,
        ...(system ? { system } : {}),
        ...(temperature !== undefined ? { temperature } : {}),
      });

      for await (const event of anthropicStream) {
        res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
      res.end();
    } else {
      const message = await anthropic.messages.create({
        model,
        max_tokens: maxTokens,
        messages,
        ...(system ? { system } : {}),
        ...(temperature !== undefined ? { temperature } : {}),
      });
      res.json(message);
    }
  } catch (err: unknown) {
    req.log.error({ err }, "Anthropic messages error");
    if (!res.headersSent) {
      res.status(500).json({ type: "error", error: { type: "api_error", message: "Upstream Anthropic error" } });
    } else {
      res.end();
    }
  }
}

async function handleOpenAIViaAnthropicFormat(model: string, body: Record<string, unknown>, req: Request, res: Response): Promise<void> {
  const messages = (body["messages"] as AnthropicMessage[]) ?? [];
  const maxTokens = (body["max_tokens"] as number) ?? 8192;
  const system = body["system"] as string | undefined;
  const temperature = body["temperature"] as number | undefined;
  const stream = body["stream"] === true;

  const openaiMessages: { role: "system" | "user" | "assistant"; content: string }[] = [];
  if (system) {
    openaiMessages.push({ role: "system", content: system });
  }
  for (const m of messages) {
    openaiMessages.push({ role: m.role, content: m.content });
  }

  const messageId = `msg_proxy_${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  try {
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      const startEvent = {
        type: "message_start",
        message: {
          id: messageId,
          type: "message",
          role: "assistant",
          content: [],
          model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      };
      res.write(`event: message_start\ndata: ${JSON.stringify(startEvent)}\n\n`);
      res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`);
      res.write(`event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`);

      const completionStream = await openai.chat.completions.create({
        model,
        messages: openaiMessages,
        stream: true,
        ...(temperature !== undefined ? { temperature } : {}),
        max_completion_tokens: maxTokens,
      });

      let outputTokens = 0;
      let stopReason = "end_turn";

      for await (const chunk of completionStream) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          outputTokens += 1;
          const deltaEvent = {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: content },
          };
          res.write(`event: content_block_delta\ndata: ${JSON.stringify(deltaEvent)}\n\n`);
        }
        const finishReason = chunk.choices[0]?.finish_reason;
        if (finishReason) {
          stopReason = finishReason === "stop" ? "end_turn" : finishReason;
        }
        if (chunk.usage) {
          outputTokens = chunk.usage.completion_tokens ?? outputTokens;
        }
      }

      res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
      res.write(`event: message_delta\ndata: ${JSON.stringify({ type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } })}\n\n`);
      res.write(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
      res.end();
    } else {
      const completion = await openai.chat.completions.create({
        model,
        messages: openaiMessages,
        stream: false,
        ...(temperature !== undefined ? { temperature } : {}),
        max_completion_tokens: maxTokens,
      });

      const text = completion.choices[0]?.message?.content ?? "";
      const finishReason = completion.choices[0]?.finish_reason ?? "stop";

      res.json({
        id: messageId,
        type: "message",
        role: "assistant",
        content: [{ type: "text", text }],
        model,
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
  const model = (body["model"] as string) || "claude-sonnet-4-6";

  if (isOpenAIModel(model)) {
    await handleOpenAIViaAnthropicFormat(model, body, req, res);
  } else {
    await handleAnthropicRoute(model, body, req, res);
  }
});

export default router;
