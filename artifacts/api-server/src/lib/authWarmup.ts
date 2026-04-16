import { anthropic } from "@workspace/integrations-anthropic-ai";
import { logger } from "./logger.js";

export let authReady = false;

const WARMUP_MODELS = [
  "claude-haiku-4-5-20251001",
  "claude-sonnet-4-5",
  "claude-opus-4-6",
];

async function warmupModel(model: string, maxWaitMs: number): Promise<void> {
  const start = Date.now();
  let attempt = 0;

  while (Date.now() - start < maxWaitMs) {
    attempt++;
    try {
      await anthropic.messages.create({
        model,
        max_tokens: 1,
        messages: [{ role: "user", content: "." }],
      });
      logger.info({ model, attempt, elapsedMs: Date.now() - start }, "Model auth warmed up");
      return;
    } catch (err: unknown) {
      const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
      if (msg.includes("auth_unavailable")) {
        const delay = Math.min(5000 * attempt, 20_000);
        logger.warn({ model, attempt, delay, elapsedMs: Date.now() - start }, "Auth unavailable, retrying...");
        await new Promise((r) => setTimeout(r, delay));
      } else {
        // Any non-auth error (e.g. 400 invalid_request) means auth IS working for this model
        logger.info({ model, attempt, elapsedMs: Date.now() - start, err: msg }, "Model auth ready (non-auth error)");
        return;
      }
    }
  }

  logger.error({ model, maxWaitMs }, "Model auth warmup timed out");
}

export async function warmupAnthropicAuth(maxWaitMs = 120_000): Promise<void> {
  // Warm up all models concurrently so cold-start auth is ready for any model.
  const results = await Promise.allSettled(
    WARMUP_MODELS.map((model) => warmupModel(model, maxWaitMs))
  );

  const failed = results.filter((r) => r.status === "rejected").length;
  const succeeded = results.length - failed;

  authReady = succeeded > 0;
  logger.info({ succeeded, failed, total: results.length }, "Anthropic auth warmup complete");
}
