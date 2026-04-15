import { anthropic } from "@workspace/integrations-anthropic-ai";
import { logger } from "./logger.js";

export let authReady = false;

export async function warmupAnthropicAuth(maxWaitMs = 120_000): Promise<void> {
  const start = Date.now();
  let attempt = 0;

  while (Date.now() - start < maxWaitMs) {
    attempt++;
    try {
      await anthropic.messages.create({
        model: "claude-haiku-4-5",
        max_tokens: 1,
        messages: [{ role: "user", content: "." }],
      });
      authReady = true;
      logger.info({ attempt, elapsedMs: Date.now() - start }, "Anthropic auth warmed up");
      return;
    } catch (err: unknown) {
      const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
      if (msg.includes("auth_unavailable")) {
        const delay = Math.min(5000 * attempt, 20_000);
        logger.warn({ attempt, delay, elapsedMs: Date.now() - start }, "Anthropic auth unavailable, retrying...");
        await new Promise((r) => setTimeout(r, delay));
      } else {
        // Any non-auth error (e.g. invalid_request, 400) means auth IS working
        authReady = true;
        logger.info({ attempt, elapsedMs: Date.now() - start, err: msg }, "Anthropic auth ready (non-auth error received)");
        return;
      }
    }
  }

  logger.error({ maxWaitMs }, "Anthropic auth warmup timed out — proceeding anyway");
  authReady = false;
}
