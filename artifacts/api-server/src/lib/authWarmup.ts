import { anthropic } from "@workspace/integrations-anthropic-ai";
import { logger } from "./logger.js";

export let authReady = false;

const WARMUP_MODELS = [
  "claude-haiku-4-5-20251001",
  "claude-sonnet-4-5",
  "claude-sonnet-4-6",
  "claude-opus-4-5",
  "claude-opus-4-6",
];

// Ping every 25 minutes — well within Vertex AI's auth token window.
const KEEPALIVE_INTERVAL_MS = 25 * 60 * 1000;

async function pingModel(model: string): Promise<void> {
  try {
    await anthropic.messages.create({
      model,
      max_tokens: 1,
      messages: [{ role: "user", content: "." }],
    });
    logger.debug({ model }, "Keepalive ping succeeded");
  } catch (err: unknown) {
    const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
    // auth_unavailable on keepalive means the token expired — log as warning.
    // Any other error (400, 429) still means auth is alive.
    if (msg.includes("auth_unavailable")) {
      logger.warn({ model, err: msg }, "Keepalive: auth expired, re-warming...");
      await warmupModel(model, 60_000).catch(() => {});
    }
  }
}

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
        // Any non-auth error (e.g. 400 invalid_request) means auth IS working
        logger.info({ model, attempt, elapsedMs: Date.now() - start, err: msg }, "Model auth ready (non-auth error)");
        return;
      }
    }
  }

  logger.error({ model, maxWaitMs }, "Model auth warmup timed out");
}

export async function warmupAnthropicAuth(maxWaitMs = 120_000): Promise<void> {
  // Warm up all models concurrently.
  const results = await Promise.allSettled(
    WARMUP_MODELS.map((model) => warmupModel(model, maxWaitMs))
  );

  const failed = results.filter((r) => r.status === "rejected").length;
  const succeeded = results.length - failed;

  authReady = succeeded > 0;
  logger.info({ succeeded, failed, total: results.length }, "Anthropic auth warmup complete");
}

export function startKeepalive(): void {
  setInterval(() => {
    logger.debug("Running auth keepalive pings...");
    for (const model of WARMUP_MODELS) {
      pingModel(model).catch(() => {});
    }
  }, KEEPALIVE_INTERVAL_MS);

  logger.info({ intervalMin: KEEPALIVE_INTERVAL_MS / 60_000 }, "Auth keepalive started");
}
