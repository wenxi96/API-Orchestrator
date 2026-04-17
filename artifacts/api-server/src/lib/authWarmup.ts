import { anthropic } from "@workspace/integrations-anthropic-ai";
import { openai } from "@workspace/integrations-openai-ai-server";
import { logger } from "./logger.js";

export let authReady = false;

const WARMUP_CLAUDE = [
  "claude-haiku-4-5-20251001",
  "claude-sonnet-4-5",
  "claude-sonnet-4-6",
  "claude-opus-4-5",
  "claude-opus-4-6",
  "claude-opus-4-7",
];

const WARMUP_OPENAI = [
  "gpt-5.2",
  "gpt-5.4",
];

const KEEPALIVE_INTERVAL_MS = 25 * 60 * 1000;

async function pingClaude(model: string): Promise<void> {
  try {
    await anthropic.messages.create({
      model, max_tokens: 1,
      messages: [{ role: "user", content: "." }],
    });
  } catch (err: unknown) {
    const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
    if (msg.includes("auth_unavailable")) {
      logger.warn({ model, err: msg }, "Keepalive: Claude auth expired, re-warming...");
      await warmupClaude(model, 60_000).catch(() => {});
    }
  }
}

async function pingOpenAI(model: string): Promise<void> {
  try {
    await openai.chat.completions.create({
      model, max_completion_tokens: 1,
      messages: [{ role: "user", content: "." }],
    });
  } catch (err: unknown) {
    const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
    if (msg.includes("auth_unavailable")) {
      logger.warn({ model, err: msg }, "Keepalive: OpenAI auth expired, re-warming...");
      await warmupOpenAI(model, 60_000).catch(() => {});
    }
  }
}

async function warmupClaude(model: string, maxWaitMs: number): Promise<void> {
  const start = Date.now();
  let attempt = 0;
  while (Date.now() - start < maxWaitMs) {
    attempt++;
    try {
      await anthropic.messages.create({
        model, max_tokens: 1,
        messages: [{ role: "user", content: "." }],
      });
      logger.info({ model, attempt, elapsedMs: Date.now() - start }, "Claude auth warmed up");
      return;
    } catch (err: unknown) {
      const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
      if (msg.includes("auth_unavailable")) {
        const delay = Math.min(5000 * attempt, 20_000);
        logger.warn({ model, attempt, delay }, "Auth unavailable, retrying...");
        await new Promise((r) => setTimeout(r, delay));
      } else {
        logger.info({ model, attempt, err: msg }, "Claude auth ready (non-auth error)");
        return;
      }
    }
  }
  logger.error({ model, maxWaitMs }, "Claude auth warmup timed out");
}

async function warmupOpenAI(model: string, maxWaitMs: number): Promise<void> {
  const start = Date.now();
  let attempt = 0;
  while (Date.now() - start < maxWaitMs) {
    attempt++;
    try {
      await openai.chat.completions.create({
        model, max_completion_tokens: 1,
        messages: [{ role: "user", content: "." }],
      });
      logger.info({ model, attempt, elapsedMs: Date.now() - start }, "OpenAI auth warmed up");
      return;
    } catch (err: unknown) {
      const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
      if (msg.includes("auth_unavailable")) {
        const delay = Math.min(5000 * attempt, 20_000);
        logger.warn({ model, attempt, delay }, "OpenAI auth unavailable, retrying...");
        await new Promise((r) => setTimeout(r, delay));
      } else {
        logger.info({ model, attempt, err: msg }, "OpenAI auth ready (non-auth error)");
        return;
      }
    }
  }
  logger.error({ model, maxWaitMs }, "OpenAI auth warmup timed out");
}

export async function warmupAllAuth(maxWaitMs = 120_000): Promise<void> {
  const claudeResults = await Promise.allSettled(
    WARMUP_CLAUDE.map((m) => warmupClaude(m, maxWaitMs))
  );
  const openaiResults = await Promise.allSettled(
    WARMUP_OPENAI.map((m) => warmupOpenAI(m, maxWaitMs))
  );

  const total = claudeResults.length + openaiResults.length;
  const failed = [...claudeResults, ...openaiResults].filter((r) => r.status === "rejected").length;
  const succeeded = total - failed;

  authReady = succeeded > 0;
  logger.info({ succeeded, failed, total }, "Auth warmup complete");
}

export function startKeepalive(): void {
  setInterval(() => {
    logger.debug("Running auth keepalive pings...");
    for (const model of WARMUP_CLAUDE) pingClaude(model).catch(() => {});
    for (const model of WARMUP_OPENAI) pingOpenAI(model).catch(() => {});
  }, KEEPALIVE_INTERVAL_MS);
  logger.info({ intervalMin: KEEPALIVE_INTERVAL_MS / 60_000 }, "Auth keepalive started");
}
