import type { Request } from "express";
import { randomUUID } from "node:crypto";

export function normalizeModel(model: string): string {
  return model.replace(/-\d{8}$/, "");
}

export function parseBetaHeaders(req: Request): string[] {
  const raw = req.headers["anthropic-beta"];
  if (!raw) return [];
  const value = Array.isArray(raw) ? raw.join(",") : raw;
  return value.split(",").map((s) => s.trim()).filter(Boolean);
}

export function generateId(prefix: string): string {
  return `${prefix}_${randomUUID().replace(/-/g, "")}`;
}

export function isNetworkError(msg: string): boolean {
  return msg.includes("i/o timeout")
    || msg.includes("dial tcp")
    || msg.includes("EOF")
    || msg.includes("connection reset")
    || msg.includes("ECONNRESET");
}

export function isTransientError(err: unknown): boolean {
  const e = err as Record<string, unknown>;
  const msg = (e["message"] as string) ?? "";
  const status = e["status"] as number | undefined;
  // auth_unavailable is a multi-minute Replit Integration cooldown — pass through
  if (msg.includes("auth_unavailable")) return false;
  if (status === 429) return true;
  if (isNetworkError(msg)) return true;
  return false;
}

function retryDelay(err: unknown, attempt: number): number {
  const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
  if (isNetworkError(msg)) return 1000;
  return 1500 * attempt;
}

function maxAttempts(err: unknown, defaultMax: number): number {
  const msg = ((err as Record<string, unknown>)["message"] as string) ?? "";
  if (isNetworkError(msg)) return 3;
  return defaultMax;
}

export async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const max = maxAttempts(err, attempts);
      if (!isTransientError(err) || attempt >= max) throw err;
      await new Promise((r) => setTimeout(r, retryDelay(err, attempt)));
    }
  }
  throw lastErr;
}

// ── stop_reason / finish_reason mapping ───────────────────────────────────────
// Anthropic stop_reason values: end_turn | max_tokens | stop_sequence | tool_use | pause_turn | refusal
// OpenAI finish_reason values:  stop | length | tool_calls | content_filter | function_call

export function mapAnthropicStopToOpenAI(reason: string | null | undefined): string {
  switch (reason) {
    case "end_turn":
    case "stop_sequence":
    case "pause_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
    default:
      return "stop";
  }
}

export function mapOpenAIFinishToAnthropic(reason: string | null | undefined): string {
  switch (reason) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
    default:
      return "end_turn";
  }
}

// ── upstream cancellation ─────────────────────────────────────────────────────
// Wire client disconnect → AbortController so an interrupted client doesn't
// keep burning upstream tokens to completion.
export function abortOnClientClose(req: Request): AbortController {
  const controller = new AbortController();
  req.on("close", () => {
    if (!req.complete) controller.abort();
  });
  return controller;
}
