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

// ── auth_unavailable detection & model fallback ───────────────────────────────
// Replit's managed Claude integration can enter a cooldown ("auth_unavailable")
// when a specific model gets rate-limited. We auto-fallback to a related model
// and surface a friendly bilingual message if the entire chain is exhausted.

// Structured detection: upstream may surface auth_unavailable through any of
// `message`, `code`, `error.code`, `error.type`, or HTTP status hints. Pure
// string matching on `message` misses errors that put the marker elsewhere.
export function isAuthUnavailableError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as Record<string, unknown>;
  const inner = (e["error"] as Record<string, unknown> | undefined) ?? {};
  // Replit Integration proxy nests the marker as `{ error: { error: { message, type, code } } }`
  // when the SDK parses the upstream JSON body and exposes it on `err.error`.
  // Walk one level deeper as well so we catch that shape.
  const innerInner = (inner["error"] as Record<string, unknown> | undefined) ?? {};
  const fields = [
    e["message"],
    e["code"],
    e["type"],
    inner["message"],
    inner["code"],
    inner["type"],
    innerInner["message"],
    innerInner["code"],
    innerInner["type"],
  ];
  for (const f of fields) {
    if (typeof f === "string" && f.includes("auth_unavailable")) return true;
  }
  // Some SDKs nest the original payload under `response.data` / `cause`.
  const response = e["response"] as Record<string, unknown> | undefined;
  if (response) {
    const data = response["data"];
    if (typeof data === "string" && data.includes("auth_unavailable")) return true;
    if (data && typeof data === "object") {
      const d = data as Record<string, unknown>;
      const dErr = (d["error"] as Record<string, unknown> | undefined) ?? {};
      for (const f of [d["message"], d["code"], dErr["message"], dErr["code"], dErr["type"]]) {
        if (typeof f === "string" && f.includes("auth_unavailable")) return true;
      }
    }
  }
  const cause = e["cause"];
  if (cause && cause !== err) return isAuthUnavailableError(cause);
  return false;
}

// Models to try in order when the primary model returns auth_unavailable.
// Picked to preserve capability tier when possible (opus → opus → sonnet).
// Lookups are normalized (date suffix stripped) so aliases like
// `claude-sonnet-4-5-20251001` resolve to the `claude-sonnet-4-5` chain.
const MODEL_FALLBACK_CHAIN_RAW: Record<string, string[]> = {
  "claude-opus-4-7": ["claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-6"],
  "claude-opus-4-6": ["claude-opus-4-5", "claude-sonnet-4-6", "claude-sonnet-4-5"],
  "claude-opus-4-5": ["claude-sonnet-4-6", "claude-sonnet-4-5"],
  "claude-sonnet-4-6": ["claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
  "claude-sonnet-4-5": ["claude-haiku-4-5-20251001"],
};

export const MODEL_FALLBACK_CHAIN: Record<string, string[]> = MODEL_FALLBACK_CHAIN_RAW;

export function getFallbackChain(model: string): string[] {
  return MODEL_FALLBACK_CHAIN_RAW[normalizeModel(model)] ?? [];
}

export interface FallbackAttemptCtx {
  /** Per-attempt signal: aborts if the client disconnects but never carries
   *  abort state from a previous failed attempt. Use this for the upstream
   *  SDK call so that one attempt's cleanup can't poison the next. */
  signal: AbortSignal;
  /** True if this is a fallback (i.e. not the originally requested model). */
  isFallback: boolean;
}

/**
 * Try `fn` with the original model; on auth_unavailable, walk the fallback chain.
 * Each attempt receives a *fresh* AbortSignal linked to `parentSignal`, so abort
 * state never bleeds between attempts. If `parentSignal` is aborted (client
 * disconnect), the chain bails immediately and rethrows the last error.
 */
export async function withModelFallback<T>(
  originalModel: string,
  fn: (model: string, ctx: FallbackAttemptCtx) => Promise<T>,
  options: {
    parentSignal?: AbortSignal;
    onFallback?: (failed: string, next: string) => void;
  } = {},
): Promise<{ result: T; usedModel: string }> {
  const { parentSignal, onFallback } = options;
  const chain = [originalModel, ...getFallbackChain(originalModel)];
  let lastErr: unknown;

  for (let i = 0; i < chain.length; i++) {
    if (parentSignal?.aborted) {
      throw lastErr ?? new Error("client aborted before fallback could complete");
    }
    const model = chain[i] as string;

    // Fresh controller per attempt, forwarded from parent if present.
    const attemptController = new AbortController();
    const onParentAbort = (): void => attemptController.abort();
    if (parentSignal) {
      if (parentSignal.aborted) attemptController.abort();
      else parentSignal.addEventListener("abort", onParentAbort, { once: true });
    }

    try {
      const result = await fn(model, {
        signal: attemptController.signal,
        isFallback: i > 0,
      });
      return { result, usedModel: model };
    } catch (err) {
      lastErr = err;
      if (!isAuthUnavailableError(err)) throw err;
      const next = chain[i + 1];
      if (next && onFallback) onFallback(model, next);
    } finally {
      if (parentSignal) parentSignal.removeEventListener("abort", onParentAbort);
    }
  }
  throw lastErr;
}

// Friendly bilingual message for users when the upstream Claude auth is in
// cooldown. Includes which models were tried so the user knows what happened.
export function formatAuthUnavailableMessage(
  originalModel: string,
  triedModels: string[],
): string {
  const triedList = triedModels.join(", ");
  const tier = originalModel.includes("opus") ? "claude-sonnet-4-6 / claude-haiku-4-5"
    : originalModel.includes("sonnet") ? "claude-haiku-4-5"
    : "其他较小的模型";
  return (
    `模型 \`${originalModel}\` 当前不可用：上游 Claude 集成认证处于冷却状态（rate-limited / cooldown）。\n` +
    `已自动尝试备用模型 [${triedList}]，但全部因相同原因失败。\n\n` +
    `建议：\n` +
    `  1. 等待数分钟后重试（冷却通常会自动恢复）\n` +
    `  2. 手动切换到 ${tier}\n` +
    `  3. 减少单次请求的上下文长度，避免触发限流\n\n` +
    `--- English ---\n` +
    `Model \`${originalModel}\` is currently unavailable: upstream Claude integration auth is in cooldown (rate-limited).\n` +
    `Auto-fallback chain [${triedList}] all failed with the same reason.\n` +
    `Please wait a few minutes and retry, or switch to a smaller-tier model.`
  );
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
