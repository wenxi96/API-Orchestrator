import { Request, Response, NextFunction } from "express";

// Simple in-memory sliding-window rate limiter.
// Keyed by client IP. Resets every WINDOW_MS. Defends against PROXY_API_KEY
// leaks burning through the entire upstream Replit AI Integration quota.

const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = 200;

type Bucket = { count: number; resetAt: number };
const buckets = new Map<string, Bucket>();

// Periodic GC so the map doesn't grow unbounded
setInterval(() => {
  const now = Date.now();
  for (const [ip, b] of buckets) {
    if (b.resetAt <= now) buckets.delete(ip);
  }
}, WINDOW_MS).unref();

function clientIp(req: Request): string {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string") return forwarded.split(",")[0]!.trim();
  return req.socket.remoteAddress ?? "unknown";
}

export function rateLimit(req: Request, res: Response, next: NextFunction): void {
  const ip = clientIp(req);
  const now = Date.now();
  const b = buckets.get(ip);

  if (!b || b.resetAt <= now) {
    buckets.set(ip, { count: 1, resetAt: now + WINDOW_MS });
    next();
    return;
  }

  b.count += 1;
  if (b.count > MAX_PER_WINDOW) {
    const retryAfter = Math.ceil((b.resetAt - now) / 1000);
    res.setHeader("Retry-After", String(retryAfter));
    res.status(429).json({
      error: {
        message: `Rate limit exceeded: ${MAX_PER_WINDOW} requests per minute. Retry after ${retryAfter}s.`,
        type: "rate_limit_error",
      },
    });
    return;
  }

  next();
}
