import app from "./app";
import { logger } from "./lib/logger";
import { warmupAllAuth, startKeepalive } from "./lib/authWarmup.js";

// ── Required secrets check ────────────────────────────────────────────────────
// PROXY_API_KEY must be set before the server starts. Without it every proxied
// request will be rejected. Fail loudly here so the deployment health-check
// catches the missing secret immediately and the operator sees a clear message
// in the logs rather than a cryptic runtime 500 for each request.
if (!process.env["PROXY_API_KEY"]) {
  logger.error(
    "PROXY_API_KEY is not set. " +
    "Please add it as a Secret in the Replit Secrets panel (key: PROXY_API_KEY) " +
    "before deploying. The server will not start without it."
  );
  process.exit(1);
}
// ─────────────────────────────────────────────────────────────────────────────

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// Complete auth warmup for ALL models BEFORE opening the port.
// This ensures Replit's health check only passes after every model is ready,
// so the very first routed request will never hit auth_unavailable.
logger.info("Starting auth warmup before accepting connections...");
warmupAllAuth()
  .then(() => {
    app.listen(port, (err) => {
      if (err) {
        logger.error({ err }, "Error listening on port");
        process.exit(1);
      }
      logger.info({ port }, "Server listening");

      // Keep auth tokens alive with periodic pings so they don't expire
      // during long-running idle periods between cold starts.
      startKeepalive();
    });
  })
  .catch((e) => {
    logger.error({ err: e }, "Auth warmup failed — starting server anyway");
    app.listen(port, () => {
      logger.info({ port }, "Server listening (warmup failed)");
      startKeepalive();
    });
  });
