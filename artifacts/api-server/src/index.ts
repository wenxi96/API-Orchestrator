import app from "./app";
import { logger } from "./lib/logger";
import { warmupAnthropicAuth, startKeepalive } from "./lib/authWarmup.js";

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
warmupAnthropicAuth()
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
