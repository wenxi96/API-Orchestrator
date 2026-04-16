import app from "./app";
import { logger } from "./lib/logger";
import { warmupAnthropicAuth } from "./lib/authWarmup.js";

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

app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");

  // Warm up Anthropic auth in the background so the first real request
  // doesn't hit an auth_unavailable cold-start error.
  warmupAnthropicAuth().catch((e) => {
    logger.error({ err: e }, "Auth warmup failed unexpectedly");
  });
});
