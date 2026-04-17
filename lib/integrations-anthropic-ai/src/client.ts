import Anthropic from "@anthropic-ai/sdk";

const anthropicBaseUrl = process.env.AI_INTEGRATIONS_ANTHROPIC_BASE_URL;
const anthropicApiKey = process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY;

if (!anthropicBaseUrl) {
  throw new Error(
    "AI_INTEGRATIONS_ANTHROPIC_BASE_URL must be set. Did you forget to provision the Anthropic AI integration?",
  );
}

if (!anthropicApiKey) {
  throw new Error(
    "AI_INTEGRATIONS_ANTHROPIC_API_KEY must be set. Did you forget to provision the Anthropic AI integration?",
  );
}

// Self-loop guard: if the base URL points back to a *.replit.app domain,
// the proxy would forward requests to itself causing i/o timeouts.
// This indicates AI_INTEGRATIONS_ANTHROPIC_BASE_URL was set incorrectly.
if (anthropicBaseUrl.includes(".replit.app")) {
  const msg =
    `AI_INTEGRATIONS_ANTHROPIC_BASE_URL is set to "${anthropicBaseUrl}" which points back to a Replit app domain. ` +
    "This causes a self-referential request loop. " +
    "Re-provision the Anthropic AI integration to get the correct upstream URL.";
  // Log to stderr so it is visible in deployment logs.
  process.stderr.write(`[FATAL] ${msg}\n`);
  throw new Error(msg);
}

// Log the effective base URL at startup (host only, no credentials).
try {
  const { hostname } = new URL(anthropicBaseUrl);
  process.stdout.write(`[integrations-anthropic-ai] baseURL host: ${hostname}\n`);
} catch {
  // non-critical
}

export const anthropic = new Anthropic({
  apiKey: anthropicApiKey,
  baseURL: anthropicBaseUrl,
});
