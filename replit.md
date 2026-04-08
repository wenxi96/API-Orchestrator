# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Dual-Provider AI Reverse Proxy

The API server includes a fully functional OpenAI + Anthropic dual-compatible reverse proxy.

### Authentication

All proxy endpoints require: `Authorization: Bearer <PROXY_API_KEY>`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/models` | List all available models from both providers |
| POST | `/api/v1/chat/completions` | OpenAI Chat Completions format (auto-routes to OpenAI or Anthropic based on model name) |
| POST | `/api/v1/messages` | Anthropic Messages format (auto-routes to Anthropic or OpenAI based on model name) |

### Model Routing Logic

- Model names starting with `claude-` → routed to Anthropic (format translated if needed)
- All other model names (e.g. `gpt-*`, `o3`, `o4-mini`) → routed to OpenAI (format translated if needed)

### Available Models

**OpenAI:** `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `o4-mini`, `o3`

**Anthropic:** `claude-opus-4-6`, `claude-opus-4-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`

### Provider Setup

- Uses Replit AI Integrations (no user API keys needed — billed to Replit credits)
- `AI_INTEGRATIONS_OPENAI_BASE_URL` + `AI_INTEGRATIONS_OPENAI_API_KEY` auto-provisioned
- `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` + `AI_INTEGRATIONS_ANTHROPIC_API_KEY` auto-provisioned

### Source Files

- `artifacts/api-server/src/middlewares/proxyAuth.ts` — Bearer token auth middleware
- `artifacts/api-server/src/routes/proxy/openaiCompat.ts` — OpenAI-compatible endpoint + model routing
- `artifacts/api-server/src/routes/proxy/anthropicCompat.ts` — Anthropic-compatible endpoint + model routing

### Example Usage

**Using OpenAI SDK:**
```javascript
import OpenAI from "openai";
const client = new OpenAI({
  apiKey: "<PROXY_API_KEY>",
  baseURL: "https://your-domain.replit.app/api/v1"
});

// Use any model — gpt or claude — through the same client
const response = await client.chat.completions.create({
  model: "claude-sonnet-4-6",  // Claude model via OpenAI format!
  messages: [{ role: "user", content: "Hello!" }]
});
```

**Using Anthropic SDK:**
```javascript
import Anthropic from "@anthropic-ai/sdk";
const client = new Anthropic({
  apiKey: "<PROXY_API_KEY>",
  baseURL: "https://your-domain.replit.app/api/v1"
});

// Use any model — claude or gpt — through the same client
const response = await client.messages.create({
  model: "gpt-5.2",  // GPT model via Anthropic format!
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello!" }]
});
```
