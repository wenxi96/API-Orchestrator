import { useState, useRef, type ReactNode } from "react";

const MODELS = [
  { id: "gpt-5.2", label: "GPT-5.2", provider: "OpenAI", color: "text-emerald-400" },
  { id: "gpt-5-mini", label: "GPT-5 Mini", provider: "OpenAI", color: "text-emerald-400" },
  { id: "gpt-5-nano", label: "GPT-5 Nano", provider: "OpenAI", color: "text-emerald-400" },
  { id: "o4-mini", label: "o4-mini", provider: "OpenAI", color: "text-emerald-400" },
  { id: "claude-opus-4-6", label: "Claude Opus 4.6", provider: "Anthropic", color: "text-orange-400" },
  { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", provider: "Anthropic", color: "text-orange-400" },
  { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", provider: "Anthropic", color: "text-orange-400" },
];

const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

function Badge({ children, color }: { children: ReactNode; color: string }) {
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${color}`}>
      {children}
    </span>
  );
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="relative group">
      <pre className="code-block text-[13px] leading-relaxed">{code}</pre>
      <button
        onClick={copy}
        className="absolute top-3 right-3 px-2 py-1 text-xs rounded bg-secondary text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {copied ? "Copied!" : "Copy"}
      </button>
    </div>
  );
}

function Endpoint({ method, path, desc }: { method: string; path: string; desc: string }) {
  const colors: Record<string, string> = {
    GET: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    POST: "bg-blue-500/15 text-blue-400 border-blue-500/30",
  };
  return (
    <div className="flex items-start gap-3 py-3 border-b border-border last:border-0">
      <span className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${colors[method]}`}>
        {method}
      </span>
      <div className="min-w-0">
        <code className="text-sm font-mono text-foreground">{path}</code>
        <p className="text-sm text-muted-foreground mt-0.5">{desc}</p>
      </div>
    </div>
  );
}

function LiveTester() {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-5-nano");
  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const run = async () => {
    if (!apiKey.trim()) { setError("Enter your Proxy API Key"); return; }
    if (!prompt.trim()) { setError("Enter a message"); return; }
    setError(""); setResponse(""); setLoading(true);
    abortRef.current = new AbortController();
    try {
      const res = await fetch(`${BASE}/api/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` },
        body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], stream: true }),
        signal: abortRef.current.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as { error?: { message?: string } })?.error?.message ?? `HTTP ${res.status}`);
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;
          try {
            const json = JSON.parse(data);
            const content = json?.choices?.[0]?.delta?.content;
            if (content) setResponse(prev => prev + content);
          } catch { /* skip */ }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-muted-foreground mb-1.5 block">Proxy API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="Your PROXY_API_KEY"
            className="w-full bg-secondary border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
        <div>
          <label className="text-xs text-muted-foreground mb-1.5 block">Model</label>
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-secondary border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <optgroup label="OpenAI">
              {MODELS.filter(m => m.provider === "OpenAI").map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </optgroup>
            <optgroup label="Anthropic (via proxy)">
              {MODELS.filter(m => m.provider === "Anthropic").map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </optgroup>
          </select>
        </div>
      </div>
      <div>
        <label className="text-xs text-muted-foreground mb-1.5 block">Message</label>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={3}
          className="w-full bg-secondary border border-border rounded-md px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <button
        onClick={loading ? () => { abortRef.current?.abort(); setLoading(false); } : run}
        className="px-4 py-2 rounded-md text-sm font-medium bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
      >
        {loading ? "Stop" : "Send"}
      </button>
      {(loading || response) && (
        <div className="bg-muted rounded-lg p-4 text-sm font-mono whitespace-pre-wrap min-h-[4rem]">
          {response || <span className="text-muted-foreground animate-pulse">Streaming…</span>}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const domain = window.location.origin;
  const baseUrl = `${domain}/api/v1`;

  const openaiCode = `import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "YOUR_PROXY_API_KEY",
  baseURL: "${baseUrl}",
});

// GPT model
const gpt = await client.chat.completions.create({
  model: "gpt-5.2",
  messages: [{ role: "user", content: "Hello!" }],
});

// Claude model — same client!
const claude = await client.chat.completions.create({
  model: "claude-sonnet-4-6",
  messages: [{ role: "user", content: "Hello!" }],
});`;

  const anthropicCode = `import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: "YOUR_PROXY_API_KEY",
  baseURL: "${baseUrl}",
});

// Claude model
const claude = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello!" }],
});

// GPT model — same client!
const gpt = await client.messages.create({
  model: "gpt-5.2",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Hello!" }],
});`;

  const curlCode = `# OpenAI format
curl -X POST ${baseUrl}/chat/completions \\
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Hi"}]}'

# Anthropic format
curl -X POST ${baseUrl}/messages \\
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"gpt-5.2","max_tokens":512,"messages":[{"role":"user","content":"Hi"}]}'`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-4xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
              <span className="text-primary text-lg">⚡</span>
            </div>
            <span className="font-semibold text-lg">AI Proxy</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge color="border-emerald-500/40 text-emerald-400">OpenAI</Badge>
            <Badge color="border-orange-500/40 text-orange-400">Anthropic</Badge>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10 space-y-10">
        {/* Hero */}
        <section>
          <h1 className="text-3xl font-bold mb-3">Dual-Provider AI Proxy</h1>
          <p className="text-muted-foreground text-lg leading-relaxed max-w-2xl">
            A single API endpoint that gives you access to both OpenAI and Anthropic models.
            Use either SDK — the proxy handles routing and format translation automatically.
          </p>
        </section>

        {/* Base URL */}
        <section className="bg-card border border-border rounded-xl p-5">
          <p className="text-xs text-muted-foreground mb-2">BASE URL</p>
          <code className="text-primary font-mono text-sm">{baseUrl}</code>
          <p className="text-xs text-muted-foreground mt-2">
            All requests require: <code className="font-mono text-foreground">Authorization: Bearer YOUR_PROXY_API_KEY</code>
          </p>
        </section>

        {/* Endpoints */}
        <section>
          <h2 className="text-xl font-semibold mb-4">Endpoints</h2>
          <div className="bg-card border border-border rounded-xl px-5">
            <Endpoint method="GET" path="/v1/models" desc="List all available models from both providers" />
            <Endpoint method="POST" path="/v1/chat/completions" desc="OpenAI Chat Completions format — routes to OpenAI or Anthropic based on model name" />
            <Endpoint method="POST" path="/v1/messages" desc="Anthropic Messages format — routes to Anthropic or OpenAI based on model name" />
          </div>
        </section>

        {/* Models */}
        <section>
          <h2 className="text-xl font-semibold mb-4">Available Models</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-card border border-border rounded-xl p-4">
              <p className="text-xs font-medium text-emerald-400 mb-3">OPENAI</p>
              <div className="space-y-1.5">
                {MODELS.filter(m => m.provider === "OpenAI").map(m => (
                  <div key={m.id} className="flex items-center gap-2">
                    <code className="text-xs font-mono text-foreground">{m.id}</code>
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-card border border-border rounded-xl p-4">
              <p className="text-xs font-medium text-orange-400 mb-3">ANTHROPIC</p>
              <div className="space-y-1.5">
                {MODELS.filter(m => m.provider === "Anthropic").map(m => (
                  <div key={m.id} className="flex items-center gap-2">
                    <code className="text-xs font-mono text-foreground">{m.id}</code>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-3">
            Model names starting with <code className="font-mono text-orange-400">claude-</code> are routed to Anthropic.
            All others (<code className="font-mono text-emerald-400">gpt-*</code>, <code className="font-mono text-emerald-400">o3</code>, <code className="font-mono text-emerald-400">o4-mini</code>) are routed to OpenAI.
            Cross-provider requests are automatically translated.
          </p>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-xl font-semibold mb-4">Code Examples</h2>
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">OpenAI SDK</p>
              <CodeBlock code={openaiCode} />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">Anthropic SDK</p>
              <CodeBlock code={anthropicCode} />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">cURL</p>
              <CodeBlock code={curlCode} />
            </div>
          </div>
        </section>

        {/* Live Tester */}
        <section>
          <h2 className="text-xl font-semibold mb-1">Live Tester</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Try the proxy right here — uses streaming via the OpenAI Chat Completions format.
          </p>
          <div className="bg-card border border-border rounded-xl p-5">
            <LiveTester />
          </div>
        </section>
      </main>

      <footer className="border-t border-border mt-16">
        <div className="max-w-4xl mx-auto px-6 py-5 text-sm text-muted-foreground">
          Powered by Replit AI Integrations — no external API keys required.
        </div>
      </footer>
    </div>
  );
}
