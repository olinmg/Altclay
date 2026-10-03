import type { OpenAIModelId } from "./types";

interface AzureEnrichResult {
  data: Record<string, string>;
  inputTokens: number;
  outputTokens: number;
}

export interface AzureConfig {
  endpoint: string;
  deployment: string;
  key: string;
}

/** Thrown on HTTP 429 so callers can back off. `retryAfter` is in seconds if Azure provided it. */
export class RateLimitError extends Error {
  retryAfter?: number;
  constructor(message: string, retryAfter?: number) {
    super(message);
    this.name = "RateLimitError";
    this.retryAfter = retryAfter;
  }
}

export async function enrichRowAzure(
  config: AzureConfig,
  modelId: OpenAIModelId,
  prompt: string,
  useWebSearch: boolean = true
): Promise<AzureEnrichResult> {
  const res = await fetch("/api/enrich", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: "azure",
      // The API key field carries the Azure config as a JSON blob so the
      // request shape matches every other provider (see route.ts).
      apiKey: JSON.stringify(config),
      modelId,
      prompt,
      useWebSearch,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Request failed" }));
    if (res.status === 429 || err.rateLimited) {
      throw new RateLimitError(err.error || "Rate limit exceeded", err.retryAfter);
    }
    throw new Error(err.error || `API error: ${res.status}`);
  }

  const result = await res.json();
  return {
    data: result.data,
    inputTokens: result.inputTokens || 0,
    outputTokens: result.outputTokens || 0,
  };
}
