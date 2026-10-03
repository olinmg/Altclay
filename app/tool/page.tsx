"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import Link from "next/link";
import type {
  Provider,
  ModelId,
  OutputColumn,
  ParsedFile,
  EnrichmentResult,
  AnthropicModelId,
  GeminiModelId,
  GrokModelId,
  OpenAIModelId,
} from "@/lib/types";
import {
  ANTHROPIC_MODELS,
  GEMINI_MODELS,
  GROK_MODELS,
  OPENAI_MODELS,
  MODEL_GUIDANCE,
  PRICING_LAST_UPDATED,
  ANTHROPIC_PRICING_URL,
  GEMINI_PRICING_URL,
  GROK_PRICING_URL,
  OPENAI_PRICING_URL,
  VERTEX_PRICING_URL,
  AZURE_PRICING_URL,
} from "@/lib/pricing";
import { buildPrompt, buildPromptTemplate } from "@/lib/promptTemplates";
import { estimateInputTokensPerRow, estimateOutputTokensPerRow, calculateCostRange, calculateCostFromActualTokens } from "@/lib/costEstimator";
import { parseFile, exportToFile } from "@/lib/fileParser";
import { enrichRowAnthropic } from "@/lib/anthropic";
import { enrichRowGemini } from "@/lib/gemini";
import { enrichRowGrok } from "@/lib/grok";
import { enrichRowOpenAI } from "@/lib/openai";
import { saveSession, loadSession, clearSession, hasMeaningfulWork, timeAgo, type SavedSession } from "@/lib/sessionStore";
import { enrichRowVertex } from "@/lib/vertex";
import { enrichRowAzure, RateLimitError } from "@/lib/azure";

/* ------------------------------------------------------------------ */
/*  Light-themed Helpers                                                */
/* ------------------------------------------------------------------ */

function Card({ children, className = "", glow = false }: { children: React.ReactNode; className?: string; glow?: boolean }) {
  return (
    <div className={`rounded-2xl border bg-white p-5 shadow-sm transition-all ${glow ? "border-zinc-300 shadow-md" : "border-zinc-200"} ${className}`}>
      {children}
    </div>
  );
}

function StepHeader({ num, title, subtitle, done, active }: { num: number; title: string; subtitle?: string; done: boolean; active: boolean }) {
  return (
    <div className="mb-4 flex items-start gap-3">
      <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold transition-all ${done ? "bg-emerald-500 text-white shadow-sm" : active ? "bg-zinc-900 text-white shadow-sm" : "bg-zinc-100 text-zinc-400"}`}>
        {done ? (
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>
        ) : num}
      </div>
      <div>
        <h3 className="text-sm font-semibold text-zinc-900">{title}</h3>
        {subtitle && <p className="text-[11px] text-zinc-500">{subtitle}</p>}
      </div>
    </div>
  );
}

function TrustBadge({ text }: { text: string }) {
  return (
    <div className="mt-3 flex items-start gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-700">
      <svg className="mt-0.5 h-3 w-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" /></svg>
      {text}
    </div>
  );
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="group relative ml-1 inline-flex cursor-help">
      <svg className="h-3.5 w-3.5 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" /></svg>
      <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 w-56 -translate-x-1/2 rounded-lg bg-zinc-900 px-3 py-2 text-[11px] leading-relaxed text-zinc-100 opacity-0 shadow-xl transition-opacity group-hover:opacity-100">
        {text}
      </span>
    </span>
  );
}

const SPEED_COLORS = { fast: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200", medium: "bg-amber-50 text-amber-700 ring-1 ring-amber-200", slow: "bg-red-50 text-red-700 ring-1 ring-red-200" };
const QUALITY_COLORS = { good: "bg-zinc-50 text-zinc-600 ring-1 ring-zinc-200", great: "bg-blue-50 text-blue-700 ring-1 ring-blue-200", best: "bg-purple-50 text-purple-700 ring-1 ring-purple-200" };

/* ------------------------------------------------------------------ */
/*  Prompt templates                                                   */
/* ------------------------------------------------------------------ */

const PROMPT_TEMPLATES = [
  { label: "Company research", description: "Find the CEO name, total funding raised, employee count, and a brief company description" },
  { label: "Job postings", description: "Find the number of open job postings, most common roles being hired, and hiring page URL" },
  { label: "Recent news", description: "Find the most recent news headline, news date, and a brief summary of the article" },
  { label: "Tech stack", description: "Find the primary programming languages, cloud provider, and key technologies used" },
  { label: "University info", description: "Find the university ranking, acceptance rate, annual tuition, and notable alumni" },
];

/* ------------------------------------------------------------------ */
/*  Smart column detection                                             */
/* ------------------------------------------------------------------ */

const SMART_COLUMN_PATTERNS = [
  /^(company|organization|org|business|brand)[\s_-]*(name)?$/i,
  /^(domain|website|url|site|web)[\s_-]*(name|url)?$/i,
  /^(name|full[\s_-]*name|person[\s_-]*name)$/i,
  /^(email|e-mail)[\s_-]*(address)?$/i,
  /^(linkedin|twitter|x|github)[\s_-]*(url|link|profile)?$/i,
  /^(ticker|symbol|stock)[\s_-]*(symbol)?$/i,
  /^(university|school|college|institution)[\s_-]*(name)?$/i,
  /^(country|city|state|location|address)$/i,
  /^(product|app|service|tool)[\s_-]*(name)?$/i,
];

/** Human-friendly ETA: "45s", "3m 20s", "1h 5m". */
function formatEta(totalSeconds: number): string {
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

function detectSmartColumns(columns: string[]): string[] {
  const matches = columns.filter((col) =>
    SMART_COLUMN_PATTERNS.some((pattern) => pattern.test(col.trim()))
  );
  return matches.length > 0 ? matches : columns.length > 0 ? [columns[0]] : [];
}

function detectOutputColumns(description: string): OutputColumn[] {
  if (!description.trim()) return [];
  const cleaned = description
    .replace(/find\s+(out\s+)?(the\s+)?/gi, "")
    .replace(/get\s+(me\s+)?(the\s+)?/gi, "")
    .replace(/look\s+up\s+(the\s+)?/gi, "")
    .replace(/research\s+(the\s+)?/gi, "")
    .replace(/for\s+each\s+\w+/gi, "")
    .replace(/of\s+each\s+\w+/gi, "")
    .replace(/for\s+every\s+\w+/gi, "")
    .replace(/per\s+\w+/gi, "")
    .trim();

  const parts = cleaned
    .split(/,\s*|\s+and\s+/i)
    .map((p) => p.replace(/^(the|their|its|a|an)\s+/i, "").trim())
    .filter((p) => p.length > 1 && p.length < 60 && !p.includes("."));

  if (parts.length === 0) return [];

  return parts.map((label) => {
    const key = label.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    return { key, label };
  }).filter((c) => c.key.length > 0);
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function ToolPage() {
  /* ---- state ---- */
  const [file, setFile] = useState<ParsedFile | null>(null);
  const [fileError, setFileError] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [enrichmentDescription, setEnrichmentDescription] = useState("");
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [outputColumns, setOutputColumns] = useState<OutputColumn[]>([]);
  const [newColumnName, setNewColumnName] = useState("");
  const [autoDetected, setAutoDetected] = useState(false);
  const [showAllColumns, setShowAllColumns] = useState(false);
  const [columnsAutoSelected, setColumnsAutoSelected] = useState(false);

  const [provider, setProvider] = useState<Provider>("gemini");
  const [modelId, setModelId] = useState<ModelId>("gemini-3.1-pro-preview");

  const [apiKey, setApiKey] = useState("");
  // Azure OpenAI needs an endpoint + deployment name alongside the key.
  // These two are safe to persist (not secret); the key is never saved.
  const [azureEndpoint, setAzureEndpoint] = useState("");
  const [azureDeployment, setAzureDeployment] = useState("");
  const [keyValid, setKeyValid] = useState(false);
  const [validating, setValidating] = useState(false);
  const [keyError, setKeyError] = useState("");
  const [keyWarning, setKeyWarning] = useState("");

  const [testRunning, setTestRunning] = useState(false);
  const [testResults, setTestResults] = useState<EnrichmentResult[]>([]);
  const [testDone, setTestDone] = useState(false);
  const [fullRunning, setFullRunning] = useState(false);
  const [fullPaused, setFullPaused] = useState(false);
  const [fullCompleted, setFullCompleted] = useState(0);
  const [fullFailed, setFullFailed] = useState(0);
  const [fullResults, setFullResults] = useState<EnrichmentResult[]>([]);
  const [fullDone, setFullDone] = useState(false);
  const pauseRef = useRef(false);
  const stopRef = useRef(false);

  // Rate-limit backoff: timestamp (ms) when the soonest paused request resumes.
  const [rateLimitResetAt, setRateLimitResetAt] = useState<number | null>(null);
  const [rlCountdown, setRlCountdown] = useState(0); // seconds remaining, for display
  // Shared gate across all workers: no request fires until Date.now() passes this.
  // A 429 from ANY worker pushes this out, pausing the whole fleet together.
  const rateLimitGateRef = useRef(0);
  // Shared, adaptive backoff (ms). Grows on every 429, decays on success — so the
  // fleet converges to the deployment's real throughput instead of resetting to
  // the floor on each new row. Persists across rows and workers.
  const rlBackoffRef = useRef(0);
  const RL_BACKOFF_FLOOR = 500;   // first backoff step
  const RL_BACKOFF_CAP = 60_000;  // never wait longer than this between tries

  // ETA: rolling window of the last N per-request latencies (ms). Averaged and
  // extrapolated over the remaining rows (÷ worker count) to estimate time left.
  const reqDurationsRef = useRef<number[]>([]);
  const ETA_WINDOW = 40;
  // Concurrency: how many rows are enriched in parallel. The slider edits a DRAFT
  // value; nothing changes until the user presses "Apply". On apply we commit to
  // `appliedConcurrency`, whose ref mirror the in-flight worker pool reads live —
  // so the running batch grows/shrinks to match without a restart (see runFull()).
  const MAX_CONCURRENCY = 25;
  const [concurrency, setConcurrency] = useState(5);          // draft (slider position)
  const [appliedConcurrency, setAppliedConcurrency] = useState(5); // committed value
  const concurrencyRef = useRef(5);
  useEffect(() => { concurrencyRef.current = appliedConcurrency; }, [appliedConcurrency]);
  // Live count of workers actually running right now — surfaced in the UI so the
  // chosen concurrency is observably in effect (e.g. distinguishes "17 workers,
  // all throttled by 429" from "actually running fewer than requested").
  const [liveWorkers, setLiveWorkers] = useState(0);
  // TRUE server-side concurrency, polled from /api/enrich-stats during a run.
  // This is the reliable measure: a client-side counter can't distinguish a fetch
  // on the wire from one queued in the browser's per-host connection pool (the ~6
  // HTTP/1.1 cap), so it over-reports. The server only sees a request once the
  // browser actually sends it, so `serverInFlight`/`serverPeak` reflect the real
  // parallelism reaching the backend.
  const [serverInFlight, setServerInFlight] = useState(0);
  const [serverPeak, setServerPeak] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  // Suggested Azure TPM (tokens/min) to sustain this workload without throttling —
  // derived from actual token usage × achievable request rate during the run.
  const [tpmEstimate, setTpmEstimate] = useState<number | null>(null);

  const [useWebSearch, setUseWebSearch] = useState(true);
  const [realCostEstimate, setRealCostEstimate] = useState<import("@/lib/types").CostEstimate | null>(null);

  const [advancedMode, setAdvancedMode] = useState(false);
  const [customPrompt, setCustomPrompt] = useState("");

  // Session persistence (survives reload + tab close). API key is never saved.
  const [restoredNotice, setRestoredNotice] = useState<{ at: number; fileName: string; done: number; total: number } | null>(null);
  const [saveWarning, setSaveWarning] = useState(false);
  const hydratedRef = useRef(false);

  // Expanded table view
  const [expandedView, setExpandedView] = useState<"test" | "full" | null>(null);

  // Collapsible sidebar cost cards — when folded, only the total is shown.
  const [estimateCollapsed, setEstimateCollapsed] = useState(false);
  const [liveCostCollapsed, setLiveCostCollapsed] = useState(false);

  /* ---- derived ---- */
  const models = provider === "anthropic" ? Object.entries(ANTHROPIC_MODELS) : provider === "grok" ? Object.entries(GROK_MODELS) : provider === "openai" || provider === "azure" ? Object.entries(OPENAI_MODELS) : Object.entries(GEMINI_MODELS);
  const describeReady = file && enrichmentDescription.trim().length > 0 && inputColumns.length > 0 && outputColumns.length > 0;
  // Editable-prompt seed: a TEMPLATE with {column} placeholders — NOT row 0's baked
  // values. buildPrompt() substitutes each row's real data into these at run time.
  const generatedPrompt = file && inputColumns.length > 0 && outputColumns.length > 0 ? buildPromptTemplate(inputColumns, outputColumns, enrichmentDescription, useWebSearch) : "";
  const configReady = describeReady && (!advancedMode || customPrompt.trim().length > 0);
  const runReady = configReady && keyValid;
  const hasPartialRun = fullResults.length > 0 && !fullRunning && !fullDone;
  const remainingRows = file ? Math.max(0, file.totalRows - fullResults.filter((r) => r.success).length) : 0;
  // Successful rows where the model returned "N/A" for at least one output value —
  // the row ran fine but some value couldn't be determined. Not a failure.
  const fullPartial = fullResults.filter((r) => r.success && Object.values(r.data).some((v) => v === "N/A")).length;

  const costRange = useMemo(() => {
    if (!file || inputColumns.length === 0 || outputColumns.length === 0) return null;
    const sample = buildPrompt(inputColumns, file.rows[0], outputColumns, enrichmentDescription, advancedMode ? customPrompt : undefined, useWebSearch);
    const inp = estimateInputTokensPerRow(sample, provider, modelId);
    const out = estimateOutputTokensPerRow(outputColumns);
    return calculateCostRange(file.totalRows, inp, out, provider, modelId, useWebSearch);
  }, [file, inputColumns, outputColumns, enrichmentDescription, customPrompt, advancedMode, provider, modelId, useWebSearch]);

  // Live cost actually INCURRED so far — from the real token counts of every
  // successful call that ran (test rows + full run; both cost real money). Unlike
  // the projected estimate, this only ever counts work that has already happened.
  const liveCost = useMemo(() => {
    if (!file) return null;
    const runRows = [...testResults, ...fullResults].filter(
      (r) => r.success && r.inputTokens != null && r.outputTokens != null
    );
    if (runRows.length === 0) return null;
    const sumIn = runRows.reduce((s, r) => s + (r.inputTokens || 0), 0);
    const sumOut = runRows.reduce((s, r) => s + (r.outputTokens || 0), 0);
    const count = runRows.length;
    // Feed AVERAGE tokens × the real row count back into the estimator so the
    // per-token costs equal the true sums and search fees are charged per row run.
    const est = calculateCostFromActualTokens(
      count,
      Math.round(sumIn / count),
      Math.round(sumOut / count),
      provider,
      modelId,
      useWebSearch
    );
    return { ...est, rowsRun: count, totalInputTokens: sumIn, totalOutputTokens: sumOut };
  }, [file, testResults, fullResults, provider, modelId, useWebSearch]);

  const smartColumns = useMemo(() => {
    if (!file) return { recommended: [] as string[], other: [] as string[] };
    const rec = detectSmartColumns(file.columns);
    const other = file.columns.filter((c) => !rec.includes(c));
    return { recommended: rec, other };
  }, [file]);

  useEffect(() => {
    if (file && !columnsAutoSelected) {
      const smart = detectSmartColumns(file.columns);
      if (smart.length > 0) {
        setInputColumns(smart);
        setColumnsAutoSelected(true);
      }
    }
  }, [file, columnsAutoSelected]);

  /* ---- session restore (on mount) ---- */
  useEffect(() => {
    const s: SavedSession | null = loadSession();
    if (s && hasMeaningfulWork(s)) {
      setFile(s.file);
      setEnrichmentDescription(s.enrichmentDescription);
      setInputColumns(s.inputColumns);
      setOutputColumns(s.outputColumns);
      setColumnsAutoSelected(true); // don't re-run auto-select over restored choices
      setProvider(s.provider);
      setModelId(s.modelId);
      setAzureEndpoint(s.azureEndpoint || "");
      setAzureDeployment(s.azureDeployment || "");
      setUseWebSearch(s.useWebSearch);
      setAdvancedMode(s.advancedMode);
      setCustomPrompt(s.customPrompt);
      setTestResults(s.testResults);
      setTestDone(s.testDone);
      setFullResults(s.fullResults);
      setFullCompleted(s.fullCompleted);
      setFullFailed(s.fullFailed);
      setFullDone(s.fullDone);
      setRestoredNotice({
        at: s.savedAt,
        fileName: s.file?.fileName || "your data",
        done: s.fullResults.length,
        total: s.file?.totalRows || 0,
      });
    }
    hydratedRef.current = true;
  }, []);

  /* ---- session auto-save (debounced) ---- */
  useEffect(() => {
    if (!hydratedRef.current) return; // don't overwrite before the restore runs
    const handle = setTimeout(() => {
      const res = saveSession({
        file,
        enrichmentDescription,
        inputColumns,
        outputColumns,
        provider,
        modelId,
        azureEndpoint,
        azureDeployment,
        useWebSearch,
        advancedMode,
        customPrompt,
        testResults,
        testDone,
        fullResults,
        fullCompleted,
        fullFailed,
        fullDone,
      });
      setSaveWarning(res.quotaExceeded === true);
    }, 600);
    return () => clearTimeout(handle);
  }, [file, enrichmentDescription, inputColumns, outputColumns, provider, modelId, azureEndpoint, azureDeployment, useWebSearch, advancedMode, customPrompt, testResults, testDone, fullResults, fullCompleted, fullFailed, fullDone]);

  /* ---- rate-limit countdown ticker ---- */
  useEffect(() => {
    if (rateLimitResetAt == null) { setRlCountdown(0); return; }
    const tick = () => {
      const secs = Math.max(0, Math.ceil((rateLimitResetAt - Date.now()) / 1000));
      setRlCountdown(secs);
      if (secs === 0) setRateLimitResetAt(null); // window elapsed — clear the banner
    };
    tick();
    const id = setInterval(tick, 250);
    return () => clearInterval(id);
  }, [rateLimitResetAt]);

  /* ---- ETA ticker ----
     Recompute the estimate reactively while a run is active: on an interval AND
     immediately whenever the applied concurrency or progress changes. Computing
     it only on row completion made it look frozen after a concurrency change (the
     divisor updated, but nothing re-ran the math until the next row finished). */
  useEffect(() => {
    if (!fullRunning || !file) return;
    const recompute = () => {
      const durations = reqDurationsRef.current;
      const avgMs = durations.length > 0 ? durations.reduce((s, d) => s + d, 0) / durations.length : 0;
      if (avgMs <= 0) return;
      const remaining = file.rows.length - (fullCompleted + fullFailed);
      // ETA reflects the ACTUAL running rate (the applied worker count).
      const etaWorkers = Math.max(1, appliedConcurrency);
      setEtaSeconds(remaining <= 0 ? 0 : Math.ceil((remaining / etaWorkers) * avgMs / 1000));
      // Suggested Azure TPM previews the DRAFT slider value (`concurrency`), so the
      // number moves as the user drags the slider — before they press Apply — to
      // help them size their deployment's limit for the count they're considering.
      // TPM = avg tokens/request × requests/min sustainable = workers × (60s / avgMs).
      const tpmWorkers = Math.max(1, concurrency);
      const enriched = fullResults.filter((x) => x?.success && x.inputTokens && x.outputTokens);
      if (enriched.length >= 3) {
        const avgIn = enriched.reduce((s, x) => s + (x.inputTokens || 0), 0) / enriched.length;
        const avgOut = enriched.reduce((s, x) => s + (x.outputTokens || 0), 0) / enriched.length;
        const reqPerMin = (tpmWorkers * 60_000) / avgMs;
        setTpmEstimate(Math.ceil(((avgIn + avgOut) * reqPerMin) / 1000) * 1000); // round to 1k
      }
    };
    recompute();
    const id = setInterval(recompute, 1000);
    return () => clearInterval(id);
  }, [fullRunning, file, appliedConcurrency, concurrency, fullCompleted, fullFailed, fullResults]);

  /* ---- server-side concurrency poller ----
     Poll /api/enrich-stats while a run is active to show the TRUE number of
     requests the backend is handling in parallel (and the peak). This is the
     honest measure of concurrency — the client can't see it because browser-
     queued fetches look "in flight" to JS even when only ~6 are on the wire. */
  useEffect(() => {
    if (!fullRunning) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch("/api/enrich-stats");
        if (!res.ok) return;
        const s = await res.json();
        if (cancelled) return;
        setServerInFlight(s.inFlight ?? 0);
        setServerPeak(s.peak ?? 0);
      } catch { /* ignore transient errors */ }
    };
    poll();
    const id = setInterval(poll, 500);
    return () => { cancelled = true; clearInterval(id); };
  }, [fullRunning]);

  /* ---- close expanded view on Escape ---- */
  useEffect(() => {
    if (!expandedView) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setExpandedView(null); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [expandedView]);

  /* ---- warn before leaving while a run is active ---- */
  useEffect(() => {
    const active = testRunning || fullRunning;
    if (!active) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [testRunning, fullRunning]);

  const startFresh = () => {
    clearSession();
    setRestoredNotice(null);
    setFile(null); setFileError(""); setEnrichmentDescription("");
    setInputColumns([]); setOutputColumns([]); setAutoDetected(false);
    setColumnsAutoSelected(false); setShowAllColumns(false);
    setTestResults([]); setTestDone(false);
    setFullResults([]); setFullCompleted(0); setFullFailed(0); setFullDone(false);
    setRealCostEstimate(null); setAdvancedMode(false); setCustomPrompt("");
  };

  // Reset just the run: clear preview + full results but keep the file, columns,
  // description, provider, key, and all settings — back to the pre-preview state.
  const resetRun = () => {
    setTestResults([]); setTestDone(false);
    setFullResults([]); setFullCompleted(0); setFullFailed(0); setFullDone(false);
    setFullPaused(false); setRealCostEstimate(null); setRateLimitResetAt(null);
    pauseRef.current = false; stopRef.current = false;
  };

  /* ---- handlers ---- */
  // Azure carries its endpoint + deployment + key as a JSON blob in the apiKey field.
  const azureConfigJSON = () =>
    JSON.stringify({ endpoint: azureEndpoint.trim(), deployment: azureDeployment.trim(), key: apiKey.trim() });

  const validateKey = async () => {
    if (!apiKey.trim()) { setKeyError("Please enter an API key"); return; }
    if (provider === "azure" && (!azureEndpoint.trim() || !azureDeployment.trim())) {
      setKeyError("Please enter your Azure endpoint and deployment name"); return;
    }
    setValidating(true); setKeyError(""); setKeyWarning("");
    try {
      const keyPayload = provider === "azure" ? azureConfigJSON() : apiKey.trim();
      const res = await fetch("/api/validate-key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider, apiKey: keyPayload }) });
      const data = await res.json();
      if (data.valid) { setKeyValid(true); if (data.warning) setKeyWarning(data.warning); } else { setKeyError(data.error || "Invalid API key"); }
    } catch { setKeyError("Validation failed. Try again."); }
    finally { setValidating(false); }
  };

  const handleFile = async (f: File) => {
    setFileError("");
    if (f.size > 10 * 1024 * 1024) { setFileError("File exceeds 10MB limit."); return; }
    try {
      const parsed = await parseFile(f);
      if (parsed.totalRows === 0) { setFileError("File is empty."); return; }
      setFile(parsed);
      setInputColumns([]); setOutputColumns([]); setAutoDetected(false); setColumnsAutoSelected(false);
      setShowAllColumns(false);
      setTestDone(false); setTestResults([]); setFullDone(false); setFullResults([]);
      setFullCompleted(0); setFullFailed(0); setRestoredNotice(null);
    } catch (err) { setFileError((err as Error).message); }
  };

  const toggleColumn = (col: string) => setInputColumns((p) => p.includes(col) ? p.filter((c) => c !== col) : [...p, col]);

  const addOutputColumn = () => {
    const name = newColumnName.trim();
    if (!name) return;
    const key = name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    if (outputColumns.some((c) => c.key === key)) return;
    setOutputColumns((p) => [...p, { key, label: name }]);
    setNewColumnName("");
  };

  const removeOutputColumn = (key: string) => setOutputColumns((p) => p.filter((c) => c.key !== key));

  const handleDescriptionBlur = () => {
    if (outputColumns.length === 0 && enrichmentDescription.trim()) {
      const detected = detectOutputColumns(enrichmentDescription);
      if (detected.length > 0) { setOutputColumns(detected); setAutoDetected(true); }
    }
  };

  const applyTemplate = (template: typeof PROMPT_TEMPLATES[0]) => {
    setEnrichmentDescription(template.description);
    const detected = detectOutputColumns(template.description);
    if (detected.length > 0) { setOutputColumns(detected); setAutoDetected(true); }
  };

  const enrichSingleRow = useCallback(async (row: Record<string, string>, index: number): Promise<EnrichmentResult> => {
    const prompt = buildPrompt(inputColumns, row, outputColumns, enrichmentDescription, advancedMode ? customPrompt : undefined, useWebSearch);
    let attempt = 0;        // generic (non-rate-limit) failures — these can still fail
    const MAX_ATTEMPTS = 3;
    while (true) {
      if (stopRef.current) return { rowIndex: index, success: false, data: {}, error: "Stopped" };
      // Respect the shared rate-limit gate: if another worker hit a 429, hold here
      // until its cooldown passes so all workers resume together (no lockstep re-trip).
      while (rateLimitGateRef.current > Date.now()) {
        if (stopRef.current) return { rowIndex: index, success: false, data: {}, error: "Stopped" };
        await new Promise((r) => setTimeout(r, Math.min(250, rateLimitGateRef.current - Date.now())));
      }
      try {
        const startedAt = Date.now();
        const result = provider === "anthropic"
          ? await enrichRowAnthropic(apiKey, modelId as AnthropicModelId, prompt, useWebSearch)
          : provider === "grok"
          ? await enrichRowGrok(apiKey, modelId as GrokModelId, prompt, useWebSearch)
          : provider === "openai"
          ? await enrichRowOpenAI(apiKey, modelId as OpenAIModelId, prompt, useWebSearch)
          : provider === "vertex"
          ? await enrichRowVertex(apiKey, modelId as GeminiModelId, prompt, useWebSearch)
          : provider === "azure"
          ? await enrichRowAzure({ endpoint: azureEndpoint.trim(), deployment: azureDeployment.trim(), key: apiKey.trim() }, modelId as OpenAIModelId, prompt, useWebSearch)
          : await enrichRowGemini(apiKey, modelId as GeminiModelId, prompt, useWebSearch);
        // Record latency in the rolling window (drop the oldest) for the ETA estimate.
        const durations = reqDurationsRef.current;
        durations.push(Date.now() - startedAt);
        if (durations.length > ETA_WINDOW) durations.shift();
        // Success — gently decay the shared backoff so the fleet can speed back up
        // toward the deployment's real throughput after a burst of 429s.
        rlBackoffRef.current = Math.floor(rlBackoffRef.current / 2);
        if (rlBackoffRef.current < RL_BACKOFF_FLOOR) rlBackoffRef.current = 0;
        return { rowIndex: index, success: true, data: result.data, inputTokens: result.inputTokens, outputTokens: result.outputTokens };
      } catch (err) {
        if (err instanceof RateLimitError) {
          // Rate limits are NEVER terminal: a throttled row queues and retries
          // indefinitely until it succeeds (or the user Stops). Nothing fails on 429.
          // Grow the SHARED, persistent backoff (doubling, floor 0.5s, cap 60s) so the
          // whole fleet converges to the real throughput instead of resetting per-row.
          const grown = rlBackoffRef.current > 0
            ? Math.min(RL_BACKOFF_CAP, rlBackoffRef.current * 2)
            : RL_BACKOFF_FLOOR;
          rlBackoffRef.current = grown;
          // Honor Azure's Retry-After header when it's longer than our backoff.
          const waitMs = Math.max(
            err.retryAfter != null ? err.retryAfter * 1000 : 0,
            grown
          ) + Math.floor(Math.random() * 500); // jitter so workers don't re-trip in lockstep
          // Extend the SHARED gate so every worker pauses together, and surface the
          // resume time to the UI countdown. Never pull the gate earlier than it is.
          const target = Date.now() + waitMs;
          rateLimitGateRef.current = Math.max(rateLimitGateRef.current, target);
          setRateLimitResetAt((prev) => (prev && prev > target ? prev : target));
          // The gate loop at the top of the while() performs the actual wait.
          continue; // does NOT consume a generic attempt, and has no retry cap
        }
        if (attempt >= MAX_ATTEMPTS - 1) return { rowIndex: index, success: false, data: {}, error: (err as Error).message };
        await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)));
        attempt++;
      }
    }
  }, [inputColumns, outputColumns, enrichmentDescription, advancedMode, customPrompt, provider, apiKey, azureEndpoint, azureDeployment, modelId, useWebSearch]);

  const runTest = async () => {
    if (!file) return;
    setTestRunning(true); setTestResults([]); setTestDone(false); setRealCostEstimate(null);
    setRateLimitResetAt(null); rateLimitGateRef.current = 0; rlBackoffRef.current = 0;
    const rows = file.rows.slice(0, 3);
    const results: EnrichmentResult[] = [];
    for (let i = 0; i < rows.length; i++) {
      const r = await enrichSingleRow(rows[i], i);
      results.push(r);
      setTestResults([...results]);
    }
    const successResults = results.filter((r) => r.success && r.inputTokens && r.outputTokens);
    if (successResults.length > 0) {
      const avgInput = Math.round(successResults.reduce((s, r) => s + (r.inputTokens || 0), 0) / successResults.length);
      const avgOutput = Math.round(successResults.reduce((s, r) => s + (r.outputTokens || 0), 0) / successResults.length);
      const precise = calculateCostFromActualTokens(file.totalRows, avgInput, avgOutput, provider, modelId, useWebSearch);
      setRealCostEstimate(precise);
    }
    setTestDone(true); setTestRunning(false);
  };

  const runFull = async (resume = false) => {
    if (!file) return;
    setRestoredNotice(null);
    setFullRunning(true); setFullDone(false);
    setServerPeak(0); setServerInFlight(0);
    // Reset the server-side peak so this run's parallelism is measured fresh.
    fetch("/api/enrich-stats", { method: "POST" }).catch(() => {});
    stopRef.current = false; pauseRef.current = false; setFullPaused(false);
    setRateLimitResetAt(null); rateLimitGateRef.current = 0; rlBackoffRef.current = 0;
    reqDurationsRef.current = []; setEtaSeconds(null); setTpmEstimate(null);

    // Pre-populate from existing results when resuming, so already-enriched rows are skipped.
    const all: EnrichmentResult[] = new Array(file.rows.length);
    let done = 0, fail = 0;
    if (resume) {
      for (const r of fullResults) {
        if (r.rowIndex >= 0 && r.rowIndex < all.length) {
          all[r.rowIndex] = r;
          r.success ? done++ : fail++;
        }
      }
    } else {
      setFullResults([]);
    }
    setFullCompleted(done); setFullFailed(fail);

    let nextIdx = 0;
    let activeWorkers = 0; // live count of running workers, for dynamic pool sizing
    const worker = async () => {
      activeWorkers++;
      setLiveWorkers(activeWorkers);
      try {
      while (nextIdx < file.rows.length) {
        if (stopRef.current) return;
        // Live concurrency: if the user dialed the worker count DOWN, surplus
        // workers retire here (after finishing their current row) until the pool
        // matches the target. The supervisor below tops it back up when raised.
        if (activeWorkers > Math.max(1, concurrencyRef.current)) return;
        while (pauseRef.current) { await new Promise((r) => setTimeout(r, 200)); if (stopRef.current) return; }
        const idx = nextIdx++;
        if (idx >= file.rows.length) return;
        if (all[idx]?.success) continue; // already enriched — skip on resume
        const r = await enrichSingleRow(file.rows[idx], idx);
        all[idx] = r;
        r.success ? done++ : fail++;
        setFullCompleted(done); setFullFailed(fail); setFullResults([...all.filter(Boolean)]);
        // ETA and the suggested Azure TPM are computed reactively by the ETA ticker
        // effect (recomputes on an interval + on concurrency/progress change), so
        // they aren't derived here.
        // Refine the cost estimate live from REAL token counts across every enriched
        // row so far (far more accurate than the 3-row test). The Azure TPM figure
        // is computed in the ETA ticker effect instead, so it tracks the applied
        // concurrency live rather than only updating when a row finishes here.
        const enriched = all.filter((x) => x?.success && x.inputTokens && x.outputTokens);
        if (enriched.length >= 3) {
          const avgIn = Math.round(enriched.reduce((s, x) => s + (x.inputTokens || 0), 0) / enriched.length);
          const avgOut = Math.round(enriched.reduce((s, x) => s + (x.outputTokens || 0), 0) / enriched.length);
          setRealCostEstimate(calculateCostFromActualTokens(file.totalRows, avgIn, avgOut, provider, modelId, useWebSearch));
        }
      }
      } finally {
        activeWorkers--;
        setLiveWorkers(activeWorkers);
      }
    };
    // Dynamic worker pool. Workers coordinate through the shared rate-limit gate
    // (see enrichSingleRow): a 429 from any worker pauses the whole fleet, so
    // concurrency gives throughput without lockstep re-tripping the limit.
    // A supervisor keeps the pool sized to `concurrencyRef` LIVE: it spawns new
    // workers when the user dials the count up; workers self-retire when dialed
    // down (see the guard at the top of worker()). This lets the user tune
    // concurrency mid-run without restarting the batch.
    const pool: Promise<void>[] = [];
    while (nextIdx < file.rows.length && !stopRef.current) {
      const target = Math.min(MAX_CONCURRENCY, Math.max(1, concurrencyRef.current));
      while (activeWorkers < target && nextIdx < file.rows.length) {
        pool.push(worker());
      }
      // Poll periodically to react to concurrency changes and to detect completion.
      await new Promise((r) => setTimeout(r, 200));
    }
    await Promise.all(pool);
    setLiveWorkers(0);
    setFullRunning(false); setFullDone(true); setRateLimitResetAt(null); setEtaSeconds(null);
  };

  const handleDownload = () => {
    if (!file) return;
    const enriched = file.rows.map((_, i) => {
      const r = fullResults.find((x) => x.rowIndex === i);
      if (r?.success) return r.data;
      const empty: Record<string, string> = {};
      for (const c of outputColumns) empty[c.key] = r ? `Error: ${r.error || "Failed"}` : "";
      return empty;
    });
    const blob = exportToFile(file, enriched, outputColumns.map((c) => c.key));
    const ext = file.fileType === "csv" ? "csv" : "xlsx";
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `${file.fileName.replace(/\.[^.]+$/, "")}_enriched.${ext}`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  };

  const providerPricingUrl = provider === "anthropic" ? ANTHROPIC_PRICING_URL : provider === "grok" ? GROK_PRICING_URL : provider === "openai" ? OPENAI_PRICING_URL : provider === "azure" ? AZURE_PRICING_URL : provider === "vertex" ? VERTEX_PRICING_URL : GEMINI_PRICING_URL;

  /* ---- render ---- */
  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      {/* Top bar */}
      <header className="sticky top-0 z-40 border-b border-zinc-200 bg-white/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link href="/" className="flex items-center gap-2">
            <img src="/icon.svg" alt="OpenClay" className="h-7 w-7 rounded-lg" />
            <span className="text-base font-bold">OpenClay</span>
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] text-emerald-700">
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" /></svg>
              Your data stays in your browser
            </div>
            <span className="hidden rounded-full bg-zinc-100 px-3 py-1 text-[11px] font-medium text-zinc-600 ring-1 ring-zinc-200 sm:inline-flex">100% free</span>
            <a href="https://www.linkedin.com/in/-raghav/" target="_blank" rel="noopener noreferrer" className="hidden items-center gap-1.5 rounded-full border border-zinc-200 px-3 py-1 text-[11px] text-zinc-500 transition hover:border-zinc-300 hover:text-zinc-900 sm:inline-flex" title="Feedback? Connect with the creator">
              <svg className="h-3 w-3" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
              Feedback
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {/* Restored-session banner */}
        {restoredNotice && (
          <div className="mb-5 flex flex-col gap-2 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-2.5">
              <svg className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              <div className="text-xs text-blue-800">
                <span className="font-semibold">Session restored</span> from {timeAgo(restoredNotice.at)} — {restoredNotice.fileName}
                {restoredNotice.done > 0 && <> ({restoredNotice.done}/{restoredNotice.total} rows enriched)</>}.
                {restoredNotice.done > 0 && !keyValid && <span className="text-blue-600"> Re-enter your API key to resume.</span>}
              </div>
            </div>
            <div className="flex shrink-0 gap-2">
              <button onClick={() => setRestoredNotice(null)} className="rounded-lg border border-blue-200 bg-white px-3 py-1.5 text-[11px] font-medium text-blue-700 hover:bg-blue-100">Dismiss</button>
              <button onClick={startFresh} className="rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-[11px] font-medium text-zinc-600 hover:bg-zinc-50">Start fresh</button>
            </div>
          </div>
        )}

        {/* Auto-save quota warning */}
        {saveWarning && (
          <div className="mb-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
            <span className="font-semibold">Heads up:</span> this dataset is too large to auto-save in your browser. Use <span className="font-medium">&ldquo;Download results so far&rdquo;</span> periodically so you don&apos;t lose progress.
          </div>
        )}

        <div className="grid gap-5 lg:grid-cols-[1fr_340px]">

          {/* ============ LEFT COLUMN ============ */}
          <div className="min-w-0 space-y-5">

            {/* --- 1. Upload File --- */}
            <Card glow={!file}>
              <StepHeader num={1} title="Upload your spreadsheet" subtitle="CSV, XLS, or XLSX — max 10MB" done={!!file} active={!file} />
              {!file ? (
                <div>
                  <div
                    onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
                    onClick={() => fileInputRef.current?.click()}
                    className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-12 transition ${dragging ? "border-zinc-400 bg-zinc-100" : "border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50"}`}>
                    <svg className="mb-3 h-10 w-10 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>
                    <p className="text-sm font-medium text-zinc-700">Drop your file here or click to browse</p>
                    <p className="mt-1 text-xs text-zinc-400">.xlsx, .xls, or .csv</p>
                    <input ref={fileInputRef} type="file" accept=".xlsx,.xls,.csv" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} className="hidden" />
                  </div>
                  {fileError && <p className="mt-2 text-xs text-red-600">{fileError}</p>}
                  <TrustBadge text="Files are parsed in your browser. Nothing is uploaded to any server." />
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between rounded-lg bg-zinc-50 px-3 py-2 ring-1 ring-zinc-200">
                    <div>
                      <p className="text-xs font-medium text-zinc-900">{file.fileName}</p>
                      <p className="text-[11px] text-zinc-500">{file.totalRows} rows &middot; {file.columns.length} columns</p>
                    </div>
                    <button onClick={() => { setFile(null); setInputColumns([]); setOutputColumns([]); setTestDone(false); setFullDone(false); setColumnsAutoSelected(false); }} className="text-xs text-red-500 hover:text-red-700">Remove</button>
                  </div>
                  <div className="overflow-hidden rounded-lg border border-zinc-200">
                    <div className="max-h-48 overflow-auto">
                      <table className="min-w-full text-[11px]">
                        <thead className="sticky top-0 bg-zinc-50"><tr>{file.columns.map((c) => <th key={c} className="whitespace-nowrap px-2.5 py-1.5 text-left font-medium text-zinc-600">{c}</th>)}</tr></thead>
                        <tbody>{file.rows.slice(0, 6).map((r, i) => <tr key={i} className="border-t border-zinc-100">{file.columns.map((c) => <td key={c} className="max-w-[160px] truncate whitespace-nowrap px-2.5 py-1.5 text-zinc-500">{r[c]}</td>)}</tr>)}</tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </Card>

            {/* --- 2. Tell us what to enrich --- */}
            <Card className={!file ? "opacity-30 pointer-events-none" : ""} glow={!!file && !describeReady}>
              <StepHeader num={2} title="Tell us what to enrich" subtitle="Describe what you need, and we'll handle the rest" done={!!describeReady} active={!!file && !describeReady} />

              {/* Visual flow diagram */}
              {file && (
                <div className="mb-5 flex items-center justify-center gap-3 rounded-xl bg-zinc-50 px-4 py-3 text-[11px] ring-1 ring-zinc-100">
                  <div className="flex items-center gap-1.5 rounded-lg bg-white px-3 py-1.5 ring-1 ring-zinc-200">
                    <svg className="h-3 w-3 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0112 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M3.375 8.25c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m17.25-3.75h-7.5c-.621 0-1.125.504-1.125 1.125m8.625-1.125c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M12 10.875v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125M13.125 12h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125M20.625 12c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5M12 14.625v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 14.625c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m0 0v.375" /></svg>
                    <span className="text-zinc-500">Your columns</span>
                  </div>
                  <svg className="h-3.5 w-3.5 shrink-0 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" /></svg>
                  <div className="flex items-center gap-1.5 rounded-lg bg-blue-50 px-3 py-1.5 ring-1 ring-blue-200">
                    <svg className="h-3 w-3 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" /></svg>
                    <span className="text-blue-700">AI + Web Search</span>
                  </div>
                  <svg className="h-3.5 w-3.5 shrink-0 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" /></svg>
                  <div className="flex items-center gap-1.5 rounded-lg bg-emerald-50 px-3 py-1.5 ring-1 ring-emerald-200">
                    <svg className="h-3 w-3 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                    <span className="text-emerald-700">New columns added</span>
                  </div>
                </div>
              )}

              {/* A: Select input columns */}
              {file && (
                <div className="mb-5">
                  <label className="mb-2 flex items-center text-xs font-medium text-zinc-700">
                    Columns to look up
                    <InfoTip text="Select columns that contain the data AI should search for." />
                  </label>

                  <div className="flex flex-wrap gap-1.5">
                    {smartColumns.recommended.map((col) => (
                      <button key={col} onClick={() => toggleColumn(col)}
                        className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${inputColumns.includes(col) ? "border-zinc-400 bg-zinc-900 text-white" : "border-zinc-200 text-zinc-500 hover:border-zinc-300 hover:bg-zinc-50"}`}>
                        {col}
                      </button>
                    ))}
                  </div>

                  {smartColumns.other.length > 0 && (
                    <div className="mt-2">
                      {showAllColumns ? (
                        <>
                          <div className="flex flex-wrap gap-1.5">
                            {smartColumns.other.map((col) => (
                              <button key={col} onClick={() => toggleColumn(col)}
                                className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${inputColumns.includes(col) ? "border-zinc-400 bg-zinc-900 text-white" : "border-zinc-200 text-zinc-500 hover:border-zinc-300 hover:bg-zinc-50"}`}>
                                {col}
                              </button>
                            ))}
                          </div>
                          <button onClick={() => setShowAllColumns(false)} className="mt-2 text-[11px] text-zinc-400 hover:text-zinc-700">Show less</button>
                        </>
                      ) : (
                        <button onClick={() => setShowAllColumns(true)} className="mt-1 text-[11px] text-zinc-400 hover:text-zinc-700">
                          + {smartColumns.other.length} more columns
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* B: Describe what AI should find */}
              <div className="mb-5">
                <label className="mb-2 flex items-center text-xs font-medium text-zinc-700">
                  What should AI find for each row?
                  <InfoTip text="Describe in plain English. We'll auto-suggest output columns." />
                </label>
                <textarea
                  value={enrichmentDescription}
                  onChange={(e) => { setEnrichmentDescription(e.target.value); if (autoDetected) { setAutoDetected(false); } }}
                  onBlur={handleDescriptionBlur}
                  rows={3}
                  placeholder="e.g. Find the CEO name, total funding raised, employee count, and a brief company description"
                  className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                />

                {/* Prompt templates */}
                <div className="mt-2.5">
                  {!enrichmentDescription && <p className="mb-1.5 text-[11px] text-zinc-400">Or start from a template:</p>}
                  <div className="flex flex-wrap gap-2">
                    {PROMPT_TEMPLATES.map((t) => (
                      <button key={t.label} onClick={() => applyTemplate(t)}
                        className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-1.5 text-[11px] font-medium text-zinc-500 transition hover:border-zinc-300 hover:bg-white hover:text-zinc-900">
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* C: New columns to add */}
              <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4">
                <label className="mb-3 flex items-center text-xs font-medium text-zinc-700">
                  <svg className="mr-1.5 h-3.5 w-3.5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                  New columns to add
                  <InfoTip text="These columns will be filled with AI-generated data for each row." />
                </label>

                {outputColumns.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-2">
                    {outputColumns.map((col) => (
                      <span key={col.key} className="flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
                        {col.label}
                        <button onClick={() => removeOutputColumn(col.key)} className="text-emerald-400 hover:text-red-500 transition">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex gap-2">
                  <input
                    value={newColumnName}
                    onChange={(e) => setNewColumnName(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addOutputColumn(); } }}
                    placeholder="Type a column name, e.g. CEO Name"
                    className="flex-1 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                  />
                  <button onClick={addOutputColumn} disabled={!newColumnName.trim()}
                    className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-medium text-white transition hover:bg-emerald-500 disabled:opacity-30">
                    Add
                  </button>
                </div>
              </div>

              {/* Advanced */}
              <div className="mt-4 border-t border-zinc-200 pt-3">
                <label className="flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={advancedMode}
                    onChange={(e) => { setAdvancedMode(e.target.checked); if (e.target.checked && generatedPrompt) setCustomPrompt(generatedPrompt); }}
                    className="rounded border-zinc-300 text-zinc-900 focus:ring-zinc-300" />
                  <span className="text-zinc-500">Advanced: Edit the prompt template</span>
                </label>
                {advancedMode && (
                  <div className="mt-2">
                    <textarea value={customPrompt} onChange={(e) => setCustomPrompt(e.target.value)} rows={8}
                      className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 font-mono text-[11px] text-zinc-700 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none" />
                    <div className="mt-1 flex justify-between text-[11px] text-zinc-400">
                      <span>Use {"{column_name}"} to reference columns</span>
                      <button onClick={() => setCustomPrompt(generatedPrompt)} className="text-zinc-500 hover:text-zinc-900">Reset to generated</button>
                    </div>
                  </div>
                )}
              </div>
            </Card>

            {/* --- 3. Provider + Model --- */}
            <Card className={!file ? "opacity-30 pointer-events-none" : ""}>
              <StepHeader num={3} title="Choose provider & model" subtitle="All models include live web search" done={!!modelId} active={!!file} />

              <div className="mb-4 flex flex-wrap gap-2">
                {([
                  { p: "gemini" as Provider, label: "Gemini", model: "gemini-3.1-pro-preview" as ModelId },
                  { p: "vertex" as Provider, label: "Vertex AI", model: "gemini-3.1-pro-preview" as ModelId },
                  { p: "openai" as Provider, label: "OpenAI", model: "gpt-5.4-mini" as ModelId },
                  { p: "azure" as Provider, label: "Azure OpenAI", model: "gpt-5.4-mini" as ModelId },
                  { p: "anthropic" as Provider, label: "Claude", model: "claude-sonnet-4-5-20250929" as ModelId },
                  { p: "grok" as Provider, label: "Grok", model: "grok-4-0320" as ModelId },
                ] as const).map(({ p, label, model }) => (
                  <button key={p} onClick={() => { setProvider(p); setModelId(model); setKeyValid(false); setApiKey(""); setKeyError(""); setKeyWarning(""); setRealCostEstimate(null); }}
                    className={`rounded-lg border-2 px-3 py-2.5 text-xs font-medium transition ${provider === p ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-200 text-zinc-500 hover:border-zinc-300"}`}>
                    {label}
                  </button>
                ))}
              </div>

              <div className="space-y-2">
                {models.map(([id, model]) => {
                  const guidance = MODEL_GUIDANCE[id as ModelId];
                  return (
                    <button key={id} onClick={() => setModelId(id as ModelId)}
                      className={`w-full rounded-lg border-2 px-3 py-3 text-left transition ${modelId === id ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 hover:border-zinc-300"}`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-zinc-900">{model.name}</span>
                          {model.recommended && <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 ring-1 ring-emerald-200">Recommended</span>}
                        </div>
                        <span className="text-[11px] text-zinc-400">${model.inputPer1M} / ${model.outputPer1M} per 1M tokens</span>
                      </div>
                      {guidance && (
                        <div className="mt-1.5 flex items-center gap-2">
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${SPEED_COLORS[guidance.speed]}`}>
                            {guidance.speed === "fast" ? "Fast" : guidance.speed === "medium" ? "Medium" : "Slow"}
                          </span>
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${QUALITY_COLORS[guidance.quality]}`}>
                            {guidance.quality === "good" ? "Good" : guidance.quality === "great" ? "Great" : "Best"} quality
                          </span>
                          <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 ring-1 ring-blue-200">Web search</span>
                          <InfoTip text={guidance.bestFor} />
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
              <p className="mt-3 text-[11px] text-zinc-400">Not sure? The recommended model is a great default for most tasks.</p>
              {provider === "azure" && (
                <p className="mt-1.5 text-[11px] text-blue-600">Azure routes by your deployment name, not the model. Pick the model that matches your deployment — this drives cost estimates only.</p>
              )}

              {/* Web search toggle */}
              <div className="mt-4 border-t border-zinc-200 pt-3">
                <label className="flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={useWebSearch}
                    onChange={(e) => { setUseWebSearch(e.target.checked); setRealCostEstimate(null); }}
                    className="rounded border-zinc-300 text-zinc-900 focus:ring-zinc-300" />
                  <span className="text-zinc-700">Enable web search</span>
                  <InfoTip text="Web search lets AI look up live data from the internet. Disable to use AI knowledge only (cheaper but may be outdated)." />
                </label>
                {!useWebSearch && (
                  <p className="mt-1.5 ml-6 text-[11px] text-amber-600">AI will use its training data only — results may not reflect the latest information.</p>
                )}
              </div>
            </Card>

            {/* --- 4. API Key --- */}
            <Card className={!configReady ? "opacity-30 pointer-events-none" : ""} glow={!!configReady && !keyValid}>
              <StepHeader
                num={4}
                title={`Connect your ${provider === "anthropic" ? "Anthropic" : provider === "grok" ? "xAI" : provider === "openai" ? "OpenAI" : provider === "azure" ? "Azure OpenAI" : provider === "vertex" ? "Vertex AI" : "Google"} API key`}
                subtitle="Your key is never stored — it stays in browser memory only"
                done={keyValid}
                active={!!configReady && !keyValid}
              />

              {!keyValid ? (
                <div className="space-y-3">
                  {provider === "azure" && (
                    <>
                      <input type="text" value={azureEndpoint} onChange={(e) => { setAzureEndpoint(e.target.value); setKeyError(""); }}
                        placeholder="https://your-resource.openai.azure.com (or paste the full Responses URL)"
                        className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                      />
                      <p className="text-[11px] text-zinc-400">Paste your base resource URL, or the full <code className="rounded bg-zinc-100 px-1">.../responses?api-version=...</code> URL from the Azure portal — either works.</p>
                      <input type="text" value={azureDeployment} onChange={(e) => { setAzureDeployment(e.target.value); setKeyError(""); }}
                        placeholder="Deployment name (e.g. my-gpt-4o)"
                        className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                      />
                    </>
                  )}
                  {provider === "vertex" ? (
                    <textarea value={apiKey} onChange={(e) => { setApiKey(e.target.value); setKeyError(""); }}
                      placeholder={'Paste your service account JSON here:\n{"type": "service_account", "project_id": "...", ...}'}
                      rows={4}
                      className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-xs font-mono text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                    />
                  ) : (
                    <input type="password" value={apiKey} onChange={(e) => { setApiKey(e.target.value); setKeyError(""); }}
                      placeholder={provider === "anthropic" ? "sk-ant-..." : provider === "grok" ? "xai-..." : provider === "openai" ? "sk-..." : provider === "azure" ? "Azure API key" : "AIza..."}
                      className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-300 focus:outline-none"
                    />
                  )}
                  {keyError && <p className="text-xs text-red-600">{keyError}</p>}
                  <button onClick={validateKey} disabled={validating || !apiKey.trim()}
                    className="w-full rounded-lg bg-zinc-900 py-2.5 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:opacity-50">
                    {validating ? "Validating..." : "Validate & Connect"}
                  </button>
                  <p className="text-center text-[11px] text-zinc-400">
                    {provider === "anthropic" && <a href="https://console.anthropic.com" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Get a key from Anthropic</a>}
                    {provider === "gemini" && <a href="https://aistudio.google.com" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Get a key from Google AI Studio</a>}
                    {provider === "vertex" && <a href="https://console.cloud.google.com/iam-admin/serviceaccounts" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Create a service account in Google Cloud</a>}
                    {provider === "openai" && <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Get a key from OpenAI</a>}
                    {provider === "grok" && <a href="https://console.x.ai" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Get a key from xAI Console</a>}
                    {provider === "azure" && <a href="https://portal.azure.com" target="_blank" rel="noopener noreferrer" className="text-zinc-600 underline hover:text-zinc-900">Find your endpoint, deployment & key in the Azure portal</a>}
                  </p>
                  <TrustBadge text="Your API key is never stored, logged, or sent to our servers. It goes directly from your browser to the AI provider." />
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between rounded-lg bg-emerald-50 px-3 py-2 ring-1 ring-emerald-200">
                    <span className="text-xs font-medium text-emerald-700">
                      {provider === "anthropic" ? "Anthropic" : provider === "grok" ? "Grok" : provider === "openai" ? "OpenAI" : provider === "azure" ? "Azure OpenAI" : provider === "vertex" ? "Vertex AI" : "Gemini"} connected
                    </span>
                    <button onClick={() => { setKeyValid(false); setApiKey(""); setKeyWarning(""); }} className="text-xs text-red-500 hover:text-red-700">Disconnect</button>
                  </div>
                  {keyWarning && <p className="rounded-lg bg-amber-50 px-3 py-2 text-[11px] text-amber-700 ring-1 ring-amber-200">{keyWarning}</p>}
                </div>
              )}
            </Card>

            {/* --- 5. Test, Run & Download --- */}
            <Card className={!runReady ? "opacity-30 pointer-events-none" : ""} glow={!!runReady && !fullDone}>
              <StepHeader num={5} title="Preview, run & download" subtitle="Test on 3 rows first, then run all" done={fullDone} active={!!runReady && !fullDone} />

              {!testDone && !testRunning && (
                <div>
                  <p className="mb-3 text-xs text-zinc-500">
                    We&apos;ll test with the first 3 rows so you can verify results before running the full batch.
                  </p>
                  <button onClick={runTest} disabled={!runReady}
                    className="w-full rounded-lg bg-zinc-900 py-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:opacity-50">
                    Preview with first {file ? Math.min(3, file.totalRows) : 3} rows
                  </button>
                </div>
              )}

              {(testRunning || testDone) && (
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100">
                      <div className="h-full rounded-full bg-zinc-900 transition-all" style={{ width: `${(testResults.length / Math.min(3, file?.totalRows || 3)) * 100}%` }} />
                    </div>
                    <span className="text-[11px] text-zinc-500">{testResults.length}/{Math.min(3, file?.totalRows || 3)}</span>
                  </div>

                  {testResults.length > 0 && (
                    <div className="overflow-hidden rounded-lg border border-zinc-200">
                      <div className="flex items-center justify-between bg-zinc-50 px-2.5 py-1.5">
                        <span className="text-[11px] font-medium text-zinc-500">Preview results</span>
                        <button onClick={() => setExpandedView("test")} className="flex items-center gap-1 text-[11px] font-medium text-zinc-500 hover:text-zinc-900 transition">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9m11.25-5.25v4.5m0-4.5h-4.5m4.5 0L15 9m-11.25 11.25v-4.5m0 4.5h4.5m-4.5 0L9 15m11.25 5.25v-4.5m0 4.5h-4.5m4.5 0L15 15" /></svg>
                          Expand
                        </button>
                      </div>
                      <div className="max-h-72 overflow-auto">
                        <table className="min-w-full text-[11px]">
                          <thead className="sticky top-0 bg-zinc-50">
                            <tr>
                              {inputColumns.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1.5 text-left font-medium text-zinc-500">{c}</th>)}
                              {outputColumns.map((c) => <th key={c.key} className="whitespace-nowrap bg-emerald-50 px-2 py-1.5 text-left font-medium text-emerald-700">{c.label}</th>)}
                              <th className="px-2 py-1.5 text-left font-medium text-zinc-400">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {testResults.map((r, i) => (
                              <tr key={i} className="border-t border-zinc-100">
                                {inputColumns.map((c) => <td key={c} className="max-w-[120px] truncate whitespace-nowrap px-2 py-1.5 text-zinc-500">{file?.rows[i]?.[c]}</td>)}
                                {outputColumns.map((c) => <td key={c.key} className="max-w-[180px] truncate whitespace-nowrap bg-emerald-50/50 px-2 py-1.5 text-emerald-800">{r.data[c.key] || "-"}</td>)}
                                <td className="px-2 py-1.5">{r.success ? <span className="font-medium text-emerald-600">OK</span> : <span className="font-medium text-red-600" title={r.error}>Err</span>}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {testDone && !fullRunning && !fullDone && (
                    <div className="flex gap-2">
                      <button onClick={() => { setTestDone(false); setTestResults([]); }}
                        className="flex-1 rounded-lg border border-zinc-200 py-2.5 text-xs font-medium text-zinc-600 hover:bg-zinc-50">
                        Adjust & re-test
                      </button>
                      <button onClick={() => runFull(false)}
                        className="flex-1 rounded-lg bg-emerald-600 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500">
                        Looks good — Run all {file?.totalRows} rows
                      </button>
                    </div>
                  )}
                </div>
              )}

              {(fullRunning || fullDone || hasPartialRun) && (
                <div className="mt-4 space-y-3 border-t border-zinc-200 pt-4">
                  <div className="flex items-center justify-between text-xs text-zinc-500">
                    <span>
                      {fullCompleted + fullFailed} / {file?.totalRows} processed
                      {fullRunning && etaSeconds != null && (
                        <span className="ml-2 text-zinc-400">· ~{formatEta(etaSeconds)} left</span>
                      )}
                    </span>
                    <span className="tabular-nums">{fullCompleted} OK{fullPartial > 0 ? ` · ${fullPartial} with N/A` : ""} &middot; {fullFailed} failed</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-zinc-100">
                    <div className={`h-full rounded-full transition-all ${fullDone ? "bg-emerald-500" : "bg-zinc-900"}`} style={{ width: `${((fullCompleted + fullFailed) / (file?.totalRows || 1)) * 100}%` }} />
                  </div>

                  {/* Rate-limit backoff notice with live countdown */}
                  {fullRunning && rlCountdown > 0 && (
                    <div className="flex items-center gap-2 rounded-lg bg-amber-50 px-3 py-2 ring-1 ring-amber-200">
                      <svg className="h-3.5 w-3.5 shrink-0 animate-spin text-amber-500" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                      <span className="text-[11px] text-amber-700">Rate limited by Azure — backing off, retrying in <span className="font-semibold tabular-nums">{rlCountdown}s</span>. Already-enriched rows are kept.</span>
                    </div>
                  )}

                  {/* Results table preview */}
                  {fullResults.length > 0 && (
                    <div className="overflow-hidden rounded-lg border border-zinc-200">
                      <div className="flex items-center justify-between bg-zinc-50 px-2.5 py-1.5">
                        <span className="text-[11px] font-medium text-zinc-500">{fullResults.length} row{fullResults.length !== 1 ? "s" : ""} enriched</span>
                        <button onClick={() => setExpandedView("full")} className="flex items-center gap-1 text-[11px] font-medium text-zinc-500 hover:text-zinc-900 transition">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9m11.25-5.25v4.5m0-4.5h-4.5m4.5 0L15 9m-11.25 11.25v-4.5m0 4.5h4.5m-4.5 0L9 15m11.25 5.25v-4.5m0 4.5h-4.5m4.5 0L15 15" /></svg>
                          Expand
                        </button>
                      </div>
                      <div className="max-h-52 overflow-auto">
                        <table className="min-w-full text-[11px]">
                          <thead className="sticky top-0 bg-zinc-50">
                            <tr>
                              <th className="whitespace-nowrap px-2 py-1.5 text-left font-medium text-zinc-400">#</th>
                              {inputColumns.map((c) => <th key={c} className="whitespace-nowrap px-2 py-1.5 text-left font-medium text-zinc-500">{c}</th>)}
                              {outputColumns.map((c) => <th key={c.key} className="whitespace-nowrap bg-emerald-50 px-2 py-1.5 text-left font-medium text-emerald-700">{c.label}</th>)}
                              <th className="px-2 py-1.5 text-left font-medium text-zinc-400">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {fullResults.slice(0, 20).map((r) => (
                              <tr key={r.rowIndex} className="border-t border-zinc-100">
                                <td className="whitespace-nowrap px-2 py-1.5 text-zinc-400">{r.rowIndex + 1}</td>
                                {inputColumns.map((c) => <td key={c} className="max-w-[120px] truncate whitespace-nowrap px-2 py-1.5 text-zinc-500">{file?.rows[r.rowIndex]?.[c]}</td>)}
                                {outputColumns.map((c) => <td key={c.key} className="max-w-[180px] truncate whitespace-nowrap bg-emerald-50/50 px-2 py-1.5 text-emerald-800">{r.data[c.key] || "-"}</td>)}
                                <td className="px-2 py-1.5">{r.success ? <span className="font-medium text-emerald-600">OK</span> : <span className="font-medium text-red-600" title={r.error}>Err</span>}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {fullResults.length > 20 && (
                        <div className="border-t border-zinc-100 px-2.5 py-1.5 text-center">
                          <button onClick={() => setExpandedView("full")} className="text-[11px] text-zinc-500 hover:text-zinc-900">
                            View all {fullResults.length} rows
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Partial run interrupted (e.g. after a reload) — offer to resume */}
                  {hasPartialRun && (
                    <div className="space-y-3">
                      <div className="rounded-lg bg-amber-50 px-4 py-3 ring-1 ring-amber-200">
                        <p className="text-sm font-semibold text-amber-800">Run interrupted</p>
                        <p className="mt-0.5 text-xs text-amber-700">{fullResults.filter((r) => r.success).length} rows already enriched. Resume to finish the remaining {remainingRows}.</p>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => runFull(true)} disabled={!keyValid}
                          className="flex-1 rounded-lg bg-emerald-600 py-2.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50">
                          Resume — {remainingRows} rows left
                        </button>
                        <button onClick={handleDownload}
                          className="flex-1 rounded-lg border border-zinc-200 py-2.5 text-xs font-medium text-zinc-600 hover:bg-zinc-50">
                          Download results so far
                        </button>
                      </div>
                      {!keyValid && <p className="text-[11px] text-amber-600">Re-enter your API key in step 4 to resume.</p>}
                    </div>
                  )}

                  {fullRunning && (
                    <>
                      <div className="flex gap-2">
                        <button onClick={() => { pauseRef.current = !pauseRef.current; setFullPaused(!fullPaused); }}
                          className="flex-1 rounded-lg border border-zinc-200 py-2 text-xs font-medium text-zinc-600 hover:bg-zinc-50">
                          {fullPaused ? "Resume" : "Pause"}
                        </button>
                        <button onClick={() => { stopRef.current = true; pauseRef.current = false; setFullRunning(false); setFullDone(true); }}
                          className="flex-1 rounded-lg border border-red-200 py-2 text-xs font-medium text-red-600 hover:bg-red-50">Stop</button>
                      </div>
                      <button onClick={handleDownload}
                        className="w-full rounded-lg border border-zinc-200 py-2 text-xs font-medium text-zinc-600 hover:bg-zinc-50">
                        Download results so far
                      </button>
                    </>
                  )}

                  {fullDone && (
                    <div className="space-y-3">
                      <div className="rounded-lg bg-emerald-50 px-4 py-3 ring-1 ring-emerald-200">
                        <p className="text-sm font-semibold text-emerald-800">Enrichment complete</p>
                        <p className="mt-0.5 text-xs text-emerald-600">{fullCompleted} rows enriched{fullPartial > 0 ? ` (${fullPartial} with N/A values)` : ""}{fullFailed > 0 ? `, ${fullFailed} failed` : ""}. {outputColumns.length} new columns added.</p>
                      </div>
                      <button onClick={handleDownload}
                        className="w-full rounded-lg bg-emerald-600 py-3 text-sm font-semibold text-white transition hover:bg-emerald-500">
                        Download Enriched File
                      </button>
                      <button onClick={resetRun}
                        className="w-full rounded-lg border border-zinc-200 py-2.5 text-xs font-medium text-zinc-600 transition hover:bg-zinc-50">
                        Reset run — keep my setup
                      </button>
                    </div>
                  )}
                </div>
              )}
            </Card>

            {/* Disclaimer */}
            <p className="px-2 text-center text-[10px] leading-relaxed text-zinc-400">
              Disclaimer: OpenClay is provided as-is. AI-generated data may be inaccurate — always verify results. We are not responsible for the accuracy or consequences of any output. Use at your own risk.
            </p>
          </div>

          {/* ============ RIGHT COLUMN — Sidebar ============ */}
          <div className="hidden lg:block">
            <div className="sticky top-20 space-y-4">

              {/* Estimate sidebar */}
              <Card>
                {(realCostEstimate || costRange) ? (
                  <button
                    onClick={() => setEstimateCollapsed((v) => !v)}
                    className="mb-3 flex w-full items-center justify-between text-left"
                    aria-expanded={!estimateCollapsed}
                  >
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
                      {realCostEstimate ? "Precise Estimate" : "Estimate"}
                    </h3>
                    <svg className={`h-4 w-4 text-zinc-400 transition-transform ${estimateCollapsed ? "" : "rotate-180"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                  </button>
                ) : (
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-zinc-400">Estimate</h3>
                )}
                {realCostEstimate ? (
                  <div className="space-y-3">
                    {/* Total — hero treatment (always visible) */}
                    <div className="rounded-xl bg-zinc-50 px-4 py-3 text-center ring-1 ring-zinc-100">
                      <span className="block text-[10px] font-medium uppercase tracking-wider text-zinc-400">Estimated Total</span>
                      <span className="text-2xl font-bold text-zinc-900">~${realCostEstimate.totalCost.toFixed(2)}</span>
                    </div>
                    {!estimateCollapsed && (
                      <>
                        <div className="space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Rows</span><span className="font-medium text-zinc-700">{realCostEstimate.totalRows.toLocaleString()}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Model</span><span className="font-medium text-zinc-700">{realCostEstimate.modelName}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">New columns</span><span className="font-medium text-zinc-700">{outputColumns.length}</span></div>
                        </div>
                        <div className="border-t border-zinc-100 pt-2.5 space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Input tokens</span><span className="font-medium text-zinc-700">{realCostEstimate.totalInputTokens.toLocaleString()}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Output tokens</span><span className="font-medium text-zinc-700">{realCostEstimate.totalOutputTokens.toLocaleString()}</span></div>
                        </div>
                        {provider === "azure" && tpmEstimate != null && (
                          <div className="rounded-lg bg-amber-50 px-3 py-2 ring-1 ring-amber-200">
                            <div className="flex justify-between text-xs"><span className="font-medium text-amber-700">Suggested Azure TPM</span><span className="font-semibold tabular-nums text-amber-800">≥ {tpmEstimate.toLocaleString()}</span></div>
                            <p className="mt-0.5 text-[10px] text-amber-600">To run {concurrency} row{concurrency !== 1 ? "s" : ""} at once without throttling. Set your deployment&apos;s tokens-per-minute at or above this.</p>
                          </div>
                        )}
                        <div className="border-t border-zinc-100 pt-2.5 space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Platform fee</span><span className="font-semibold text-emerald-600">Free</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Input cost</span><span className="text-zinc-600">${realCostEstimate.inputCost.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Output cost</span><span className="text-zinc-600">${realCostEstimate.outputCost.toFixed(2)}</span></div>
                          {realCostEstimate.searchCost > 0 && (
                            <div className="flex justify-between"><span className="text-zinc-400">Web search</span><span className="text-zinc-600">${realCostEstimate.searchCost.toFixed(2)}</span></div>
                          )}
                        </div>
                        {realCostEstimate.freeSearchNote && <p className="text-[11px] text-emerald-600">{realCostEstimate.freeSearchNote}</p>}
                        <p className="text-center text-[10px] text-emerald-600">{fullResults.length >= 3 ? `Refined from ${fullResults.filter((r) => r.success).length} enriched rows` : "Based on your test run"}</p>
                      </>
                    )}
                  </div>
                ) : costRange ? (
                  <div className="space-y-3">
                    {/* Total range — hero treatment (always visible) */}
                    <div className="rounded-xl bg-zinc-50 px-4 py-3 text-center ring-1 ring-zinc-100">
                      <span className="block text-[10px] font-medium uppercase tracking-wider text-zinc-400">Estimated Range</span>
                      <span className="text-2xl font-bold text-zinc-900">${costRange.low.totalCost.toFixed(2)} – ${costRange.high.totalCost.toFixed(2)}</span>
                    </div>
                    {!estimateCollapsed && (
                      <>
                        <div className="space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Rows</span><span className="font-medium text-zinc-700">{costRange.low.totalRows.toLocaleString()}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Model</span><span className="font-medium text-zinc-700">{costRange.low.modelName}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">New columns</span><span className="font-medium text-zinc-700">{outputColumns.length}</span></div>
                        </div>
                        <div className="border-t border-zinc-100 pt-2.5 space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Input tokens</span><span className="font-medium text-zinc-700">{costRange.low.totalInputTokens.toLocaleString()}{costRange.high.totalInputTokens !== costRange.low.totalInputTokens ? ` – ${costRange.high.totalInputTokens.toLocaleString()}` : ""}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Output tokens</span><span className="font-medium text-zinc-700">{costRange.low.totalOutputTokens.toLocaleString()}</span></div>
                        </div>
                        <div className="border-t border-zinc-100 pt-2.5 space-y-1.5 text-xs">
                          <div className="flex justify-between"><span className="text-zinc-400">Platform fee</span><span className="font-semibold text-emerald-600">Free</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Input cost</span><span className="text-zinc-600">${costRange.low.inputCost.toFixed(2)} – ${costRange.high.inputCost.toFixed(2)}</span></div>
                          <div className="flex justify-between"><span className="text-zinc-400">Output cost</span><span className="text-zinc-600">${costRange.low.outputCost.toFixed(2)}</span></div>
                          {costRange.high.searchCost > 0 && (
                            <div className="flex justify-between"><span className="text-zinc-400">Web search</span><span className="text-zinc-600">${costRange.low.searchCost.toFixed(2)}</span></div>
                          )}
                        </div>
                        {costRange.low.freeSearchNote && <p className="text-[11px] text-emerald-600">{costRange.low.freeSearchNote}</p>}
                        <p className="text-center text-[10px] text-amber-600">{useWebSearch ? "Run a test for a precise estimate" : "Estimate based on prompt tokens"}</p>
                      </>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-zinc-400">Upload a file and describe your enrichment to see an estimate.</p>
                )}
              </Card>

              {/* Live cost incurred — real money spent so far, from actual tokens */}
              {liveCost && (
                <Card>
                  <button
                    onClick={() => setLiveCostCollapsed((v) => !v)}
                    className="mb-3 flex w-full items-center justify-between text-left"
                    aria-expanded={!liveCostCollapsed}
                  >
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Cost so far</h3>
                    <div className="flex items-center gap-2">
                      {fullRunning && (
                        <span className="flex items-center gap-1 text-[10px] font-medium text-emerald-600">
                          <span className="relative flex h-1.5 w-1.5">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
                          </span>
                          Live
                        </span>
                      )}
                      <svg className={`h-4 w-4 text-zinc-400 transition-transform ${liveCostCollapsed ? "" : "rotate-180"}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" /></svg>
                    </div>
                  </button>
                  {/* Total — always visible */}
                  <div className="rounded-xl bg-emerald-50 px-4 py-3 text-center ring-1 ring-emerald-100">
                    <span className="block text-[10px] font-medium uppercase tracking-wider text-emerald-500">Actually Incurred</span>
                    <span className="text-2xl font-bold text-emerald-700">${liveCost.totalCost.toFixed(liveCost.totalCost < 1 ? 4 : 2)}</span>
                  </div>
                  {!liveCostCollapsed && (
                    <>
                      <div className="mt-3 space-y-1.5 text-xs">
                        <div className="flex justify-between"><span className="text-zinc-400">Rows charged</span><span className="font-medium text-zinc-700">{liveCost.rowsRun.toLocaleString()}</span></div>
                        <div className="flex justify-between"><span className="text-zinc-400">Input tokens</span><span className="font-medium text-zinc-700">{liveCost.totalInputTokens.toLocaleString()}</span></div>
                        <div className="flex justify-between"><span className="text-zinc-400">Output tokens</span><span className="font-medium text-zinc-700">{liveCost.totalOutputTokens.toLocaleString()}</span></div>
                      </div>
                      <div className="mt-2.5 border-t border-zinc-100 pt-2.5 space-y-1.5 text-xs">
                        <div className="flex justify-between"><span className="text-zinc-400">Input cost</span><span className="text-zinc-600">${liveCost.inputCost.toFixed(4)}</span></div>
                        <div className="flex justify-between"><span className="text-zinc-400">Output cost</span><span className="text-zinc-600">${liveCost.outputCost.toFixed(4)}</span></div>
                        {liveCost.searchCost > 0 && (
                          <div className="flex justify-between"><span className="text-zinc-400">Web search</span><span className="text-zinc-600">${liveCost.searchCost.toFixed(4)}</span></div>
                        )}
                      </div>
                      <p className="mt-2 text-center text-[10px] text-zinc-400">Billed to your own {liveCost.modelName} account.</p>
                    </>
                  )}
                </Card>
              )}

              {/* Concurrency control — edit the slider, then press Apply to commit.
                  Applies live, even mid-run: the worker pool resizes on the fly. */}
              <Card>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">Concurrency</h3>
                  <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] font-semibold tabular-nums text-zinc-700">{concurrency}</span>
                </div>
                <p className="mb-3 text-[11px] leading-relaxed text-zinc-500">
                  How many rows are enriched at once. Higher is faster but more likely to hit your provider&apos;s rate limits.
                </p>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setConcurrency((c) => Math.max(1, c - 1))}
                    disabled={concurrency <= 1}
                    aria-label="Decrease concurrency"
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-zinc-200 text-zinc-600 transition hover:bg-zinc-50 disabled:opacity-30"
                  >
                    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" /></svg>
                  </button>
                  <input
                    type="range"
                    min={1}
                    max={MAX_CONCURRENCY}
                    value={concurrency}
                    onChange={(e) => setConcurrency(Number(e.target.value))}
                    className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-zinc-200 accent-zinc-900"
                  />
                  <button
                    onClick={() => setConcurrency((c) => Math.min(MAX_CONCURRENCY, c + 1))}
                    disabled={concurrency >= MAX_CONCURRENCY}
                    aria-label="Increase concurrency"
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-zinc-200 text-zinc-600 transition hover:bg-zinc-50 disabled:opacity-30"
                  >
                    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" /></svg>
                  </button>
                </div>
                <div className="mt-1.5 flex justify-between text-[10px] text-zinc-400">
                  <span>1 (gentle)</span>
                  <span>{MAX_CONCURRENCY} (fastest)</span>
                </div>

                {/* Apply — commits the draft. Only then does it affect the run. */}
                <button
                  onClick={() => setAppliedConcurrency(concurrency)}
                  disabled={concurrency === appliedConcurrency}
                  className="mt-3 w-full rounded-lg bg-zinc-900 py-2 text-xs font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-default disabled:bg-zinc-100 disabled:text-zinc-400"
                >
                  {concurrency === appliedConcurrency
                    ? `Active: ${appliedConcurrency} at a time`
                    : `Apply — ${concurrency} at a time`}
                </button>

                {/* Live readout: how many workers are ACTUALLY running right now.
                    If this sits below the applied count, the fleet is being held by
                    a rate-limit backoff, not by the pool size. */}
                {fullRunning && (
                  <div className="mt-2 space-y-1 rounded-lg bg-zinc-50 px-3 py-2 text-[11px] ring-1 ring-zinc-100">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500">Workers spawned</span>
                      <span className="font-semibold tabular-nums text-zinc-700">
                        {liveWorkers} / {appliedConcurrency}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500" title="Requests the backend is actually handling right now (measured on the server, not the browser).">Reaching server now</span>
                      <span className="font-semibold tabular-nums text-zinc-700">{serverInFlight}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500" title="Highest number of requests the server processed simultaneously this run.">Peak parallel (server)</span>
                      <span className={`font-semibold tabular-nums ${serverPeak <= 6 && appliedConcurrency > 6 ? "text-amber-600" : "text-emerald-600"}`}>{serverPeak}</span>
                    </div>
                    {serverPeak <= 6 && appliedConcurrency > 6 && liveWorkers > 6 && (
                      <p className="pt-1 text-[10px] leading-tight text-amber-600">
                        Peak stuck at ≤6 → the browser is capping connections (HTTP/1.1). Use <code className="rounded bg-amber-100 px-1">npm run dev:h2</code> and open the https://localhost:3443 URL.
                      </p>
                    )}
                  </div>
                )}
              </Card>

              {/* Privacy */}
              <Card>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-400">Privacy</h3>
                <ul className="space-y-1.5 text-[11px] text-zinc-500">
                  {["API key in browser memory only", "Files parsed client-side", "No database, no cookies", "100% open source"].map((item) => (
                    <li key={item} className="flex items-start gap-1.5">
                      <svg className="mt-0.5 h-3 w-3 shrink-0 text-emerald-500" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" /></svg>
                      {item}
                    </li>
                  ))}
                </ul>
              </Card>

              {/* Feedback */}
              <Card>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-400">Feedback</h3>
                <p className="text-[11px] text-zinc-500 mb-2.5">Found a bug? Have a feature idea?</p>
                <a
                  href="https://www.linkedin.com/in/-raghav/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 rounded-lg border border-zinc-100 bg-transparent px-3 py-2 text-[11px] text-zinc-400 transition hover:border-zinc-200 hover:text-zinc-600"
                >
                  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                  Connect with me
                </a>
              </Card>

              <div className="text-center text-[10px] text-zinc-400">
                <p>Pricing last updated: {PRICING_LAST_UPDATED}</p>
                <p className="mt-0.5"><a href={providerPricingUrl} target="_blank" rel="noopener noreferrer" className="text-zinc-500 underline hover:text-zinc-900">Official pricing</a></p>
                <div className="mt-2 flex items-center justify-center gap-4">
                  <Link href="/privacy" className="hover:text-zinc-900">Privacy</Link>
                  <Link href="/terms" className="hover:text-zinc-900">Terms</Link>
                  <Link href="/data" className="hover:text-zinc-900">Data</Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Expanded table modal */}
      {expandedView && (
        <div className="fixed inset-0 z-50 flex flex-col bg-white">
          <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 sm:px-6">
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-semibold text-zinc-900">
                {expandedView === "test" ? "Preview Results" : "Enrichment Results"}
              </h2>
              <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-[11px] font-medium text-zinc-500">
                {expandedView === "test" ? testResults.length : fullResults.length} row{(expandedView === "test" ? testResults.length : fullResults.length) !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {expandedView === "full" && fullResults.length > 0 && (
                <button onClick={handleDownload} className="rounded-lg border border-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-600 hover:bg-zinc-50 transition">
                  Download
                </button>
              )}
              <button onClick={() => setExpandedView(null)} className="rounded-lg border border-zinc-200 p-1.5 text-zinc-500 hover:bg-zinc-50 hover:text-zinc-900 transition">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-4 sm:p-6">
            <table className="min-w-full text-xs">
              <thead className="sticky top-0 bg-white">
                <tr>
                  {expandedView === "full" && <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-zinc-400">#</th>}
                  {inputColumns.map((c) => <th key={c} className="whitespace-nowrap px-3 py-2 text-left font-medium text-zinc-500">{c}</th>)}
                  {outputColumns.map((c) => <th key={c.key} className="whitespace-nowrap bg-emerald-50 px-3 py-2 text-left font-medium text-emerald-700">{c.label}</th>)}
                  <th className="px-3 py-2 text-left font-medium text-zinc-400">Status</th>
                </tr>
              </thead>
              <tbody>
                {(expandedView === "test" ? testResults : fullResults).map((r) => {
                  const rowIdx = expandedView === "test" ? r.rowIndex : r.rowIndex;
                  return (
                    <tr key={r.rowIndex} className="border-t border-zinc-100 hover:bg-zinc-50">
                      {expandedView === "full" && <td className="whitespace-nowrap px-3 py-2 text-zinc-400">{r.rowIndex + 1}</td>}
                      {inputColumns.map((c) => <td key={c} className="px-3 py-2 text-zinc-600 whitespace-pre-wrap break-words max-w-sm">{file?.rows[rowIdx]?.[c]}</td>)}
                      {outputColumns.map((c) => <td key={c.key} className="px-3 py-2 text-emerald-800 bg-emerald-50/30 whitespace-pre-wrap break-words max-w-md">{r.data[c.key] || "-"}</td>)}
                      <td className="px-3 py-2">{r.success ? <span className="font-medium text-emerald-600">OK</span> : <span className="font-medium text-red-600" title={r.error}>Err</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Mobile estimate bar */}
      {(realCostEstimate || costRange) && (
        <div className="fixed bottom-0 left-0 right-0 border-t border-zinc-200 bg-white/90 p-3 backdrop-blur-xl lg:hidden">
          <div className="flex items-center justify-between text-xs">
            <span className="text-zinc-500">{(realCostEstimate || costRange?.low)?.totalRows.toLocaleString()} rows &middot; {(realCostEstimate || costRange?.low)?.modelName}</span>
            <div className="flex items-center gap-3">
              <span className="text-emerald-600 text-[10px]">Platform: $0</span>
              {realCostEstimate ? (
                <span className="font-bold text-zinc-900">~${realCostEstimate.totalCost.toFixed(2)}</span>
              ) : costRange ? (
                <span className="font-bold text-zinc-900">${costRange.low.totalCost.toFixed(2)} – ${costRange.high.totalCost.toFixed(2)}</span>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
