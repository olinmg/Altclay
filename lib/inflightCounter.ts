// Shared, server-side in-flight counter for /api/enrich.
//
// This is the ONLY trustworthy measure of real request concurrency: it counts
// requests the SERVER is actively handling. A client-side counter can't tell the
// difference between a fetch that's on the wire and one still queued in the
// browser's per-host connection pool (the ~6 HTTP/1.1 cap), so it over-reports.
// The server only sees a request once the browser actually sends it — so this
// number reflects true parallelism reaching the backend.
//
// Module state persists for the life of the server process (per worker).
let inFlight = 0;
let peak = 0;

export function enter(): number {
  inFlight++;
  if (inFlight > peak) peak = inFlight;
  return inFlight;
}

export function exit(): void {
  inFlight = Math.max(0, inFlight - 1);
}

export function stats(): { inFlight: number; peak: number } {
  return { inFlight, peak };
}

export function resetPeak(): void {
  peak = inFlight;
}
