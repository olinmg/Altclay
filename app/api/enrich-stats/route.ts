import { NextResponse } from "next/server";
import { stats, resetPeak } from "@/lib/inflightCounter";

// Reports true server-side concurrency for /api/enrich. The UI polls this during
// a run to show how many requests are ACTUALLY reaching the backend in parallel
// (as opposed to a client counter, which also counts browser-queued fetches).
export async function GET() {
  return NextResponse.json(stats());
}

// POST resets the peak — called by the UI when a fresh run starts.
export async function POST() {
  resetPeak();
  return NextResponse.json(stats());
}
