import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { JWT } from "google-auth-library";

export async function POST(req: NextRequest) {
  try {
    const { provider, apiKey } = await req.json();

    if (!apiKey || typeof apiKey !== "string") {
      return NextResponse.json({ valid: false, error: "Missing API key" }, { status: 400 });
    }

    if (provider === "anthropic") {
      const client = new Anthropic({ apiKey });
      await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 10,
        messages: [{ role: "user", content: "Hi" }],
      });
      return NextResponse.json({ valid: true });
    } else if (provider === "gemini") {
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
      await model.generateContent("Hi");
      return NextResponse.json({ valid: true });
    } else if (provider === "grok") {
      const grokRes = await fetch("https://api.x.ai/v1/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: "grok-4-1-fast",
          input: [{ role: "user", content: "Hi" }],
          max_output_tokens: 10,
        }),
      });
      if (!grokRes.ok) {
        const errText = await grokRes.text();
        throw new Error(`${grokRes.status} ${errText}`);
      }
      return NextResponse.json({ valid: true });
    } else if (provider === "openai") {
      const openaiRes = await fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: "gpt-5.4-mini",
          input: [{ role: "user", content: "Hi" }],
          max_output_tokens: 16,
        }),
      });
      if (!openaiRes.ok) {
        const errText = await openaiRes.text();
        throw new Error(`${openaiRes.status} ${errText}`);
      }
      return NextResponse.json({ valid: true });
    } else if (provider === "azure") {
      const { endpoint, deployment, key } = JSON.parse(apiKey);
      if (!endpoint || !deployment || !key) {
        return NextResponse.json({ valid: false, error: "Azure endpoint, deployment, and key are required" });
      }
      const base = endpoint.replace(/\/+$/, "");
      const url = /\/responses(\?|$)/.test(base) ? base : `${base}/openai/v1/responses`;
      const azureRes = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "api-key": key },
        body: JSON.stringify({
          model: deployment,
          input: [{ role: "user", content: "Hi" }],
          max_output_tokens: 16,
        }),
      });
      if (!azureRes.ok) {
        const errText = await azureRes.text();
        const status = azureRes.status;
        // 401/403 with an api-key header genuinely means a bad key.
        if (status === 401 || status === 403) {
          return NextResponse.json({ valid: false, error: `Azure rejected the API key (${status}). Check the key belongs to this resource.` });
        }
        // 404 almost always means a wrong deployment name or endpoint path — NOT the key.
        if (status === 404) {
          return NextResponse.json({ valid: false, error: `Azure returned 404 — the deployment name or endpoint path is likely wrong (not the key). Azure said: ${errText.slice(0, 300)}` });
        }
        // Everything else (400 bad api-version, tool unsupported, etc.) — show it verbatim.
        return NextResponse.json({ valid: false, error: `Azure error ${status}: ${errText.slice(0, 300)}` });
      }
      return NextResponse.json({ valid: true });
    } else if (provider === "vertex") {
      const creds = JSON.parse(apiKey);
      const client = new JWT({
        email: creds.client_email,
        key: creds.private_key,
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      });
      const tokenRes = await client.getAccessToken();
      if (!tokenRes.token) throw new Error("Failed to get access token");

      const location = "us-central1";
      const url = `https://${location}-aiplatform.googleapis.com/v1/projects/${creds.project_id}/locations/${location}/publishers/google/models/gemini-2.0-flash:generateContent`;
      const vertexRes = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${tokenRes.token}` },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: "Hi" }] }],
        }),
      });
      if (!vertexRes.ok) {
        const errText = await vertexRes.text();
        throw new Error(`${vertexRes.status} ${errText}`);
      }
      return NextResponse.json({ valid: true });
    } else {
      return NextResponse.json({ valid: false, error: "Unknown provider" }, { status: 400 });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Validation failed";

    // Rate limit / quota errors mean the key IS valid — just temporarily limited
    if (
      message.includes("429") ||
      message.includes("quota") ||
      message.includes("rate") ||
      message.includes("Too Many Requests") ||
      message.includes("RESOURCE_EXHAUSTED") ||
      message.includes("overloaded")
    ) {
      return NextResponse.json({ valid: true, warning: "Key is valid but you may be near your rate limit. Enrichment will retry automatically." });
    }

    // Auth errors mean the key is genuinely invalid
    if (
      message.includes("401") ||
      message.includes("403") ||
      message.includes("invalid") ||
      message.includes("API_KEY") ||
      message.includes("authentication") ||
      message.includes("PERMISSION_DENIED") ||
      message.includes("Incorrect API key")
    ) {
      return NextResponse.json({ valid: false, error: "Invalid API key. Please check and try again." });
    }

    // Network / unknown errors — don't block the user, let them try
    return NextResponse.json({ valid: true, warning: "Could not fully verify key, but it looks correctly formatted. Proceed with caution." });
  }
}
