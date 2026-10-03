/**
 * Local HTTP/2 dev proxy.
 *
 * WHY THIS EXISTS
 * ---------------
 * `next dev` (and `next start`) serve plain HTTP/1.1. Browsers allow only ~6
 * concurrent connections per origin over HTTP/1.1, and each connection carries
 * one request at a time — so the app's row-enrichment worker pool is silently
 * capped at ~6 in-flight requests no matter what the concurrency slider says.
 * (This is a browser rule, not something a website can raise.)
 *
 * HTTP/2 fixes it by MULTIPLEXING many requests over a single connection, so the
 * 6-per-host cap becomes irrelevant and all N workers actually run in parallel —
 * matching how Vercel serves the app in production.
 *
 * This script terminates HTTP/2 + TLS for the browser and reverse-proxies every
 * request (and the HMR websocket) to `next dev` on HTTP/1.1 over loopback, where
 * the connection cap doesn't matter because it's server-to-server, not browser.
 *
 * The app code is untouched. Run `npm run dev:h2` and open the HTTPS URL it prints.
 */
import http2 from "node:http2";
import http from "node:http";
import net from "node:net";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const CERT_DIR = join(ROOT, ".dev-certs");
const KEY_PATH = join(CERT_DIR, "dev-key.pem");
const CERT_PATH = join(CERT_DIR, "dev-cert.pem");

const UPSTREAM_HOST = "127.0.0.1";
// `next dev` runs on an INTERNAL port (not 3000) so that port 3000 — the address
// muscle-memory sends the browser to — is free for the redirect listener below.
const UPSTREAM_PORT = Number(process.env.NEXT_DEV_PORT || 3010); // where `next dev` listens
const LISTEN_PORT = Number(process.env.H2_PORT || 3443); // HTTPS port the browser uses
// Plain-HTTP port the user actually types (http://localhost:3000). We bounce it
// to the HTTPS proxy so "localhost:3000" lands on the fast HTTP/2 page with no
// need to type the scheme. Set REDIRECT_PORT=0 to disable the redirect listener.
const REDIRECT_PORT = Number(process.env.H2_REDIRECT_PORT ?? 3000);
// When true (the default for `npm run dev:h2`), this script also launches
// `next dev` itself so one command gives a fully-working HTTP/2 dev environment.
const SPAWN_NEXT = process.env.H2_SPAWN_NEXT !== "0";

/* ---- 1. Ensure a self-signed cert exists (generated once, gitignored) ---- */
function ensureCert() {
  if (existsSync(KEY_PATH) && existsSync(CERT_PATH)) return;
  mkdirSync(CERT_DIR, { recursive: true });
  console.log("[h2-proxy] Generating a self-signed dev certificate (one time)...");
  const res = spawnSync(
    "openssl",
    [
      "req", "-x509", "-newkey", "rsa:2048", "-nodes",
      "-keyout", KEY_PATH,
      "-out", CERT_PATH,
      "-days", "3650",
      "-subj", "/CN=localhost",
      "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ],
    { stdio: "inherit" }
  );
  if (res.status !== 0) {
    console.error("[h2-proxy] Failed to generate certificate. Is openssl installed?");
    process.exit(1);
  }
}

/* ---- 2. Forward one incoming request to `next dev` over HTTP/1.1 ---- */
function proxyRequest(headers, method, path, bodyStream, respond) {
  // Strip HTTP/2 pseudo-headers (":method", ":path", ...) before sending to H1.
  const outHeaders = {};
  for (const [k, v] of Object.entries(headers)) {
    if (!k.startsWith(":")) outHeaders[k] = v;
  }
  outHeaders.host = `${UPSTREAM_HOST}:${UPSTREAM_PORT}`;

  const upstream = http.request(
    { host: UPSTREAM_HOST, port: UPSTREAM_PORT, method, path, headers: outHeaders },
    (upRes) => respond(upRes)
  );
  upstream.on("error", (err) => {
    console.error(`[h2-proxy] upstream error for ${method} ${path}:`, err.message);
    respond(null, err);
  });
  bodyStream.pipe(upstream);
}

/* ---- 3a. Plain-HTTP redirect: http://localhost:3000 -> https://localhost:3443 ---- */
// Lets the user type "localhost:3000" (which the browser turns into http://) and
// still land on the multiplexed HTTPS page. Preserves the path + query string.
function startRedirect() {
  if (!REDIRECT_PORT) return;
  const redirectServer = http.createServer((req, res) => {
    // Use the Host the browser actually used (localhost / 127.0.0.1), swapping the
    // port for the HTTPS one, so the cert's CN=localhost keeps matching.
    const hostname = (req.headers.host || `localhost:${REDIRECT_PORT}`).split(":")[0];
    const location = `https://${hostname}:${LISTEN_PORT}${req.url || "/"}`;
    res.writeHead(302, { Location: location });
    res.end(`Redirecting to ${location}`);
  });
  redirectServer.on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.error(`[h2-proxy] Port ${REDIRECT_PORT} is in use — the http://localhost:${REDIRECT_PORT} redirect is disabled.`);
      console.error(`[h2-proxy] Open https://localhost:${LISTEN_PORT} directly instead.`);
      return; // non-fatal; the HTTPS proxy still works
    }
    console.error("[h2-proxy] redirect server error:", err.message);
  });
  redirectServer.listen(REDIRECT_PORT, () => {
    console.log(`  \x1b[36m→ or open  http://localhost:${REDIRECT_PORT}\x1b[0m  (auto-redirects to the HTTPS page)`);
  });
}

/* ---- 3. HTTP/2 server (with HTTP/1.1 fallback via ALPN) ---- */
function start() {
  startRedirect();
  const server = http2.createSecureServer({
    key: readFileSync(KEY_PATH),
    cert: readFileSync(CERT_PATH),
    allowHTTP1: true, // fall back gracefully for clients/tools that don't do H2
  });

  // HTTP/2 request path.
  server.on("stream", (stream, headers) => {
    const method = headers[":method"];
    const path = headers[":path"];
    proxyRequest(headers, method, path, stream, (upRes, err) => {
      if (err || !upRes) {
        stream.respond({ ":status": 502 });
        stream.end("Bad gateway: next dev not reachable on port " + UPSTREAM_PORT);
        return;
      }
      const respHeaders = { ":status": upRes.statusCode };
      for (const [k, v] of Object.entries(upRes.headers)) {
        // Connection-specific headers are illegal in HTTP/2.
        if (["connection", "transfer-encoding", "keep-alive", "upgrade"].includes(k.toLowerCase())) continue;
        respHeaders[k] = v;
      }
      stream.respond(respHeaders);
      upRes.pipe(stream);
    });
  });

  // HTTP/1.1 fallback request path (same-origin clients negotiating H1).
  server.on("request", (req, res) => {
    if (req.httpVersionMajor >= 2) return; // handled by 'stream'
    proxyRequest(req.headers, req.method, req.url, req, (upRes, err) => {
      if (err || !upRes) {
        res.writeHead(502);
        res.end("Bad gateway: next dev not reachable on port " + UPSTREAM_PORT);
        return;
      }
      res.writeHead(upRes.statusCode, upRes.headers);
      upRes.pipe(res);
    });
  });

  // HMR / fast-refresh websocket: proxy the raw upgrade to next dev.
  server.on("upgrade", (req, socket, head) => {
    const upstream = net.connect(UPSTREAM_PORT, UPSTREAM_HOST, () => {
      const headerLines = [
        `${req.method} ${req.url} HTTP/1.1`,
        ...Object.entries(req.headers).map(([k, v]) => `${k}: ${v}`),
        "", "",
      ].join("\r\n");
      upstream.write(headerLines);
      if (head && head.length) upstream.write(head);
      socket.pipe(upstream).pipe(socket);
    });
    upstream.on("error", () => socket.destroy());
    socket.on("error", () => upstream.destroy());
  });

  server.on("error", (err) => {
    console.error("[h2-proxy] server error:", err.message);
    process.exit(1);
  });

  server.listen(LISTEN_PORT, () => {
    console.log("");
    console.log("  \x1b[32m▲ HTTP/2 dev proxy ready\x1b[0m");
    console.log(`  \x1b[36m→ open  https://localhost:${LISTEN_PORT}\x1b[0m  (multiplexed — no 6-connection cap)`);
    console.log(`    proxying to next dev on http://${UPSTREAM_HOST}:${UPSTREAM_PORT}`);
    console.log("");
    console.log("  First visit shows a self-signed-cert warning — click \"Advanced → proceed\". It's your own local cert.");
    console.log("");
  });
}

/* ---- 4. Optionally launch `next dev` and wait until it accepts connections ---- */
function waitForUpstream(retries = 100) {
  return new Promise((resolve, reject) => {
    const attempt = (left) => {
      const sock = net.connect(UPSTREAM_PORT, UPSTREAM_HOST);
      sock.on("connect", () => { sock.destroy(); resolve(); });
      sock.on("error", () => {
        sock.destroy();
        if (left <= 0) return reject(new Error("next dev never came up"));
        setTimeout(() => attempt(left - 1), 300);
      });
    };
    attempt(retries);
  });
}

async function main() {
  ensureCert();

  let nextProc = null;
  if (SPAWN_NEXT) {
    console.log(`[h2-proxy] Starting \`next dev\` on port ${UPSTREAM_PORT}...`);
    nextProc = spawn("npx", ["next", "dev", "-p", String(UPSTREAM_PORT)], {
      cwd: ROOT,
      stdio: "inherit",
      shell: true, // needed for npx resolution on Windows
    });
    const shutdown = () => { if (nextProc) nextProc.kill(); process.exit(0); };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
    nextProc.on("exit", (code) => { console.log(`[h2-proxy] next dev exited (${code}).`); process.exit(code ?? 0); });

    try {
      await waitForUpstream();
    } catch {
      console.error("[h2-proxy] Timed out waiting for next dev. Exiting.");
      if (nextProc) nextProc.kill();
      process.exit(1);
    }
  }

  start();
}

main();
