# Altclay

Spreadsheet enrichment with your own AI provider key. This project builds on [OpenClay](https://github.com/raghav3600/Altclay).

## Run locally

```bash
npm ci
npm run dev
```

Open [http://localhost:3000/tool](http://localhost:3000/tool) (or [http://localhost:3000](http://localhost:3000) for the home page).

For **more than about five parallel row requests**, use the local HTTP/2 proxy instead of `npm run dev`:

```bash
npm run dev:h2
```

Open [https://localhost:3443/tool](https://localhost:3443/tool). `http://localhost:3000/tool` also redirects there. The command starts Next.js itself on port 3010. OpenSSL is needed once to create a local certificate; accept the browser's self-signed certificate warning. During a full run, set **Concurrency** and click **Apply**; **Peak parallel (server)** shows how many requests actually reached the backend at once.

## Added in this version

- Azure OpenAI support with your endpoint, deployment, and API key, plus cost and suggested tokens-per-minute estimates.
- Adjustable concurrency (1–25 rows), live server-side concurrency and progress, and automatic backoff and retry on rate limits.
- Local HTTP/2 proxy for high concurrency development. Azure endpoint and deployment settings survive session recovery; API keys are never saved in the browser session.
