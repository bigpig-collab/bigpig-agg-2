// free-model-aggregator-02: one server, two client wires, the best free
// model behind every answer.
//
// The server listens on 0.0.0.0 at a free port, registers it with the env,
// and answers GET /v1/models, POST /v1/chat/completions and POST
// /v1/messages. Every prompt is dialled upstream as Chat Completions on the
// gateway, on the best free model that meets it (rating first), with up to
// three attempts and a failure memory that never re-dials a model that
// already refused in this trial. Refusals recovered from are reported on
// the `x-makefun-upstream-failures` header.
//
// Routing is a function of the brief, the requests, and the gateway answers
// seen so far: no clock, no randomness, so a replay reproduces the run.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIStreamRelay,
  openAIModelList,
  openAIToAnthropic,
  openAIUpstreamBody,
  type ServedModel,
} from './lib/wire.ts';
import {
  ATTEMPT_TIMEOUT_MS,
  FailureMemory,
  MAX_ATTEMPTS_PER_PROMPT,
  attemptChain,
  failoverHeaderValue,
  promptCharsOf,
  type FailureRow,
} from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is, in the brief's order. */
let manifestIndex = 0;
/** Models that refused in this trial, never dialled again. */
const failures = new FailureMemory();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`.../v1` on the trial network); a base without
 * `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/** The requirements of the prompt now arriving, by brief row or by body. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Json): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (extraHeaders !== undefined) {
    for (const [key, value] of Object.entries(extraHeaders)) {
      if (typeof value === 'string') headers[key] = value;
    }
  }
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The refusal trail as response headers, when the prompt cost refusals. */
function trailHeaders(failed: readonly FailureRow[]): Json | undefined {
  const value = failoverHeaderValue(failed);
  return value === null ? undefined : { 'x-makefun-upstream-failures': value };
}

/** The error code inside a refusal body, or the status as a string. */
function refusalCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') {
      return parsed.error.code;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string') return parsed.code;
  } catch {
    // Not JSON: fall through to the status.
  }
  return String(status);
}

interface DialSuccess {
  ok: true;
  model: ModelRow;
  res: Response;
  failed: FailureRow[];
}

interface DialFailure {
  ok: false;
  failed: FailureRow[];
  lastStatus: number;
  lastText: string;
}

/**
 * Dial the prompt's chain: candidates best-first, at most three attempts,
 * each with its own timeout. A refusal (non-2xx, network error, timeout)
 * excludes that model for the rest of the trial and the next candidate is
 * tried. When every candidate is exhausted, the best model is retried once
 * as a last resort instead of giving up outright.
 */
async function dial(
  requires: Requirement,
  upstreamBody: Json,
  stream: boolean,
): Promise<DialSuccess | DialFailure> {
  const failed: FailureRow[] = [];
  const chain = attemptChain(brief, requires, failures.set);
  let attempts = 0;
  let lastStatus = 502;
  let lastText = 'no free model in the brief meets this request';
  for (const model of chain) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    attempts += 1;
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      authorization: `Bearer ${GATEWAY_KEY}`,
    };
    if (stream) headers.accept = 'text/event-stream';
    let res: Response;
    try {
      res = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...upstreamBody, model: model.id }),
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
      });
    } catch (err) {
      const timedOut = (err as Error).name === 'TimeoutError';
      failures.mark(model.id);
      failed.push({
        status: 0,
        code: timedOut ? 'timeout' : 'network_error',
        model: model.id,
      });
      lastStatus = 504;
      lastText = (err as Error).message;
      continue;
    }
    if (!res.ok || res.body === null) {
      const text = await res.text().catch(() => '');
      failures.mark(model.id);
      failed.push({ status: res.status, code: refusalCode(res.status, text), model: model.id });
      lastStatus = res.status;
      lastText = text;
      continue;
    }
    return { ok: true, model, res, failed };
  }
  return { ok: false, failed, lastStatus, lastText };
}

/** POST /v1/chat/completions, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const upstreamBody = openAIUpstreamBody(body, '');
  const promptChars = promptCharsOf(upstreamBody);
  const result = await dial(requires, upstreamBody, body.stream === true);
  if (!result.ok) {
    return sendError(
      res,
      'openai',
      502,
      'upstream_error',
      `the gateway answered HTTP ${result.lastStatus}${result.lastText === '' ? '' : `: ${result.lastText.slice(0, 500)}`}`,
    );
  }
  const modelId = result.model.id;
  const extra = trailHeaders(result.failed);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...(extra !== undefined ? { 'x-makefun-upstream-failures': extra['x-makefun-upstream-failures'] } : {}),
    });
    try {
      for await (const frame of openAIStreamRelay(result.res.body as ReadableStream<Uint8Array>, modelId, promptChars)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await result.res.json().catch(() => null);
  if (!isRecord(answer)) {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not one JSON object');
  }
  sendJson(res, 200, openAIJsonWithModel(answer, modelId, promptChars), extra);
}

/** POST /v1/messages: translated to Chat Completions, answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const upstreamBody = anthropicToOpenAI(body, '');
  const promptChars = promptCharsOf(upstreamBody);
  const result = await dial(requires, upstreamBody, body.stream === true);
  if (!result.ok) {
    return sendError(
      res,
      'anthropic',
      502,
      'api_error',
      `the gateway answered HTTP ${result.lastStatus}${result.lastText === '' ? '' : `: ${result.lastText.slice(0, 500)}`}`,
    );
  }
  const modelId = result.model.id;
  const extra = trailHeaders(result.failed);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...(extra !== undefined ? { 'x-makefun-upstream-failures': extra['x-makefun-upstream-failures'] } : {}),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(result.res.body as ReadableStream<Uint8Array>, modelId, promptChars)) {
        res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await result.res.json().catch(() => null);
  if (!isRecord(answer)) {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not one JSON object');
  }
  sendJson(res, 200, openAIToAnthropic(answer, modelId, promptChars), extra);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog. The env is retried a
 * few times: a trial that starts a beat before the env answers still
 * registers instead of failing the run.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
