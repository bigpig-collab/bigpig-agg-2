// Brief parsing, the free-model catalog, and requirement checks.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is prose with two JSON documents in
// it, found by their `schema` member under this contest's prefix. This module
// finds them by scanning for balanced JSON objects, never by parsing prose.
//
//   - models (`free-model-aggregator-02/models/v1`): one row per free model,
//     keyed by the gateway id (`openrouter/<vendor>/<model>:free`), with the
//     published rating in milli-points, capabilities, and the free flag.
//   - prompts (`free-model-aggregator-02/prompts/v1`): the six prompts in the
//     order the client sends them, each with its requirements.
//
// Everything here is a pure function of the brief text: no clock, no
// randomness, so `--list` and the register body build the same document.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The schema prefix every document in this contest's brief carries. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** OpenRouter's public endpoint: the catalog names it, never the gateway. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The env var the entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model row exactly as the brief lists it. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  /** Published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt row, in the client's arrival order. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as read: models, prompts, and a parse error when one is set. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model row of the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider of the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** The catalog: what `--list` prints and what /register carries. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** A model's capabilities, as the brief's row states them. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * Index of the `}` closing the object that opens at `start`, or -1 when the
 * text ends first. String literals (with escapes) are skipped.
 */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    } catch {
      // A `{` that is not a standalone JSON object: keep scanning.
    }
  }
  return found;
}

/**
 * First document under this contest's prefix carrying one of `keys` as an
 * array, with the key it was found under. Accepts the sibling contest's
 * spellings (`ratings`, `manifest`) as well as this contest's.
 */
function documentWithArray(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id` (or `model`); a bare upstream id
 * under provider `openrouter` is spelled the way the gateway routes it.
 * A row is free unless it says `free: false`; a paid row is never routed.
 */
function modelRowOf(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const rawId = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  const raw = rawId.trim();
  if (raw === '') return null;
  const named = asString(value.provider) !== '' ? asString(value.provider) : asString(value.wire);
  const id =
    named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(value.display_name) !== '' ? asString(value.display_name) : raw,
    rating_milli: asNumberOrNull(value.rating_milli) ?? 0,
    context_length: asNumberOrNull(value.context_length),
    tools: asBoolOrNull(value.tools),
    vision: asBoolOrNull(value.vision),
    json: asBoolOrNull(value.json),
    free: value.free !== false,
  };
}

function promptRowOf(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const req = isRecord(value.requires) ? value.requires : {};
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
        ? { min_context: req.min_context }
        : {}),
    },
  };
}

/** Read the brief at `path`, finding its two documents by schema. */
export function readBrief(path: string): Brief {
  if (path === '') {
    return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = documentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const row = modelRowOf(v);
    return row === null ? [] : [row];
  });
  const promptsDoc = documentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const row = promptRowOf(v);
    return row === null ? [] : [row];
  });
  return { models, prompts, error: null };
}

/** The brief's free models, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** A model's capabilities, as the brief's row states them. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/**
 * Whether a model may serve a prompt. Only JSON mode and tools constrain a
 * route in this contest: vision and context length never block one.
 * An unknown capability never meets a requirement that names it.
 */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.json === true && caps.json !== true) return false;
  if (requires.tools === true && caps.tools !== true) return false;
  return true;
}

/**
 * The catalog: one provider, the brief's free models in brief order,
 * deduped. The base_url is OpenRouter's public endpoint, never the trial
 * gateway's address, so the document is identical on a live pass and a
 * replay. With no free models the catalog is empty.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}
