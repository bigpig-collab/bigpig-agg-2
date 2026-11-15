// Routing, failure memory, the failover trail, and usage synthesis.
//
// The router ranks every free model that meets a prompt's requirements by
// rating (best first, ties by id ascending, so the order is a pure function
// of the brief). A model that fails once is excluded for the rest of the
// trial: no call is ever spent on a model that already refused. At most
// three attempts are spent on one prompt.
//
// Everything here is a pure function of the brief, the requests, and the
// gateway answers seen so far: no clock, no randomness.

import {
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  type Brief,
  type Json,
  type ModelRow,
  type Requirement,
} from './providers.ts';

/** One refused upstream call, as the failover header reports it. */
export interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** How many upstream attempts one prompt may cost. */
export const MAX_ATTEMPTS_PER_PROMPT = 3;

/** Per-attempt gateway timeout, in milliseconds (3 attempts fit the 30 s client cap). */
export const ATTEMPT_TIMEOUT_MS = 10000;

/**
 * Candidates for a prompt, best first: free models meeting the requirements
 * that have not failed yet in this trial, rating descending, id ascending.
 */
export function rankCandidates(
  brief: Brief,
  requires: Requirement,
  failed: ReadonlySet<string>,
): ModelRow[] {
  const rows = brief.models.filter(
    (m) => m.free && !failed.has(m.id) && meetsRequirements(requires, capabilitiesOf(m)),
  );
  rows.sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return rows;
}

/** The single best candidate ignoring failures: the last-resort retry. */
export function bestCandidate(brief: Brief, requires: Requirement): ModelRow | null {
  const rows = rankCandidates(brief, requires, new Set<string>());
  return rows.length > 0 ? rows[0] : null;
}

/**
 * The attempt chain for one prompt: the ranked candidates, or the single
 * best model as a last resort when every eligible model already failed.
 * The caller spends at most MAX_ATTEMPTS_PER_PROMPT of this chain.
 */
export function attemptChain(
  brief: Brief,
  requires: Requirement,
  failed: ReadonlySet<string>,
): ModelRow[] {
  const chain = rankCandidates(brief, requires, failed);
  if (chain.length > 0) return chain;
  const best = bestCandidate(brief, requires);
  return best === null ? [] : [best];
}

/** The models that have failed in this trial, so they are never re-dialled. */
export class FailureMemory {
  private failed = new Set<string>();

  mark(modelId: string): void {
    this.failed.add(modelId);
  }

  has(modelId: string): boolean {
    return this.failed.has(modelId);
  }

  get set(): ReadonlySet<string> {
    return this.failed;
  }
}

/**
 * The `x-makefun-upstream-failures` header value for a prompt's refusals, or
 * null when there were none. At most 8 rows, at most 8 KiB; rows are dropped
 * from the tail until the value fits.
 */
export function failoverHeaderValue(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, 8);
  while (rows.length > 0) {
    const text = JSON.stringify(rows);
    if (Buffer.byteLength(text) <= 8 * 1024) return text;
    rows.pop();
  }
  return null;
}

/** Characters of prompt text inside an upstream Chat Completions body. */
export function promptCharsOf(body: Json): number {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let chars = 0;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (typeof content === 'string') chars += content.length;
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (isRecord(part) && typeof part.text === 'string') chars += part.text.length;
      }
    }
  }
  return chars;
}

/** Usage synthesized from relayed text when the upstream answer has none. */
export interface SynthesizedUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export function synthesizeUsage(promptChars: number, textChars: number): SynthesizedUsage {
  const prompt_tokens = Math.max(0, Math.ceil(promptChars / 4));
  const completion_tokens = Math.max(1, Math.ceil(textChars / 4));
  return { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens };
}

/** Whether an upstream usage object is whole: input >= 0, output >= 1. */
export function usageIsWhole(usage: unknown): usage is {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
} {
  if (!isRecord(usage)) return false;
  const input = usage.prompt_tokens;
  const output = usage.completion_tokens;
  return (
    typeof input === 'number' &&
    Number.isInteger(input) &&
    input >= 0 &&
    typeof output === 'number' &&
    Number.isInteger(output) &&
    output >= 1
  );
}
