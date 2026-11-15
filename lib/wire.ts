// The two client wires, the one upstream wire, and the model field.
//
// The client calls this entry as OpenAI (POST /v1/chat/completions) or as
// Anthropic (POST /v1/messages), JSON or SSE. Every prompt is dialled
// upstream as Chat Completions only; Anthropic requests are translated
// there and the answer translated back.
//
// The `model` field always names the id actually dialled: the provider
// spells it without the gateway's `openrouter/` prefix, and these functions
// put the dialled id back on every answer and every streamed chunk.
//
// Usage is never left short: when an upstream answer carries no whole usage
// (input >= 0, output >= 1), usage is synthesized from the relayed text, so
// no answer ever reports zero output tokens.
//
// Everything here is a pure function of its arguments except the stream
// relays, which are functions of the upstream bytes: no clock, no
// randomness. Listed-model timestamps are fixed for the same reason.

import { isRecord, type Json } from './providers.ts';
import { synthesizeUsage, usageIsWhole } from './plan.ts';

/** One model as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** Stop reasons each wire accepts. */
const OPENAI_STOP_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
const ANTHROPIC_STOP_REASONS = new Set([
  'end_turn',
  'max_tokens',
  'stop_sequence',
  'tool_use',
  'refusal',
  'pause_turn',
]);

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string, or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part; null to drop it. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object; {} when they are not JSON. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function legalOpenAIStop(finish: unknown): string {
  return typeof finish === 'string' && OPENAI_STOP_REASONS.has(finish) ? finish : 'stop';
}

function legalAnthropicStop(reason: unknown): string {
  return typeof reason === 'string' && ANTHROPIC_STOP_REASONS.has(reason) ? reason : 'end_turn';
}

/**
 * An Anthropic Messages request as a Chat Completions request naming
 * `model`: system becomes a system message, images become image_url parts,
 * tool_use blocks become tool_calls, tool_result blocks become tool
 * messages, tools and tool_choice are rewritten. The output cap is always
 * present and at most 8192; `store` is never sent.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : 1024;
  out.max_tokens = Math.min(8192, Math.max(1, Math.floor(cap)));
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI-shaped client body prepared for the gateway: the dialled model
 * in, `store` out, and the output cap always present and at most 8192.
 */
export function openAIUpstreamBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.store;
  const cap =
    typeof out.max_tokens === 'number'
      ? out.max_tokens
      : typeof out.max_completion_tokens === 'number'
        ? out.max_completion_tokens
        : 1024;
  out.max_tokens = Math.min(8192, Math.max(1, Math.floor(cap)));
  return out;
}

/** Whole usage for a JSON answer: the upstream's when whole, else synthesized. */
function wholeUsage(body: Json, promptChars: number, textChars: number): Json {
  if (usageIsWhole(body.usage)) {
    const usage = body.usage as Json;
    return {
      prompt_tokens: usage.prompt_tokens,
      completion_tokens: usage.completion_tokens,
      total_tokens:
        typeof usage.total_tokens === 'number'
          ? usage.total_tokens
          : (usage.prompt_tokens as number) + (usage.completion_tokens as number),
    };
  }
  return synthesizeUsage(promptChars, textChars);
}

/** A Chat Completions answer with `model` the id dialled and usage whole. */
export function openAIJsonWithModel(body: Json, model: string, promptChars: number): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const fixed: Json = {
    ...body,
    model,
    choices: choices.map((c, i) => {
      if (!isRecord(c) || i !== 0) return c;
      return { ...c, finish_reason: legalOpenAIStop(c.finish_reason) };
    }),
    usage: wholeUsage(body, promptChars, text.length),
  };
  return fixed;
}

/** A Chat Completions answer as an Anthropic Messages object, usage whole. */
export function openAIToAnthropic(body: Json, model: string, promptChars: number): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = wholeUsage(body, promptChars, text.length);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: legalAnthropicStop(stopReasonOf(choice.finish_reason)),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event: an `event:` name when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split streamed text into SSE frames on blank lines, holding a partial tail. */
function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const frames: string[] = [];
  let at = buffer.indexOf('\n\n');
  while (at !== -1) {
    frames.push(buffer.slice(0, at));
    buffer = buffer.slice(at + 2);
    at = buffer.indexOf('\n\n');
  }
  return { frames, rest: buffer };
}

/** The data payloads of one frame (JSON-parsed where they parse). */
function dataPayloads(frame: string): unknown[] {
  const out: unknown[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '[DONE]') {
      out.push('[DONE]');
      continue;
    }
    try {
      out.push(JSON.parse(payload));
    } catch {
      // Not JSON: not a chunk this relay rewrites.
    }
  }
  return out;
}

/**
 * A Chat Completions SSE stream relayed with `model` on every chunk:
 * exactly one finish chunk (usage attached when the upstream's is missing
 * or short, synthesized from the relayed text), exactly one `data: [DONE]`
 * as the last frame, nothing after it.
 */
export async function* openAIStreamRelay(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  promptChars: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let textChars = 0;
  let created: number = FIXED_OPENAI_CREATED;
  let seenFinish = false;
  let doneSent = false;

  const finishChunk = (usage: Json): string =>
    `data: ${JSON.stringify({
      id: 'chatcmpl-relay',
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      usage,
    })}\n\n`;

  /** One upstream frame relayed, or the closing frames when it says [DONE]. */
  const relayFrame = (frame: string): string[] => {
    const out: string[] = [];
    const payloads = dataPayloads(frame);
    if (payloads.includes('[DONE]')) {
      if (!seenFinish) {
        out.push(finishChunk(synthesizeUsage(promptChars, textChars)));
        seenFinish = true;
      }
      out.push('data: [DONE]\n\n');
      doneSent = true;
      return out;
    }
    for (const payload of payloads) {
      if (!isRecord(payload)) continue;
      if (typeof payload.created === 'number') created = payload.created;
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') textChars += delta.content.length;
      const finish = choice.finish_reason;
      if (finish !== null && finish !== undefined) {
        if (seenFinish) continue; // Only one finish chunk may go out.
        seenFinish = true;
        const usage = usageIsWhole(payload.usage)
          ? (payload.usage as Json)
          : synthesizeUsage(promptChars, textChars);
        out.push(
          `data: ${JSON.stringify({ ...payload, model, choices: [{ ...choice, finish_reason: legalOpenAIStop(finish) }], usage })}\n\n`,
        );
        continue;
      }
      if (isRecord(payload.usage)) {
        const usage = usageIsWhole(payload.usage)
          ? (payload.usage as Json)
          : synthesizeUsage(promptChars, textChars);
        out.push(`data: ${JSON.stringify({ ...payload, model, usage })}\n\n`);
        continue;
      }
      out.push(`data: ${JSON.stringify({ ...payload, model })}\n\n`);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const split = splitFrames(buffer);
      buffer = split.rest;
      for (const frame of split.frames) {
        for (const event of relayFrame(frame)) yield event;
        if (doneSent) return;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of relayFrame(buffer)) yield event;
      if (doneSent) return;
    }
  } finally {
    reader.releaseLock();
  }
  if (!doneSent) {
    if (!seenFinish) yield finishChunk(synthesizeUsage(promptChars, textChars));
    yield 'data: [DONE]\n\n';
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence:
 * message_start, content_block_start, content_block_delta*,
 * content_block_stop, message_delta, message_stop — nothing after
 * message_stop. message_delta's output_tokens is the upstream usage when
 * whole, else synthesized from the accumulated text.
 */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  promptChars: number,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textChars = 0;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = Math.max(0, Math.ceil(promptChars / 4));
  let outputTokens: number | null = null;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };

  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    const out =
      outputTokens === null
        ? synthesizeUsage(promptChars, textChars).completion_tokens
        : outputTokens;
    return [
      ...closeBlocks(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: legalAnthropicStop(stopReason), stop_sequence: null },
        usage: { output_tokens: Math.max(1, out) },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One upstream frame's lines as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const payload of dataPayloads(frame)) {
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      if (!isRecord(payload)) continue;
      if (isRecord(payload.usage)) {
        if (typeof payload.usage.prompt_tokens === 'number') {
          inputTokens = Math.max(0, Math.floor(payload.usage.prompt_tokens));
        }
        if (typeof payload.usage.completion_tokens === 'number') {
          outputTokens = Math.max(0, Math.floor(payload.usage.completion_tokens));
        }
      }
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        textChars += delta.content.length;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            }),
          );
        }
        out.push(
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: delta.content },
          }),
        );
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(
              sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }),
            );
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(
            sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index,
              delta: { type: 'input_json_delta', partial_json: fn.arguments },
            }),
          );
        }
      }
      // Hold the stop reason until the usage chunk and [DONE] arrive: the
      // usage travels after the finish_reason, and message_delta carries it.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const split = splitFrames(buffer);
      buffer = split.rest;
      for (const frame of split.frames) {
        for (const event of eventsOfFrame(frame)) yield event;
        if (finished) return;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
      if (finished) return;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
