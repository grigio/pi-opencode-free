// SPDX-License-Identifier: GPL-3.0-or-later
import type { Api, Model, ProviderHeaders } from "@earendil-works/pi-ai";
import {
  getApiProvider,
  registerApiProvider,
  type ApiProvider,
  type ApiStreamFunction,
  type ApiStreamSimpleFunction,
} from "@earendil-works/pi-ai/compat";
import {
  applyOpenCodeFreeHeaders,
  getHeader,
  type MutableHeaders,
} from "./zen-headers.js";

/**
 * Zen engine adapters.
 *
 * Pi's provider composer routes a provider-level `streamSimple` solely to
 * models whose engine matches the provider-level `api`
 * (`model.api === extension.api`), and a provider declares exactly one
 * `api`. One wrapper therefore can never stamp all four Zen backends
 * (completions, responses, Anthropic, Google) — the non-completions models
 * depended solely on the ambient `before_provider_headers` hook, which
 * never fires inside foreground (`async:false`) subagent children. Those
 * children leaked `Authorization: Bearer none` and got Zen `403
 * FreeTierError`.
 *
 * Instead this module registers one custom engine adapter per Zen backend
 * (`zen-*`, each stamping then delegating to its native Pi engine). Models
 * point at the `zen-*` ids, so every request — parent, background runner,
 * foreground child — travels through the stamp on every backend, with no
 * ambient hooks required. Registration is idempotent (the compat registry
 * overwrites silently) and fails loud when the engine registry is broken.
 */

/** Native Pi engines behind the Zen gateway (exactly the stock `opencode` routing). */
export type NativeZenApi =
  | "openai-completions"
  | "openai-responses"
  | "anthropic-messages"
  | "google-generative-ai";

/** Custom engine ids: same dispatch as native, plus the Zen identity stamp. */
export type ZenApi = `zen-${NativeZenApi}`;

export const ZEN_COMPLETIONS_API: ZenApi = "zen-openai-completions";
export const ZEN_RESPONSES_API: ZenApi = "zen-openai-responses";
export const ZEN_ANTHROPIC_API: ZenApi = "zen-anthropic-messages";
export const ZEN_GOOGLE_API: ZenApi = "zen-google-generative-ai";

const NATIVE_TO_ZEN: Record<NativeZenApi, ZenApi> = {
  "openai-completions": ZEN_COMPLETIONS_API,
  "openai-responses": ZEN_RESPONSES_API,
  "anthropic-messages": ZEN_ANTHROPIC_API,
  "google-generative-ai": ZEN_GOOGLE_API,
};

const ZEN_TO_NATIVE: Record<ZenApi, NativeZenApi> = {
  [ZEN_COMPLETIONS_API]: "openai-completions",
  [ZEN_RESPONSES_API]: "openai-responses",
  [ZEN_ANTHROPIC_API]: "anthropic-messages",
  [ZEN_GOOGLE_API]: "google-generative-ai",
};

function hasOwn(obj: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

export function isNativeZenApi(api: unknown): api is NativeZenApi {
  return typeof api === "string" && hasOwn(NATIVE_TO_ZEN, api);
}

export function isZenApi(api: unknown): api is ZenApi {
  return typeof api === "string" && hasOwn(ZEN_TO_NATIVE, api);
}

/** Maps a native engine to its stamping Zen adapter; undefined for unknown values. */
export function zenApiForNative(api: string): ZenApi | undefined {
  if (!isNativeZenApi(api)) return undefined;
  return NATIVE_TO_ZEN[api];
}

/** Maps a Zen adapter id back to its native engine; undefined for anything else. */
export function nativeApiForZen(api: unknown): NativeZenApi | undefined {
  if (!isZenApi(api)) return undefined;
  return ZEN_TO_NATIVE[api];
}

export type HeaderTransform = (
  headers: ProviderHeaders,
) => ProviderHeaders | Promise<ProviderHeaders>;

/**
 * Composes Pi core's per-request `transformHeaders` (attribution + hook
 * wherever hooks run) with the Zen identity stamp, so every process —
 * parent, background runner, foreground child — emits identical traffic.
 */
export function composeZenTransformHeaders(
  parent?: HeaderTransform | undefined,
): HeaderTransform {
  return async (headers: ProviderHeaders) => {
    const base = parent ? await parent(headers) : headers;
    const stamped: ProviderHeaders = { ...(base ?? {}) };
    applyOpenCodeFreeHeaders(stamped);
    return stamped;
  };
}

/**
 * `transformHeaders` is threaded through request options by Pi core's
 * streamFn (sdk.ts) but is absent from pi-ai's public option types, hence
 * the structural read/cast here (same pattern as createZenStreamSimple).
 *
 * Stamping happens on TWO carriers, deliberately:
 *
 * 1. `options.headers` — the only field pi-ai engines actually send.
 *    Pi core's `prepareRequest` consumes `transformHeaders` *before* the
 *    engine is reached, so a stamp living there alone never reaches the
 *    wire on request paths without the ambient hook (foreground
 *    `async:false` children) — that is the leak this module exists to stop.
 * 2. `transformHeaders` — kept for any downstream layer that does apply it,
 *    composed with the caller's transform so nothing is dropped.
 *
 * The Zen adapters are only ever reached by `zen-*` models, which only this
 * extension creates, so the identity gate is forced here instead of relying
 * on provider config headers having survived every merge on the way.
 */
function withZenHeaders<O>(options: O): O {
  const raw = (options ?? {}) as Record<string, unknown>;
  const incoming = raw.headers as ProviderHeaders | undefined;
  const headers: ProviderHeaders = { ...(incoming ?? {}) };
  headers["x-opencode-client"] = "cli";
  if (getHeader(headers as MutableHeaders, "x-opencode-project") === undefined) {
    headers["x-opencode-project"] = "global";
  }
  applyOpenCodeFreeHeaders(headers as MutableHeaders);
  const parent = raw.transformHeaders as HeaderTransform | undefined;
  return {
    ...raw,
    headers,
    transformHeaders: composeZenTransformHeaders(parent),
  } as O;
}

export type EngineResolver = (
  api: Api,
) => Pick<
  NonNullable<ReturnType<typeof getApiProvider>>,
  "stream" | "streamSimple"
> | undefined;

/**
 * Builds the `{ stream, streamSimple }` pair for one Zen backend: stamp the
 * request headers, then delegate straight to the native engine adapter —
 * never to compat's top-level `stream`/`streamSimple` (that would resolve
 * the `zen-*` id back to this same wrapper and recurse).
 *
 * The delegated call carries a shallow model copy with the native `api`:
 * compat's registry wraps every engine in an api guard
 * (`model.api === api`), so handing it a `zen-*` model throws
 * `Mismatched api: zen-openai-completions expected openai-completions` on
 * the first token. The caller's model object is never mutated, and engines
 * that stamp `api` onto assistant messages then record the native id — a
 * value pi resolves without this extension loaded.
 *
 * `resolveEngine` is injectable for tests; production always uses the real
 * engine registry.
 */
export function createZenEngineFunctions(
  nativeApi: NativeZenApi,
  resolveEngine: EngineResolver = (api) => getApiProvider(api),
): Pick<ApiProvider, "stream" | "streamSimple"> {
  const resolve = () => {
    const engine = resolveEngine(nativeApi);
    if (!engine) {
      throw new Error(`No API provider registered for api: ${nativeApi}`);
    }
    return engine;
  };
  const stream: ApiStreamFunction = (model, context, options) =>
    resolve().stream(
      { ...model, api: nativeApi } as Model<Api>,
      context,
      withZenHeaders(options),
    );
  const streamSimple: ApiStreamSimpleFunction = (model, context, options) =>
    resolve().streamSimple(
      { ...model, api: nativeApi } as Model<Api>,
      context,
      withZenHeaders(options),
    );
  return { stream, streamSimple };
}

const ZEN_SOURCE_ID = "pi-opencode-free";

/**
 * Registers the four `zen-*` engine adapters. Idempotent: re-registering
 * overwrites with an equivalent wrapper. The `register` parameter is
 * injectable so tests can capture registrations without touching the
 * global engine registry.
 */
export function registerZenApiProviders(
  register: typeof registerApiProvider = registerApiProvider,
): ZenApi[] {
  const ids: ZenApi[] = [];
  for (const native of Object.keys(NATIVE_TO_ZEN) as NativeZenApi[]) {
    const zen = NATIVE_TO_ZEN[native];
    const registration: ApiProvider = {
      api: zen,
      ...createZenEngineFunctions(native),
    };
    register(registration, ZEN_SOURCE_ID);
    ids.push(zen);
  }
  return ids;
}
