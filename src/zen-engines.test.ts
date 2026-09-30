// SPDX-License-Identifier: GPL-3.0-or-later
import assert from "node:assert/strict";
import test from "node:test";
import {
  composeZenTransformHeaders,
  createZenEngineFunctions,
  isNativeZenApi,
  isZenApi,
  nativeApiForZen,
  registerZenApiProviders,
  zenApiForNative,
  ZEN_ANTHROPIC_API,
  ZEN_COMPLETIONS_API,
  ZEN_GOOGLE_API,
  ZEN_RESPONSES_API,
} from "./zen-engines.js";

test("native/zen api mapping round-trips on all four backends", async () => {
  assert.equal(zenApiForNative("openai-completions"), ZEN_COMPLETIONS_API);
  assert.equal(zenApiForNative("openai-responses"), ZEN_RESPONSES_API);
  assert.equal(zenApiForNative("anthropic-messages"), ZEN_ANTHROPIC_API);
  assert.equal(zenApiForNative("google-generative-ai"), ZEN_GOOGLE_API);
  assert.equal(zenApiForNative("totally-made-up-engine"), undefined);
  assert.equal(nativeApiForZen(ZEN_COMPLETIONS_API), "openai-completions");
  assert.equal(nativeApiForZen(ZEN_RESPONSES_API), "openai-responses");
  assert.equal(nativeApiForZen(ZEN_ANTHROPIC_API), "anthropic-messages");
  assert.equal(nativeApiForZen(ZEN_GOOGLE_API), "google-generative-ai");
  // Native ids and garbage never map back (migration stays idempotent).
  assert.equal(nativeApiForZen("openai-completions"), undefined);
  assert.equal(nativeApiForZen("totally-made-up-engine"), undefined);
  assert.equal(nativeApiForZen(undefined), undefined);
  assert.equal(nativeApiForZen(null), undefined);
  assert.equal(isNativeZenApi("anthropic-messages"), true);
  assert.equal(isNativeZenApi(ZEN_ANTHROPIC_API), false);
  assert.equal(isZenApi(ZEN_ANTHROPIC_API), true);
  assert.equal(isZenApi("anthropic-messages"), false);
});

test("registerZenApiProviders registers all four stamping adapters", async () => {
  const registrations: { api: unknown; sourceId?: string }[] = [];
  const ids = registerZenApiProviders(((registration: any, sourceId?: string) => {
    registrations.push({ ...registration, sourceId });
  }) as any);
  assert.deepEqual(ids, [
    ZEN_COMPLETIONS_API,
    ZEN_RESPONSES_API,
    ZEN_ANTHROPIC_API,
    ZEN_GOOGLE_API,
  ]);
  assert.equal(registrations.length, 4);
  for (const reg of registrations) {
    assert.ok(
      (ids as string[]).includes(reg.api as string),
      `registered api is a zen id: ${String(reg.api)}`,
    );
    assert.equal(reg.sourceId, "pi-opencode-free");
    assert.equal(typeof (reg as any).stream, "function");
    assert.equal(typeof (reg as any).streamSimple, "function");
  }
});

test("zen engine adapters stamp and delegate on both stream and streamSimple", async () => {
  const calls: { entry: string; options: any }[] = [];
  const canned = { marker: "native-engine-stream" };
  const resolveEngine = (_api: any) => ({
    stream: ((_model: any, _context: any, options: any) => {
      calls.push({ entry: "stream", options });
      return canned as any;
    }) as any,
    streamSimple: ((_model: any, _context: any, options: any) => {
      calls.push({ entry: "streamSimple", options });
      return canned as any;
    }) as any,
  });
  const functions = createZenEngineFunctions("anthropic-messages", resolveEngine);
  const model = { api: ZEN_ANTHROPIC_API, provider: "opencode-free", id: "x" };
  let parentTransformCalls = 0;
  const options = {
    temperature: 0.5,
    transformHeaders: async (headers: Record<string, string | null>) => {
      parentTransformCalls++;
      return { "x-opencode-client": "pi", ...headers };
    },
  };
  assert.equal((functions.stream as any)(model, [], options), canned);
  assert.equal((functions.streamSimple as any)(model, [], options), canned);
  assert.equal(calls.length, 2);
  for (const { options: engineOptions } of calls) {
    assert.equal(engineOptions.temperature, 0.5); // unrelated options pass through
    const stamped = await engineOptions.transformHeaders({
      "x-opencode-client": "cli",
      "x-opencode-project": "global",
      Authorization: "Bearer none",
    });
    assert.equal(stamped["x-opencode-client"], "cli"); // stamp wins over attribution
    assert.match(
      stamped["x-opencode-session"],
      /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/,
    );
    assert.match(stamped["x-opencode-request"], /^msg_[0-9a-f-]{36}$/);
    assert.equal(stamped.Authorization, null); // placeholder never leaks
  }
  assert.equal(parentTransformCalls, 2); // core transform runs inside the stamp
});

test("zen adapters stamp the header object engines actually send", () => {
  // Regression: pi core consumes `transformHeaders` in prepareRequest, so a
  // stamp living only there never reaches the engine — the adapter must
  // stamp `options.headers` itself, with no ambient hook and no parent
  // transform (the foreground-child path).
  let captured: any = null;
  const functions = createZenEngineFunctions("openai-completions", () => ({
    stream: undefined as any,
    streamSimple: (_m: any, _c: any, o: any) => {
      captured = o;
      return {} as any;
    },
  }));
  const model = { api: ZEN_COMPLETIONS_API, provider: "opencode-free", id: "x" };

  // Exactly what prepareRequest hands down for our traffic.
  (functions.streamSimple as any)(
    model,
    [],
    {
      headers: {
        Authorization: "Bearer none",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
      },
    },
  );
  assert.ok(captured, "engine received options");
  const headers = captured.headers as Record<string, string | null>;
  assert.equal(headers["x-opencode-client"], "cli");
  assert.equal(headers["x-opencode-project"], "global");
  assert.match(
    headers["x-opencode-session"] as string,
    /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/,
  );
  assert.match(headers["x-opencode-request"] as string, /^msg_[0-9a-f-]{36}$/);
  assert.equal(headers["User-Agent"], "opencode/1.18.31");
  assert.equal(headers["Authorization"], null); // never Bearer none
  assert.ok(captured.transformHeaders); // composed transform kept too

  // Identity is stamped even when the provider config headers never arrive.
  captured = null;
  (functions.streamSimple as any)(model, [], undefined);
  assert.equal(captured.headers["x-opencode-client"], "cli");
  assert.equal(captured.headers["x-opencode-project"], "global");
  assert.equal(captured.headers["Authorization"], null);
});

test("zen adapters hand the native engine the api it was registered under", () => {
  // Regression: compat wraps every registered engine in
  //   (model) => model.api === api ? ... : throw `Mismatched api`
  // so delegating the `zen-*` model verbatim threw
  // "Mismatched api: zen-openai-completions expected openai-completions"
  // before any request was made.
  const seen: any[] = [];
  const functions = createZenEngineFunctions("openai-completions", () => ({
    stream: undefined as any,
    streamSimple: (m: any, _c: any, o: any) => {
      seen.push({ model: m, options: o });
      return "delegated" as any;
    },
  }));
  const model = { api: ZEN_COMPLETIONS_API, provider: "opencode-free", id: "x" };
  assert.equal((functions.streamSimple as any)(model, [], undefined), "delegated");
  assert.equal(seen.length, 1);
  assert.equal(seen[0].model.api, "openai-completions"); // passes compat's guard
  assert.equal(seen[0].model.provider, "opencode-free");
  assert.equal(seen[0].model.id, "x");
  assert.equal(model.api, ZEN_COMPLETIONS_API); // caller's model untouched
  assert.notEqual(seen[0].model, model);
});

test("zen engine adapters leave foreign traffic untouched", async () => {
  let captured: any = null;
  const functions = createZenEngineFunctions("google-generative-ai", () => ({
    stream: undefined as any,
    streamSimple: ((_model: any, _context: any, options: any) => {
      captured = options;
      return {} as any;
    }) as any,
  }));
  (functions.streamSimple as any)(
    { api: ZEN_GOOGLE_API, provider: "opencode-free", id: "x" },
    [],
    undefined, // no parent transform at all
  );
  const out = await captured.transformHeaders({
    Authorization: "Bearer real-user-key",
  });
  assert.equal(out.Authorization, "Bearer real-user-key");
  assert.equal(out["x-opencode-session"], undefined);
});

test("zen engine adapters fail loud on unknown native engine", async () => {
  const functions = createZenEngineFunctions("openai-responses", () => undefined);
  assert.throws(
    () =>
      (functions.streamSimple as any)(
        { api: ZEN_RESPONSES_API, provider: "opencode-free", id: "x" },
        [],
        undefined,
      ),
    /No API provider registered for api: openai-responses/,
  );
});

test("composeZenTransformHeaders without a parent still stamps", async () => {
  const transform = composeZenTransformHeaders(undefined);
  const out = await transform({
    "x-opencode-client": "cli",
    "x-opencode-project": "global",
  });
  assert.equal(out.Authorization, null);
  assert.match(
    (out as any)["x-opencode-session"],
    /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/,
  );
});
