import { typesafeSystemOneApi } from "../api/typesafe-system-one.lazy.ts";
import { envApiKeyAuth } from "../auth/helpers.ts";
import type { ApiKeyAuth } from "../auth/types.ts";
import { createProvider, type Provider, type RefreshModelsContext } from "../models.ts";
import type { AnyModel } from "../types.ts";
import { isModelType } from "../utils/model-operations.ts";
import { PI_AI_VERSION } from "../utils/version.ts";
import { NACE_CLASSIFIER_MODELS } from "./nace.models.ts";

const NACE_BASE_URL = "https://console.nace.ai/v1";
const NACE_MODELS_TIMEOUT_MS = 15_000;

/** Nace stores this as the request's `client_app` and maps `pi-nace` to its Pi traffic channel. */
const NACE_CLIENT_HEADERS = { "X-Nace-Client": `pi-nace/${PI_AI_VERSION}` };

const envAuth = envApiKeyAuth("Nace API key", ["NACE_API_KEY"]);

const naceAuth: ApiKeyAuth = {
	...envAuth,
	resolve: async (input) => {
		const resolved = await envAuth.resolve(input);
		return resolved && { ...resolved, auth: { ...resolved.auth, headers: NACE_CLIENT_HEADERS } };
	},
};

/**
 * Restrict the Nace catalog to the models the key may use, from `GET /models`. Drex Max is staged:
 * a key without access gets a 422 for it, so listing it would mislead. When the endpoint rejects the
 * key (401/403) or does not exist (404), for example behind a proxy configured in models.json, the
 * catalog is returned unchanged and `classify()` reports the provider's error.
 */
export async function personalizeNaceCatalog(
	catalog: readonly AnyModel[],
	context: RefreshModelsContext,
): Promise<readonly AnyModel[]> {
	const key = context.credential?.type === "api_key" ? context.credential.key : undefined;
	if (!key) return catalog;
	const response = await fetch(`${(context.baseUrl ?? NACE_BASE_URL).replace(/\/+$/, "")}/models`, {
		headers: { accept: "application/json", authorization: `Bearer ${key}`, ...NACE_CLIENT_HEADERS },
		signal: AbortSignal.any([context.signal, AbortSignal.timeout(NACE_MODELS_TIMEOUT_MS)]),
	});
	if (response.status === 401 || response.status === 403 || response.status === 404) return catalog;
	if (!response.ok) throw new Error(`Nace model list request failed: ${response.status}`);
	const body: unknown = await response.json();
	if (typeof body !== "object" || body === null || !("models" in body) || !Array.isArray(body.models)) {
		throw new Error("Invalid Nace model list response");
	}
	const available = new Set<string>();
	for (const entry of body.models as unknown[]) {
		if (typeof entry === "object" && entry !== null && "name" in entry && typeof entry.name === "string") {
			available.add(entry.name);
		}
	}
	return catalog.filter((model) => !isModelType(model, "classifier") || available.has(model.id));
}

export function naceProvider(): Provider<"typesafe-system-one"> {
	return createProvider<"typesafe-system-one">({
		id: "nace",
		name: "Nace",
		baseUrl: NACE_BASE_URL,
		auth: { apiKey: naceAuth },
		models: Object.values(NACE_CLASSIFIER_MODELS),
		classifiers: { "typesafe-system-one": typesafeSystemOneApi() },
	});
}
