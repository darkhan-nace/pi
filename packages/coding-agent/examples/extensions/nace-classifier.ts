/**
 * Nace classifier provider - Drex decision models for `models.classify()` and codemode.
 *
 * Registers the `nace` provider with Drex's classifier models. Drex answers typed questions about
 * JSON state (`bool`, `choice`, `score`) with probabilities and reports `usage.input_tokens`.
 * It speaks the System One protocol, so the provider reuses Pi's `typesafe-system-one` API and sends
 * `X-Nace-Client: pi-nace/<version>` so Nace can attribute the requests to Pi.
 *
 * Setup: export NACE_API_KEY (or log in with `/login`), then enable codemode in settings with
 * `"defaultTools": ["+codemode"]` and let a script call a Drex model:
 *
 *   const drex = await models.getModelOfType("classifier", "nace", "drex-latest");
 *   const result = await models.classify(drex, {
 *     state: { ticket: "I was charged twice for March." },
 *     questions: { refund: { type: "bool", instructions: "Does the customer want a refund?",
 *       criteria: { true: "Asks for a refund", false: "Does not" } } },
 *   });
 *   return { answers: result.answers, usage: result.usage };
 *
 * Extensions call `ctx.modelRegistry.classify()` the same way, see jev-router.ts.
 *
 * Usage: pi -e ./nace-classifier.ts
 *
 * Cost: `result.usage.cost` is an estimate from the catalog price below ($0.03 per million input
 * tokens, from Drex's price list). Nace bills `usage.input_tokens`; its invoice is authoritative.
 *
 * Images: Drex Max models read PNG, JPEG, and WebP images passed in `images`, at most four per
 * request from Pi (Drex accepts eight). `drex-v2.0` is text only.
 */

import {
	type ApiKeyAuth,
	type ClassifierModel,
	createProvider,
	envApiKeyAuth,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { typesafeSystemOneApi } from "@earendil-works/pi-ai/api/typesafe-system-one.lazy";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const ADAPTER_VERSION = "0.1.0";
const BASE_URL = "https://console.nace.ai/v1";
const MODELS_TIMEOUT_MS = 15_000;
const CLIENT_HEADERS = { "X-Nace-Client": `pi-nace/${ADAPTER_VERSION}` };

const COST = { input: 0.03, output: 0, cacheRead: 0, cacheWrite: 0 };

// `drex-latest` points to `drex-max-v1.1` today and moves only on a date Nace announces.
const MAX_MODELS = [
	{ id: "drex-latest", name: "Drex Latest" },
	{ id: "drex-max-v1.1", name: "Drex Max v1.1" },
	{ id: "drex-max-finance", name: "Drex Max Finance" },
];

const CLASSIFIER_MODELS: ClassifierModel<"typesafe-system-one">[] = [
	...MAX_MODELS.map(
		({ id, name }): ClassifierModel<"typesafe-system-one"> => ({
			type: "classifier",
			id,
			name,
			api: "typesafe-system-one",
			provider: "nace",
			baseUrl: `${BASE_URL}/`,
			input: ["text", "image"],
			cost: COST,
			contextWindow: 262144,
			compat: { imageInput: "images", sendImageOptInHeader: false },
		}),
	),
	{
		type: "classifier",
		id: "drex-v2.0",
		name: "Drex 2.0",
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: `${BASE_URL}/`,
		input: ["text"],
		cost: COST,
		contextWindow: 131072,
	},
];

const envAuth = envApiKeyAuth("Nace API key", ["NACE_API_KEY"]);

const auth: ApiKeyAuth = {
	...envAuth,
	resolve: async (input) => {
		const resolved = await envAuth.resolve(input);
		return resolved && { ...resolved, auth: { ...resolved.auth, headers: CLIENT_HEADERS } };
	},
};

/**
 * The model ids the key may use, from `GET /models`, or undefined when the endpoint does not
 * answer for the key (401, 403, 404). Drex Max is staged, so a key without access gets a 422 for it.
 */
async function listAvailable(context: RefreshModelsContext): Promise<ReadonlySet<string> | undefined> {
	const key = context.credential?.type === "api_key" ? context.credential.key : undefined;
	if (!key) return undefined;
	const response = await fetch(`${(context.baseUrl ?? BASE_URL).replace(/\/+$/, "")}/models`, {
		headers: { accept: "application/json", authorization: `Bearer ${key}`, ...CLIENT_HEADERS },
		signal: AbortSignal.any([context.signal, AbortSignal.timeout(MODELS_TIMEOUT_MS)]),
	});
	if (response.status === 401 || response.status === 403 || response.status === 404) return undefined;
	if (!response.ok) throw new Error(`Nace model list request failed: ${response.status}`);
	const body: unknown = await response.json();
	if (typeof body !== "object" || body === null || !("models" in body) || !Array.isArray(body.models)) {
		throw new Error("Invalid Nace model list response");
	}
	const names = (body.models as unknown[]).flatMap((entry) =>
		typeof entry === "object" && entry !== null && "name" in entry && typeof entry.name === "string"
			? [entry.name]
			: [],
	);
	return new Set(names);
}

export default function (pi: ExtensionAPI) {
	let available: ReadonlySet<string> | undefined;
	pi.registerProvider(
		createProvider<"typesafe-system-one">({
			id: "nace",
			name: "Nace",
			baseUrl: BASE_URL,
			auth: { apiKey: auth },
			models: CLASSIFIER_MODELS,
			classifiers: { "typesafe-system-one": typesafeSystemOneApi() },
			fetchModels: async (context) => {
				available = await listAvailable(context);
				return [];
			},
			filterAllModels: (models) => (available ? models.filter((model) => available?.has(model.id)) : models),
		}),
	);
}
