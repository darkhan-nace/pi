/**
 * Nace classifier provider - Drex decision models for `models.classify()` and codemode.
 *
 * Registers the `nace` provider with Drex's classifier models. Drex answers typed questions about
 * JSON state (`bool`, `choice`, `score`) with probabilities and reports `usage.input_tokens`.
 * Requests go straight to Nace's System One endpoint and carry `X-Nace-Client: pi-nace/<version>`
 * so Nace can attribute them to Pi.
 *
 * This file imports only `@earendil-works/pi-ai` and `@earendil-works/pi-coding-agent`, so it also
 * loads in Pi releases that predate the built-in `nace` provider. Pi releases with that provider do
 * not need it.
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
 * Images: Drex Max models read PNG, JPEG, and WebP images passed in `images`, in order, before the
 * state. Drex accepts eight per request and answers a wrong format or count with a 422.
 * `drex-v2.0` is text only.
 *
 * Limits against the built-in provider: this extension does not retry, and it does not call the
 * `onPayload` and `onResponse` hooks.
 */

import {
	type ApiKeyAuth,
	type ClassifierAnswer,
	type ClassifierContext,
	type ClassifierModel,
	type ClassifierOptions,
	type ClassifierResult,
	calculateCost,
	createProvider,
	envApiKeyAuth,
	type RefreshModelsContext,
	type Usage,
} from "@earendil-works/pi-ai";
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

const CLASSIFIER_MODELS: ClassifierModel<"nace-system-one">[] = [
	...MAX_MODELS.map(
		({ id, name }): ClassifierModel<"nace-system-one"> => ({
			type: "classifier",
			id,
			name,
			api: "nace-system-one",
			provider: "nace",
			baseUrl: `${BASE_URL}/`,
			input: ["text", "image"],
			cost: COST,
			contextWindow: 262144,
		}),
	),
	{
		type: "classifier",
		id: "drex-v2.0",
		name: "Drex 2.0",
		api: "nace-system-one",
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

type WireAnswer = Record<string, unknown>;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function number(value: unknown, field: string): number {
	if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Nace returned an invalid ${field}`);
	return value;
}

/** Maps one wire answer to Pi's public answer type; `bool` is `noul` on the wire. */
function toAnswer(id: string, type: string, answer: WireAnswer): ClassifierAnswer {
	if (type === "choice") {
		if (answer.type !== "choice" || typeof answer.choice !== "string" || !isRecord(answer.probabilities)) {
			throw new Error(`Nace did not return a choice answer for ${id}`);
		}
		return {
			type: "choice",
			choice: answer.choice,
			probabilities: Object.fromEntries(
				Object.entries(answer.probabilities).map(([label, p]) => [
					label,
					number(p, `probability for ${id}.${label}`),
				]),
			),
			confidence: number(answer.confidence, `confidence for ${id}`),
		};
	}
	if (type === "score") {
		if (answer.type !== "score") throw new Error(`Nace did not return a score answer for ${id}`);
		return {
			type: "score",
			score: number(answer.score, `score for ${id}`),
			confidence: number(answer.confidence, `confidence for ${id}`),
		};
	}
	if (answer.type !== "noul") throw new Error(`Nace did not return a bool answer for ${id}`);
	return { type: "bool", probability: number(answer.noul, `probability for ${id}`) };
}

function toUsage(model: ClassifierModel<"nace-system-one">, value: unknown): Usage | undefined {
	if (!isRecord(value) || value.input_tokens === undefined) return undefined;
	const input = typeof value.input_tokens === "number" ? value.input_tokens : 0;
	const output = typeof value.output_tokens === "number" ? value.output_tokens : 0;
	const usage: Usage = {
		input,
		output,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	calculateCost(model, usage);
	return usage;
}

async function classify(
	model: ClassifierModel<"nace-system-one">,
	context: ClassifierContext,
	options: ClassifierOptions | undefined,
): Promise<ClassifierResult> {
	const result: ClassifierResult = {
		api: model.api,
		provider: model.provider,
		model: model.id,
		answers: {},
		stopReason: "stop",
		timestamp: Date.now(),
	};
	try {
		if (!options?.apiKey) throw new Error(`No API key for provider: ${model.provider}`);
		const body = {
			model: model.id,
			state: context.state,
			questions: Object.fromEntries(
				Object.entries(context.questions).map(([id, question]) => [
					id,
					question.type === "bool" ? { ...question, type: "noul" } : question,
				]),
			),
			...(context.images?.length
				? { images: context.images.map((image) => `data:${image.mimeType};base64,${image.data}`) }
				: {}),
		};
		const timeout = options.timeoutMs === undefined ? undefined : AbortSignal.timeout(options.timeoutMs);
		const signal =
			options.signal && timeout ? AbortSignal.any([options.signal, timeout]) : (options.signal ?? timeout);
		const response = await (options.fetch ?? fetch)(new URL("systemone", model.baseUrl), {
			method: "POST",
			headers: {
				...Object.fromEntries(Object.entries(options.headers ?? {}).filter(([, value]) => value !== null)),
				authorization: `Bearer ${options.apiKey}`,
				"content-type": "application/json",
			},
			body: JSON.stringify(body),
			signal,
		});
		const text = await response.text();
		if (!response.ok) throw new Error(`Nace API error (${response.status}): ${text}`);
		const wire: unknown = JSON.parse(text);
		if (!isRecord(wire) || !isRecord(wire.answers)) throw new Error("Nace returned an unexpected response");
		// Set before parsing answers: a request with malformed answers was still billed.
		const usage = toUsage(model, wire.usage);
		if (usage) result.usage = usage;
		for (const [id, question] of Object.entries(context.questions)) {
			const answer = wire.answers[id];
			if (!isRecord(answer)) throw new Error(`Nace did not return an answer for ${id}`);
			result.answers[id] = toAnswer(id, question.type, answer);
		}
		return result;
	} catch (error) {
		result.stopReason = options?.signal?.aborted ? "aborted" : "error";
		result.errorMessage = error instanceof Error ? error.message : String(error);
		return result;
	}
}

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
	if (!isRecord(body) || !Array.isArray(body.models)) throw new Error("Invalid Nace model list response");
	const names = (body.models as unknown[]).flatMap((entry) =>
		isRecord(entry) && typeof entry.name === "string" ? [entry.name] : [],
	);
	return new Set(names);
}

export default function (pi: ExtensionAPI) {
	let available: ReadonlySet<string> | undefined;
	pi.registerProvider(
		createProvider<"nace-system-one">({
			id: "nace",
			name: "Nace",
			baseUrl: BASE_URL,
			auth: { apiKey: auth },
			models: CLASSIFIER_MODELS,
			classifiers: {
				"nace-system-one": {
					classify: (model, context, options) =>
						classify(model as ClassifierModel<"nace-system-one">, context, options),
				},
			},
			fetchModels: async (context) => {
				available = await listAvailable(context);
				return [];
			},
			filterAllModels: (models) => (available ? models.filter((model) => available?.has(model.id)) : models),
		}),
	);
}
