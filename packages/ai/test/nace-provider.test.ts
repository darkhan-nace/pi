import { afterEach, describe, expect, it, vi } from "vitest";
import { InMemoryCredentialStore } from "../src/auth/credential-store.ts";
import type { AuthContext } from "../src/auth/types.ts";
import type { RefreshModelsContext } from "../src/models.ts";
import { builtinModels, getBuiltinClassifierModel, getBuiltinClassifierModels } from "../src/providers/all.ts";
import { personalizeNaceCatalog } from "../src/providers/nace.ts";
import type { ClassifierContext, ClassifierModel } from "../src/types.ts";
import { PI_AI_VERSION } from "../src/utils/version.ts";

const context: ClassifierContext = {
	state: { ticket: "I was charged twice for March. Please refund one charge." },
	questions: {
		category: {
			type: "choice",
			instructions: "What is this ticket about?",
			criteria: { billing: "Payments and refunds", technical: "Bugs and outages", other: "Anything else" },
		},
		upset: {
			type: "score",
			instructions: "How upset is the customer?",
			criteria: ["Calm", "Annoyed", "Angry"],
		},
		wants_refund: {
			type: "bool",
			instructions: "Is the customer asking for a refund?",
			criteria: { true: "Asks for money back", false: "Does not ask for money back" },
		},
	},
};

const wireResponse = {
	model: "drex-max-v1.1",
	answers: {
		category: {
			type: "choice",
			choice: "billing",
			probabilities: { billing: 0.9895, technical: 0.005, other: 0.0055 },
			confidence: 0.986,
		},
		upset: {
			type: "score",
			score: 1.2,
			legend: ["Calm", "Annoyed", "Angry"],
			probabilities: { "0": 0.2, "1": 0.6, "2": 0.2 },
			confidence: 0.7183,
		},
		wants_refund: { type: "noul", noul: 0.9868, confidence: 0.9868 },
	},
	usage: { input_tokens: 106, output_tokens: 212 },
	evaluation_time_ms: 142.6,
	request_id: "req_0123456789abcdef0123456789abcdef",
};

function authContext(env: Record<string, string>): AuthContext {
	return { env: async (name) => env[name], fileExists: async () => false };
}

async function naceModels() {
	const credentials = new InMemoryCredentialStore();
	await credentials.modify("nace", async () => ({ type: "api_key", key: "nace_sk_test" }));
	return builtinModels({ credentials });
}

type NaceModelId = "drex-latest" | "drex-max-v1.1" | "drex-max-finance" | "drex-v2.0";

function model(id: NaceModelId): ClassifierModel<"typesafe-system-one"> {
	return getBuiltinClassifierModel("nace", id) as ClassifierModel<"typesafe-system-one">;
}

describe("Nace classifier provider", () => {
	it("lists the Drex models as classifier models only, priced from the Drex price list", () => {
		expect(getBuiltinClassifierModels("nace").map((entry) => entry.id)).toEqual([
			"drex-latest",
			"drex-max-finance",
			"drex-max-v1.1",
			"drex-v2.0",
		]);
		for (const entry of getBuiltinClassifierModels("nace")) {
			expect(entry).toMatchObject({
				type: "classifier",
				api: "typesafe-system-one",
				provider: "nace",
				baseUrl: "https://console.nace.ai/v1/",
				// $0.03 per million input tokens, nothing for output tokens.
				cost: { input: 0.03, output: 0, cacheRead: 0, cacheWrite: 0 },
			});
		}
		expect(model("drex-v2.0")).toMatchObject({ input: ["text"], contextWindow: 131072 });
		expect(model("drex-max-v1.1")).toMatchObject({
			input: ["text", "image"],
			contextWindow: 262144,
			compat: { imageInput: "images", sendImageOptInHeader: false },
		});
		expect(builtinModels().getModel("nace", "drex-v2.0")).toBeUndefined();
	});

	it("returns typed answers with probabilities, question ids and priced usage, and attributes the request to pi-nace", async () => {
		const models = await naceModels();
		const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
		const result = await models.classify(model("drex-latest"), context, {
			fetch: async (input, init) => {
				requests.push({
					url: String(input),
					headers: new Headers(init?.headers),
					body: JSON.parse(String(init?.body)) as Record<string, unknown>,
				});
				return Response.json(wireResponse);
			},
		});

		expect(requests).toHaveLength(1);
		expect(requests[0].url).toBe("https://console.nace.ai/v1/systemone");
		expect(requests[0].headers.get("authorization")).toBe("Bearer nace_sk_test");
		expect(requests[0].headers.get("x-nace-client")).toBe(`pi-nace/${PI_AI_VERSION}`);
		expect(requests[0].body).toMatchObject({
			model: "drex-latest",
			state: context.state,
			questions: { wants_refund: { type: "noul" }, category: { type: "choice" }, upset: { type: "score" } },
		});

		expect(result.stopReason).toBe("stop");
		expect(Object.keys(result.answers).sort()).toEqual(["category", "upset", "wants_refund"]);
		expect(result.answers.wants_refund).toEqual({ type: "bool", probability: 0.9868 });
		expect(result.answers.category).toEqual({
			type: "choice",
			choice: "billing",
			probabilities: { billing: 0.9895, technical: 0.005, other: 0.0055 },
			confidence: 0.986,
		});
		expect(result.answers.upset).toEqual({ type: "score", score: 1.2, confidence: 0.7183 });
		expect(result.usage).toMatchObject({ input: 106, output: 212 });
		expect(result.usage?.cost.input).toBeCloseTo(106 * 0.03e-6, 12);
	});

	it("attributes the request to pi-nace when the key comes from NACE_API_KEY", async () => {
		const models = builtinModels({ authContext: authContext({ NACE_API_KEY: "nace_sk_env" }) });
		const headers: Headers[] = [];
		const result = await models.classify(model("drex-v2.0"), context, {
			fetch: async (_input, init) => {
				headers.push(new Headers(init?.headers));
				return Response.json(wireResponse);
			},
		});
		expect(result.stopReason).toBe("stop");
		expect(headers[0].get("authorization")).toBe("Bearer nace_sk_env");
		expect(headers[0].get("x-nace-client")).toBe(`pi-nace/${PI_AI_VERSION}`);
	});

	it("sends images as an ordered list of data URLs to Drex Max and refuses them for Drex 2.0", async () => {
		const models = await naceModels();
		const bodies: Array<Record<string, unknown>> = [];
		const fetch = async (_input: string | URL | Request, init?: RequestInit) => {
			bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
			return Response.json(wireResponse);
		};
		const images = [
			{ type: "image" as const, data: "cG5n", mimeType: "image/png" },
			{ type: "image" as const, data: "d2VicA==", mimeType: "image/webp" },
		];

		const max = await models.classify(model("drex-max-v1.1"), { ...context, images }, { fetch });
		expect(max.stopReason).toBe("stop");
		expect(bodies[0].images).toEqual(["data:image/png;base64,cG5n", "data:image/webp;base64,d2VicA=="]);
		expect(bodies[0]).not.toHaveProperty("assets");

		const text = await models.classify(model("drex-v2.0"), { ...context, images }, { fetch });
		expect(text.stopReason).toBe("error");
		expect(text.errorMessage).toBe("Model nace/drex-v2.0 does not accept image input");
		expect(bodies).toHaveLength(1);
	});

	it("surfaces Nace's error type, message and request id without falling back to another provider", async () => {
		const models = await naceModels();
		const failure = (status: number, type: string, message: string) => async () =>
			Response.json({ error: { type, message }, request_id: "req_failed" }, { status });

		const unauthorized = await models.classify(model("drex-v2.0"), context, {
			fetch: failure(401, "authentication_error", "Invalid API key"),
			maxRetries: 0,
		});
		expect(unauthorized.stopReason).toBe("error");
		expect(unauthorized.provider).toBe("nace");
		expect(unauthorized.errorMessage).toContain("401");
		expect(unauthorized.errorMessage).toContain("authentication_error");
		expect(unauthorized.errorMessage).toContain("Invalid API key");
		expect(unauthorized.errorMessage).toContain("req_failed");

		const invalid = await models.classify(model("drex-latest"), context, {
			fetch: failure(422, "invalid_request_error", 'model: model "drex-max-v1.1" is not available yet'),
			maxRetries: 0,
		});
		expect(invalid.stopReason).toBe("error");
		expect(invalid.errorMessage).toContain("422");
		expect(invalid.errorMessage).toContain("is not available yet");
		expect(invalid.answers).toEqual({});
	});

	it("reports a missing key as an error result instead of calling the API", async () => {
		const fetch = vi.fn();
		const result = await builtinModels({ authContext: authContext({}) }).classify(model("drex-v2.0"), context, {
			fetch,
		});
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe("Provider is not configured: nace");
		expect(fetch).not.toHaveBeenCalled();
	});
});

describe("personalizeNaceCatalog", () => {
	afterEach(() => vi.restoreAllMocks());

	function refreshContext(credential: RefreshModelsContext["credential"]): RefreshModelsContext {
		return { credential, publish: async () => true, allowNetwork: true, signal: new AbortController().signal };
	}

	const catalog = getBuiltinClassifierModels("nace");

	it("keeps the models the key may use, with the key and the pi-nace header", async () => {
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
			Response.json({
				models: [
					{ name: "drex-v2.0", alias_for: null },
					{ name: "drex-v1.5", alias_for: "drex-max-v1.1" },
				],
				request_id: "req_models",
			}),
		);
		const result = await personalizeNaceCatalog(catalog, refreshContext({ type: "api_key", key: "nace_sk_test" }));

		expect(result.map((entry) => entry.id)).toEqual(["drex-v2.0"]);
		const [url, init] = fetchSpy.mock.calls[0];
		expect(String(url)).toBe("https://console.nace.ai/v1/models");
		const headers = new Headers(init?.headers);
		expect(headers.get("authorization")).toBe("Bearer nace_sk_test");
		expect(headers.get("x-nace-client")).toBe(`pi-nace/${PI_AI_VERSION}`);
	});

	it("returns the catalog unchanged without a key or when the key is rejected", async () => {
		const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 401 }));
		expect(await personalizeNaceCatalog(catalog, refreshContext(undefined))).toBe(catalog);
		expect(fetchSpy).not.toHaveBeenCalled();
		expect(await personalizeNaceCatalog(catalog, refreshContext({ type: "api_key", key: "bad" }))).toBe(catalog);
	});

	it("throws on a server error so the previous list stays", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 529 }));
		await expect(
			personalizeNaceCatalog(catalog, refreshContext({ type: "api_key", key: "nace_sk_test" })),
		).rejects.toThrow("Nace model list request failed: 529");
	});
});
