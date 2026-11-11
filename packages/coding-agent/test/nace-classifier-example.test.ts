import { afterEach, describe, expect, it, vi } from "vitest";
import naceClassifier from "../examples/extensions/nace-classifier.ts";
import { createHarness, type Harness } from "./suite/harness.ts";

const context = {
	state: { ticket: "I was charged twice." },
	questions: {
		category: {
			type: "choice" as const,
			instructions: "What is this ticket about?",
			criteria: { billing: "Payments and refunds", technical: "Bugs and outages" },
		},
	},
};

describe("nace-classifier example", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		for (const harness of harnesses.splice(0)) harness.cleanup();
	});

	async function setup() {
		vi.stubEnv("NACE_API_KEY", "");
		const harness = await createHarness({ extensionFactories: [naceClassifier] });
		harnesses.push(harness);
		const runtime = harness.session.modelRuntime;
		await runtime.refresh({ allowNetwork: false });
		await runtime.setRuntimeApiKey("nace", "nace_sk_test");
		return runtime;
	}

	it("registers the Drex classifier models, separate from chat models", async () => {
		const runtime = await setup();
		expect((await runtime.getAvailableOfType("classifier", "nace")).map((model) => model.id).sort()).toEqual([
			"drex-latest",
			"drex-max-finance",
			"drex-max-v1.1",
			"drex-v2.0",
		]);
		expect(runtime.getModel("nace", "drex-latest")).toBeUndefined();
		expect(runtime.getModelOfType("classifier", "nace", "drex-v2.0")).toMatchObject({
			input: ["text"],
			cost: { input: 0.03 },
		});
	});

	it("classifies with probabilities, usage and the pi-nace header", async () => {
		const runtime = await setup();
		const drex = runtime.getModelOfType("classifier", "nace", "drex-latest")!;
		const requests: Array<{ url: string; headers: Headers }> = [];
		const result = await runtime.classify(drex, context, {
			fetch: async (input, init) => {
				requests.push({ url: String(input), headers: new Headers(init?.headers) });
				return Response.json({
					answers: {
						category: {
							type: "choice",
							choice: "billing",
							probabilities: { billing: 0.98, technical: 0.02 },
							confidence: 0.98,
						},
					},
					usage: { input_tokens: 64, output_tokens: 9 },
					request_id: "req_test",
				});
			},
		});

		expect(requests[0]?.url).toBe("https://console.nace.ai/v1/systemone");
		expect(requests[0]?.headers.get("authorization")).toBe("Bearer nace_sk_test");
		expect(requests[0]?.headers.get("x-nace-client")).toBe("pi-nace/0.1.0");
		expect(result.answers.category).toEqual({
			type: "choice",
			choice: "billing",
			probabilities: { billing: 0.98, technical: 0.02 },
			confidence: 0.98,
		});
		expect(result.usage).toMatchObject({ input: 64, output: 9 });
	});

	it("posts to /systemone under a base URL without a trailing slash", async () => {
		const runtime = await setup();
		const drex = {
			...runtime.getModelOfType("classifier", "nace", "drex-latest")!,
			baseUrl: "http://127.0.0.1:9/v1",
		};
		const urls: string[] = [];
		await runtime.classify(drex, context, {
			fetch: async (input) => {
				urls.push(String(input));
				return Response.json({ answers: {} });
			},
		});
		expect(urls).toEqual(["http://127.0.0.1:9/v1/systemone"]);
	});

	it("maps bool, choice and score questions and sends images in order before the state", async () => {
		const runtime = await setup();
		const drex = runtime.getModelOfType("classifier", "nace", "drex-max-v1.1")!;
		const bodies: Array<Record<string, unknown>> = [];
		const result = await runtime.classify(
			drex,
			{
				state: { ticket: "I was charged twice." },
				images: [
					{ type: "image", data: "cG5n", mimeType: "image/png" },
					{ type: "image", data: "d2VicA==", mimeType: "image/webp" },
				],
				questions: {
					refund: { type: "bool", instructions: "Refund?", criteria: { true: "Yes", false: "No" } },
					category: context.questions.category,
					upset: { type: "score", instructions: "How upset?", criteria: ["Calm", "Angry"] },
				},
			},
			{
				fetch: async (_input, init) => {
					bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
					return Response.json({
						answers: {
							refund: { type: "noul", noul: 0.9, confidence: 0.9 },
							category: {
								type: "choice",
								choice: "billing",
								probabilities: { billing: 1, technical: 0 },
								confidence: 1,
							},
							upset: {
								type: "score",
								score: 0.4,
								legend: ["Calm", "Angry"],
								probabilities: { "0": 0.6, "1": 0.4 },
								confidence: 0.6,
							},
						},
						usage: { input_tokens: 10, output_tokens: 1 },
					});
				},
			},
		);

		expect(bodies[0]).toMatchObject({
			model: "drex-max-v1.1",
			images: ["data:image/png;base64,cG5n", "data:image/webp;base64,d2VicA=="],
			questions: { refund: { type: "noul" }, category: { type: "choice" }, upset: { type: "score" } },
		});
		expect(result.answers.refund).toEqual({ type: "bool", probability: 0.9 });
		expect(result.answers.upset).toEqual({ type: "score", score: 0.4, confidence: 0.6 });
		expect(Object.keys(result.answers).sort()).toEqual(["category", "refund", "upset"]);
		expect(result.usage?.cost.input).toBeCloseTo(10 * 0.03e-6, 12);
	});

	it("refuses images for the text-only model, reports a missing key, and surfaces Nace errors", async () => {
		const runtime = await setup();
		const text = runtime.getModelOfType("classifier", "nace", "drex-v2.0")!;
		const image = { type: "image" as const, data: "cG5n", mimeType: "image/png" };
		const fetch = vi.fn(async () =>
			Response.json(
				{ error: { type: "authentication_error", message: "Invalid API key" }, request_id: "req_failed" },
				{ status: 401 },
			),
		);

		const withImage = await runtime.classify(text, { ...context, images: [image] }, { fetch });
		expect(withImage.errorMessage).toBe("Model nace/drex-v2.0 does not accept image input");
		expect(fetch).not.toHaveBeenCalled();

		const unauthorized = await runtime.classify(text, context, { fetch });
		expect(unauthorized.stopReason).toBe("error");
		expect(unauthorized.errorMessage).toContain("401");
		expect(unauthorized.errorMessage).toContain("Invalid API key");
		expect(unauthorized.errorMessage).toContain("req_failed");
		expect(unauthorized.answers).toEqual({});

		await runtime.removeRuntimeApiKey("nace");
		const missing = await runtime.classify(text, context, { fetch });
		expect(missing.stopReason).toBe("error");
		expect(missing.errorMessage).toContain("not configured");
	});

	it("hides models the key may not use after discovery and keeps them when discovery is rejected", async () => {
		const runtime = await setup();
		const fetchSpy = vi.spyOn(globalThis, "fetch");
		fetchSpy.mockResolvedValue(Response.json({ models: [{ name: "drex-v2.0", alias_for: null }] }));
		await runtime.refresh({ allowNetwork: true, providers: ["nace"] });
		expect((await runtime.getAvailableOfType("classifier", "nace")).map((model) => model.id)).toEqual(["drex-v2.0"]);

		fetchSpy.mockResolvedValue(new Response("{}", { status: 401 }));
		await runtime.refresh({ allowNetwork: true, providers: ["nace"] });
		expect(await runtime.getAvailableOfType("classifier", "nace")).toHaveLength(4);
	});
});
