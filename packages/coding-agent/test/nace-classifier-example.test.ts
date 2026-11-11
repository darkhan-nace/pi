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
