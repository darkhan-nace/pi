import { InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";

const context = {
	state: { ticket: "I was charged twice." },
	questions: {
		refund: {
			type: "bool" as const,
			instructions: "Does the customer want a refund?",
			criteria: { true: "Asks for a refund", false: "Does not" },
		},
	},
};

describe("ModelRuntime Nace classifier models", () => {
	afterEach(() => vi.restoreAllMocks());

	async function createRuntime(): Promise<ModelRuntime> {
		return ModelRuntime.create({
			credentials: AuthStorage.inMemory({ nace: { type: "api_key", key: "nace_sk_test" } }),
			modelsStore: new InMemoryModelsStore(),
			modelsPath: null,
			allowModelNetwork: false,
		});
	}

	it("lists only the models Nace reports for the key, asking with the pi-nace header", async () => {
		const runtime = await createRuntime();
		expect((await runtime.getAvailableOfType("classifier", "nace")).map((model) => model.id).sort()).toEqual([
			"drex-latest",
			"drex-max-finance",
			"drex-max-v1.1",
			"drex-v2.0",
		]);

		const requests: Array<{ url: string; headers: Headers }> = [];
		vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
			const url = String(input);
			requests.push({ url, headers: new Headers(init?.headers) });
			if (url === "https://console.nace.ai/v1/models") {
				return Response.json({ models: [{ name: "drex-v2.0", alias_for: null }], request_id: "req_models" });
			}
			return new Response("not found", { status: 404 });
		});
		const result = await runtime.refresh({ allowNetwork: true, providers: ["nace"] });

		expect(result.errors.size).toBe(0);
		expect((await runtime.getAvailableOfType("classifier", "nace")).map((model) => model.id)).toEqual(["drex-v2.0"]);
		const discovery = requests.find((request) => request.url === "https://console.nace.ai/v1/models");
		expect(discovery?.headers.get("authorization")).toBe("Bearer nace_sk_test");
		expect(discovery?.headers.get("x-nace-client")).toMatch(/^pi-nace\/\d+\.\d+\.\d+/);
	});

	it("classifies through the runtime with typed answers and the pi-nace header", async () => {
		const runtime = await createRuntime();
		const drex = runtime.getModelOfType("classifier", "nace", "drex-latest")!;
		const requests: Array<{ url: string; headers: Headers }> = [];
		const result = await runtime.classify(drex, context, {
			fetch: async (input, init) => {
				requests.push({ url: String(input), headers: new Headers(init?.headers) });
				return Response.json({
					answers: { refund: { type: "noul", noul: 0.97, confidence: 0.97 } },
					usage: { input_tokens: 80, output_tokens: 12 },
					request_id: "req_test",
				});
			},
		});

		expect(requests[0]?.url).toBe("https://console.nace.ai/v1/systemone");
		expect(requests[0]?.headers.get("x-nace-client")).toMatch(/^pi-nace\/\d+\.\d+\.\d+/);
		expect(result.answers.refund).toEqual({ type: "bool", probability: 0.97 });
		expect(result.usage).toMatchObject({ input: 80, output: 12 });
	});
});
