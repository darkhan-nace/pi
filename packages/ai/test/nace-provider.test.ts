import { describe, expect, it } from "vitest";
import { InMemoryCredentialStore } from "../src/auth/credential-store.ts";
import { builtinModels, getBuiltinClassifierModel, getBuiltinClassifierModels } from "../src/providers/all.ts";
import type { ClassifierContext, ClassifierModel } from "../src/types.ts";

const context: ClassifierContext = {
	state: { ticket: "I was charged twice. Please refund one charge." },
	questions: {
		category: { type: "choice", instructions: "Topic?", criteria: { billing: "Payments", technical: "Bugs" } },
		upset: { type: "score", instructions: "How upset?", criteria: ["Calm", "Angry"] },
		refund: { type: "bool", instructions: "Wants a refund?", criteria: { true: "Yes", false: "No" } },
	},
};

const wire = {
	answers: {
		category: {
			type: "choice",
			choice: "billing",
			probabilities: { billing: 0.98, technical: 0.02 },
			confidence: 0.98,
		},
		upset: { type: "score", score: 0.7, legend: ["Calm", "Angry"], confidence: 0.7 },
		refund: { type: "noul", noul: 0.95, confidence: 0.95 },
	},
	usage: { input_tokens: 100, output_tokens: 200 },
};

const image = { type: "image" as const, data: "AAAA", mimeType: "image/png" };

async function classify(id: string, request: ClassifierContext, key = "nace_sk_test") {
	const credentials = new InMemoryCredentialStore();
	if (key) await credentials.modify("nace", async () => ({ type: "api_key", key }));
	const model = getBuiltinClassifierModel("nace", id as "drex-v2.0") as ClassifierModel<"typesafe-system-one">;
	const sent: { url: string; body: Record<string, unknown> }[] = [];
	const result = await builtinModels({ credentials }).classify(model, request, {
		fetch: async (input, init) => {
			sent.push({ url: String(input), body: JSON.parse(String(init?.body)) as Record<string, unknown> });
			return Response.json(wire);
		},
	});
	return { result, sent };
}

describe("Nace classifier provider", () => {
	it("lists three classifier models priced at $0.03 per million input tokens", () => {
		const models = getBuiltinClassifierModels("nace");
		expect(models.map((model) => model.id).sort()).toEqual(["drex-max-finance", "drex-max-v1.1", "drex-v2.0"]);
		expect(models.map((model) => model.cost)).toEqual(
			models.map(() => ({ input: 0.03, output: 0, cacheRead: 0, cacheWrite: 0 })),
		);
	});

	it("returns typed answers and priced usage, and fails without a key before sending", async () => {
		const { result, sent } = await classify("drex-v2.0", context);
		expect(sent[0]?.url).toBe("https://console.nace.ai/v1/systemone");
		expect(result.answers).toMatchObject({
			category: { type: "choice", choice: "billing" },
			upset: { type: "score", score: 0.7 },
			refund: { type: "bool", probability: 0.95 },
		});
		expect(result.usage?.cost.total).toBeCloseTo(0.000003, 9);
		const missing = await classify("drex-v2.0", context, "");
		expect(missing.result.stopReason).toBe("error");
		expect(missing.sent).toEqual([]);
	});

	it("sends images to Drex Max as a list of data URLs and refuses what Drex or Pi cannot take", async () => {
		const { sent } = await classify("drex-max-v1.1", { ...context, images: [image, image] });
		expect(sent[0]?.body.images).toEqual(Array(2).fill("data:image/png;base64,AAAA"));
		const refused = [
			await classify("drex-v2.0", { ...context, images: [image] }),
			await classify("drex-max-v1.1", { ...context, images: Array(5).fill(image) }),
			await classify("drex-max-v1.1", { ...context, images: [{ ...image, mimeType: "image/gif" }] }),
		];
		expect(refused.map((run) => run.result.stopReason)).toEqual(["error", "error", "error"]);
		expect(refused[1]?.result.errorMessage).toContain("at most 4 images");
		expect(refused[2]?.result.errorMessage).toContain("image/gif");
		expect(refused.flatMap((run) => run.sent)).toEqual([]);
	});
});
