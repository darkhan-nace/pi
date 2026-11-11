import type { ClassifierModel } from "../../src/types.ts";
import { defineProvider } from "../define.ts";

// Drex serves Nace's System One decision models from https://console.nace.ai/v1/systemone, the
// protocol TypeSafe defined, so these models use the typesafe-system-one API.
// Limits and the price (30 nUSD, $0.03 per million input tokens, the same for every model) come
// from Drex's docs: https://console.nace.ai/docs/drex/reference/models
// Output tokens are not billed. `drex-latest` points to `drex-max-v1.1` today and moves only on a
// date Nace announces, so its entry follows `drex-max-v1.1`.
// Drex Max reads PNG, JPEG, and WebP images as an `images` list of data URLs, like Clef.
// `GET /models` lists what an API key may use; the coding agent narrows this catalog to it.
const BASE_URL = "https://console.nace.ai/v1/";
const COST = { input: 0.03, output: 0, cacheRead: 0, cacheWrite: 0 };

const CLASSIFIER_MODELS: ClassifierModel<"typesafe-system-one">[] = [
	{
		type: "classifier",
		id: "drex-latest",
		name: "Drex Latest",
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: BASE_URL,
		input: ["text", "image"],
		cost: COST,
		contextWindow: 262144,
		compat: { imageInput: "images", sendImageOptInHeader: false },
	},
	{
		type: "classifier",
		id: "drex-max-v1.1",
		name: "Drex Max v1.1",
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: BASE_URL,
		input: ["text", "image"],
		cost: COST,
		contextWindow: 262144,
		compat: { imageInput: "images", sendImageOptInHeader: false },
	},
	{
		type: "classifier",
		id: "drex-max-finance",
		name: "Drex Max Finance",
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: BASE_URL,
		input: ["text", "image"],
		cost: COST,
		contextWindow: 262144,
		compat: { imageInput: "images", sendImageOptInHeader: false },
	},
	{
		type: "classifier",
		id: "drex-v2.0",
		name: "Drex 2.0",
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: BASE_URL,
		input: ["text"],
		cost: COST,
		contextWindow: 131072,
	},
];

export default defineProvider({
	id: "nace",
	classifiers: () => CLASSIFIER_MODELS,
});
