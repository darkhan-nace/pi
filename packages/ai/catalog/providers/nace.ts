import type { ClassifierModel } from "../../src/types.ts";
import { defineProvider } from "../define.ts";

// Drex serves System One at https://console.nace.ai/v1/systemone. Limits and price: https://console.nace.ai/docs/drex/reference/models
// `contextWindow` is the input-token limit of a whole request. Drex Max takes images as an `images` list of data URLs.
function drex(id: string, name: string, image: boolean): ClassifierModel<"typesafe-system-one"> {
	return {
		type: "classifier",
		id,
		name,
		api: "typesafe-system-one",
		provider: "nace",
		baseUrl: "https://console.nace.ai/v1/",
		input: image ? ["text", "image"] : ["text"],
		// $0.03 per million input tokens; output tokens are not billed.
		cost: { input: 0.03, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 262144,
		...(image && { compat: { imageInput: "images", sendImageOptInHeader: false } }),
	};
}

export default defineProvider({
	id: "nace",
	classifiers: () => [
		drex("drex-v2.0", "Drex 2.0", false),
		drex("drex-max-v1.1", "Drex Max v1.1", true),
		drex("drex-max-finance", "Drex Max Finance", true),
	],
});
