import { typesafeSystemOneApi } from "../api/typesafe-system-one.lazy.ts";
import { envApiKeyAuth } from "../auth/helpers.ts";
import { createProvider, type Provider } from "../models.ts";
import { NACE_CLASSIFIER_MODELS } from "./nace.models.ts";

export function naceProvider(): Provider {
	return createProvider({
		id: "nace",
		name: "Nace",
		auth: {
			apiKey: envApiKeyAuth("Nace API key", ["NACE_API_KEY"]),
		},
		models: Object.values(NACE_CLASSIFIER_MODELS),
		classifiers: {
			"typesafe-system-one": typesafeSystemOneApi(),
		},
	});
}
