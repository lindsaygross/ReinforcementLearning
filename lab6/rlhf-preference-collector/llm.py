import time
from typing import Any, Dict, List, Tuple

from ollama import Client


class OllamaService:
    def __init__(self, host: str, model: str, num_predict: int = 1024):
        self.host = host
        self.model = model
        self.num_predict = num_predict
        self.client = Client(host=host)

    def _extract_model_names(self, model_list_response: Any) -> List[str]:
        models: List[str] = []

        # Dict-shaped response: {"models": [{"name": "..."}]}
        if isinstance(model_list_response, dict):
            model_items = model_list_response.get("models", [])
        else:
            # Newer client can return an object with `.models`.
            model_items = getattr(model_list_response, "models", [])

        for item in model_items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str):
                    models.append(name)
                continue

            # Object-shaped item (e.g., pydantic model)
            name = getattr(item, "name", None) or getattr(item, "model", None)
            if isinstance(name, str):
                models.append(name)

        return models

    def health_check(self) -> Tuple[bool, str]:
        try:
            model_list_response = self.client.list()
            available_models = self._extract_model_names(model_list_response)
        except Exception as exc:
            return (
                False,
                f"Unable to connect to Ollama at {self.host}. Error: {exc}",
            )

        normalized_target = self.model.split(":")[0]
        model_found = any(
            m == self.model or m.split(":")[0] == normalized_target for m in available_models
        )

        if not model_found:
            return (
                False,
                f"Model '{self.model}' is not available in Ollama. Available models: {', '.join(available_models) if available_models else 'none found'}",
            )

        return True, "Ollama is reachable and model is available."

    def generate_response(self, prompt: str, temperature: float) -> Tuple[str, int]:
        start = time.perf_counter()
        result = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "num_predict": self.num_predict,
            },
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        content = result.get("message", {}).get("content", "").strip()
        return content, latency_ms
