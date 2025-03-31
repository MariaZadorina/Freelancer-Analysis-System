import requests
from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "llama3"):
        self.base_url = "http://localhost:11434/api/generate"
        self.model_name = model_name

    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = f"""Ты аналитик данных. Проанализируй информацию и ответь на вопрос. Контекст: {context} Вопрос: {prompt} Ответь максимально точно, используя только предоставленные данные."""

        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            return response.json()["response"]
        except Exception as e:
            return f"Ошибка: {str(e)}. Убедитесь, что Ollama запущен (ollama serve) и модель {self.model_name} загружена (ollama pull {self.model_name})"

