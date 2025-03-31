import requests
from .base_llm import BaseLLM


class DeepSeekLLM(BaseLLM):
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt: str, context: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": "Ты аналитик данных. Отвечай точно, используя только предоставленные данные."
            },
            {
                "role": "user",
                "content": f"{context}\n\nВопрос: {prompt}"
            }
        ]

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            if e.response.status_code == 402:
                return "Ошибка: Необходимо пополнить баланс на platform.deepseek.com"
            raise
        except Exception as e:
            return f"DeepSeek Error: {str(e)}"
