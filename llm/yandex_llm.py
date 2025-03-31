import requests
from .base_llm import BaseLLM


class YandexLLM(BaseLLM):
    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def generate(self, prompt: str, context: str = "") -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-folder-id": self.folder_id
        }

        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandex-gpt-lite",
            "messages": [
                {
                    "role": "system",
                    "text": "Ты аналитик данных. Отвечай точно, используя только предоставленные данные."
                },
                {
                    "role": "user",
                    "text": f"{context}\n\nВопрос: {prompt}"
                }
            ],
            "temperature": 0.3
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            return response.json()["result"]["alternatives"][0]["message"]["text"]
        except Exception as e:
            return f"YandexGPT Error: {str(e)}"
