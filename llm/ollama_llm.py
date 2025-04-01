import logging
from dataclasses import dataclass

import requests
from requests.exceptions import RequestException
from requests.exceptions import Timeout

from llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout: int = 200  # seconds
    default_model: str = "llama3:latest"


class OllamaLLM(BaseLLM):
    """Профессиональная реализация LLM для Ollama с полной обработкой ошибок"""

    def __init__(
        self,
        model_name: str = None,
        config: OllamaConfig | None = None,
    ):
        self.config = config or OllamaConfig()
        self.model_name = self._resolve_model_name(model_name)
        self._validate_connection()

    def _resolve_model_name(self, model_name: str | None) -> str:
        """Определяет имя модели с fallback на default"""
        return (
            self.config.default_model
            if model_name is None
            else self._normalize_model_name(model_name)
        )

    def _normalize_model_name(self, name: str) -> str:
        """Нормализует имя модели (добавляет :latest при необходимости)"""
        if ":" not in name:
            return f"{name}:latest"
        return name

    def _validate_connection(self) -> None:
        """Проверяет доступность Ollama и модели"""
        try:
            models = self._get_available_models()
            if not any(m["name"] == self.model_name for m in models):
                raise ModelNotFoundError(
                    f"Модель '{self.model_name}' не найдена. "
                    f"Доступные модели: {[m['name'] for m in models]}\n"
                    f"Попробуйте: 'ollama pull {self.model_name.split(':')[0]}'",
                )
        except RequestException as e:
            raise OllamaConnectionError(
                f"Не удалось подключиться к Ollama: {str(e)}. "
                "Убедитесь, что сервер запущен (ollama serve).",
            ) from e

    def _get_available_models(self) -> list:
        """Получает список доступных моделей с обработкой таймаута"""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except Timeout:
            logger.error(f"Таймаут запроса к Ollama: {self.config.timeout} сек")
            raise
        except RequestException as e:
            logger.error(f"Ошибка запроса к Ollama: {str(e)}")
            raise

    def generate(self, prompt: str, context: str = "") -> str:
        """Генерирует ответ с полной обработкой ошибок и таймаутами"""
        full_prompt = self._build_prompt(prompt, context)

        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()["response"]

        except Timeout:
            error_msg = f"Таймаут запроса ({self.config.timeout} сек) к модели {self.model_name}"
            logger.error(error_msg)
            raise OllamaTimeoutError(error_msg)
        except KeyError as e:
            error_msg = f"Неверный формат ответа от Ollama: {str(e)}"
            logger.error(error_msg)
            raise OllamaResponseError(error_msg)
        except RequestException as e:
            error_msg = f"Ошибка запроса к Ollama: {str(e)}"
            logger.error(error_msg)
            raise OllamaConnectionError(error_msg)

    def _build_prompt(self, prompt: str, context: str) -> str:
        """Строит оптимизированный промпт для модели"""
        return (
            "Ты аналитик данных. Ответь точно и по делу.\n"
            f"Контекст: {context}\n"
            f"Вопрос: {prompt}\n"
            "Ответ должен быть основан только на предоставленных данных."
        )


class OllamaTimeoutError(Exception):
    """Кастомное исключение для таймаутов"""

    pass


class OllamaResponseError(Exception):
    """Кастомное исключение для ошибок ответа"""

    pass


class ModelNotFoundError(Exception):
    """Кастомное исключение для отсутствующих моделей"""

    pass


class OllamaConnectionError(Exception):
    """Кастомное исключение для проблем подключения"""

    pass
