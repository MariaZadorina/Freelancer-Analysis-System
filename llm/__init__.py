from .deepseek_llm import DeepSeekLLM
from .ollama_llm import OllamaLLM
from .yandex_llm import YandexLLM


def get_llm(model_type: str, **kwargs):
    if model_type == "ollama":
        return OllamaLLM(**kwargs)
    elif model_type == "deepseek":
        return DeepSeekLLM(**kwargs)
    elif model_type == "yandex":
        return YandexLLM(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")
