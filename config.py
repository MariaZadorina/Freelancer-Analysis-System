import os

from dotenv import load_dotenv

load_dotenv()

# Настройки Yandex Cloud
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_CLOUD_ID = os.getenv("YANDEX_CLOUD_ID")

# Настройки Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Настройки DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
