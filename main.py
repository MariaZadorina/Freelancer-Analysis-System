import argparse
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException

from analysis.data_analyzer import FreelancerDataAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Freelancer Analytics API",
    description="API для анализа данных фрилансеров",
    version="0.1.0",
)


class CLIApplication:
    """Класс для обработки командной строки и управления жизненным циклом приложения"""

    def __init__(self):
        self.analyzer: FreelancerDataAnalyzer | None = None

    def initialize_analyzer(self, data_path: str, llm_type: str) -> None:
        """Инициализация анализатора с проверкой входных данных"""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")

        self.analyzer = FreelancerDataAnalyzer(data_path=data_path, llm_type=llm_type)
        logger.info(f"Анализатор инициализирован с моделью {llm_type}")

    def run_interactive_mode(self) -> None:
        """Запуск интерактивного режима"""
        print("\nСистема анализа данных фрилансеров. Введите вопрос или 'exit' для выхода:")
        while True:
            try:
                question = input("\nВаш вопрос: ").strip()
                if question.lower() in {"exit", "quit", "выход"}:
                    break

                if not question:
                    print("Пожалуйста, введите вопрос.")
                    continue

                answer = self.process_question(question)
                print(f"\nОтвет:\n{answer}")

            except KeyboardInterrupt:
                print("\nЗавершение работы...")
                break
            except Exception as e:
                logger.error(f"Ошибка обработки вопроса: {str(e)}")
                print(f"Произошла ошибка: {str(e)}")

    def process_question(self, question: str) -> str:
        """Обработка одного вопроса с проверкой инициализации анализатора"""
        if not self.analyzer:
            raise RuntimeError("Анализатор не инициализирован")
        return self.analyzer.analyze_data(question)


@app.get("/analyze")
async def analyze_question(
    question: str,
    model: str = "ollama",
    data_path: str = "freelancer_earnings_bd.csv",
):
    """API endpoint для анализа вопроса"""
    try:
        analyzer = FreelancerDataAnalyzer(data_path=data_path, llm_type=model)
        answer = analyzer.analyze_data(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Система анализа данных фрилансеров",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["ollama", "yandex", "deepseek"],
        default="ollama",
        help="Выбор LLM модели",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Как распределяется доход фрилансеров в зависимости от региона проживания?",
        help="Вопрос для анализа данных",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="freelancer_earnings_bd.csv",
        help="Путь к файлу с данными",
    )
    return parser.parse_args()


def main():
    """Точка входа в приложение"""
    args = parse_arguments()
    cli_app = CLIApplication()

    try:
        cli_app.initialize_analyzer(data_path=args.data, llm_type=args.model)

        if args.question:
            answer = cli_app.process_question(args.question)
            print(f"\nОтвет на ваш вопрос:\n{answer}")
        else:
            cli_app.run_interactive_mode()

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        print(f"\nОшибка: {str(e)}")
        print("Проверьте:")
        print("1. Наличие файла данных")
        print("2. Запущен ли Ollama (ollama serve)")
        print(f"3. Загружена ли модель (ollama pull {args.model})")


if __name__ == "__main__":
    main()
