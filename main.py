from fastapi import FastAPI

from analysis.data_analyzer import FreelancerDataAnalyzer

app = FastAPI()

import argparse


def main():
    parser = argparse.ArgumentParser(description="Система анализа данных фрилансеров")
    parser.add_argument("--model", choices=["ollama", "yandex", "deepseek"], default="ollama")
    parser.add_argument('--question', type=str, default="Как распределяется доход фрилансеров в зависимости от региона проживания?", help="Вопрос для анализа данных")
    parser.add_argument('--data', type=str, default="freelancer_earnings_bd.csv", help="Путь к файлу с данными")
    args = parser.parse_args()

    try:
        analyzer = FreelancerDataAnalyzer(data_path=args.data, llm_type=args.model)

        if args.question:
            answer = analyzer.analyze_data(args.question)
            print(f"\nОтвет на ваш вопрос:\n{answer}")
        else:
            print(f"Система анализа данных фрилансеров (модель: {args.model}). Введите вопрос или 'exit':")
            while True:
                question = input("\nВаш вопрос: ").strip()
                if question.lower() in ['exit', 'quit', 'выход']:
                    break

                if question:
                    answer = analyzer.analyze_data(question)
                    print(f"\nОтвет:\n{answer}")
                else:
                    print("Пожалуйста, введите вопрос.")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        print("Проверьте:")
        print("1. Наличие файла данных")
        print("2. Запущен ли Ollama (ollama serve)")
        print(f"3. Загружена ли модель (ollama pull {args.model})")


if __name__ == "__main__":
    main()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
