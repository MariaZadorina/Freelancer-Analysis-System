import os
from typing import Any

import numpy as np
import pandas as pd

import config
from llm import get_llm


class FreelancerDataAnalyzer:
    def __init__(
        self,
        data_path: str = "freelancer_earnings_bd.csv",
        llm_type: str = "ollama",
    ):
        self.data_path = data_path
        self.df = self.load_and_preprocess_data()
        self.llm = self._initialize_llm(llm_type)

    def _initialize_llm(self, llm_type: str):
        """Инициализация LLM в зависимости от типа"""
        if llm_type == "yandex":
            return get_llm(
                "yandex",
                api_key=config.YANDEX_API_KEY,
                folder_id=config.YANDEX_FOLDER_ID,
            )
        elif llm_type == "deepseek":
            return get_llm("deepseek", api_key=config.DEEPSEEK_API_KEY)
        else:
            return get_llm("ollama", model_name=config.OLLAMA_MODEL)

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Загружает CSV-файл с данными
        Вызывает методы очистки и обогащения данных
        Обрабатывает возможные ошибки загрузки
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Файл данных {self.data_path} не найден.")

            df = pd.read_csv(self.data_path)
            df = self.clean_data(df)
            df = self.enrich_data(df)
            return df

        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет дубликаты
        Обрабатывает числовые столбцы:
        Преобразует в числовой тип
        Заполняет пропуски медианными значениями
        Стандартизирует категориальные данные (приводит к нижнему регистру, убирает пробелы)
        """
        df = df.drop_duplicates()

        # Заполнение пропущенных значений для числовых столбцов
        numeric_cols = [
            "Earnings_USD",
            "Hourly_Rate",
            "Job_Success_Rate",
            "Client_Rating",
            "Job_Duration_Days",
            "Rehire_Rate",
            "Marketing_Spend",
            "Job_Completed",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        # Очистка категориальных переменных
        categorical_cols = [
            "Payment_Method",
            "Client_Region",
            "Experience_Level",
            "Job_Category",
            "Platform",
            "Project_Type",
        ]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        return df

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обогащение данных дополнительными признаками:
        Добавляет бинарные флаги для способов оплаты (is_crypto, is_bank)
        Создает логарифмированный доход (log_earnings) для нормализации распределения
        Упорядочивает уровни опыта (beginner < intermediate < expert)
        """
        # Добавление флагов для разных способов оплаты
        if "Payment_Method" in df.columns:
            df["is_crypto"] = df["Payment_Method"].str.contains("crypto").astype(int)
            df["is_bank"] = df["Payment_Method"].str.contains("bank").astype(int)

        # Логарифмирование доходов для нормализации
        if "Earnings_USD" in df.columns:
            df["log_earnings"] = np.log1p(df["Earnings_USD"])

        # Категоризация уровня опыта
        if "Experience_Level" in df.columns:
            exp_level_order = ["beginner", "intermediate", "expert"]
            df["Experience_Level"] = pd.Categorical(
                df["Experience_Level"],
                categories=exp_level_order,
                ordered=True,
            )

        return df

    def analyze_data(self, question: str) -> str:
        """
        Координатор всего процесса:
        Извлекает релевантные данные по вопросу
        Формирует контекст
        Передаёт вопрос и контекст в LLM
        Обрабатывает возможные ошибки
        """
        try:
            relevant_data = self.retrieve_relevant_data(question)
            context = self.create_context(relevant_data, question)
            return self.llm.generate(question, context)
        except Exception as e:
            return f"Произошла ошибка при анализе данных: {str(e)}"

    def retrieve_relevant_data(self, question: str) -> dict[str, Any]:
        """
        Анализирует вопрос и извлекает соответствующую статистику:
        По способам оплаты (особенно сравнение криптовалюты с другими методами)
        По регионам клиентов
        По уровням опыта фрилансеров
        Для каждого типа анализа:
        Группирует данные
        Считает среднее, медиану, количество и стандартное отклонение
        Сортирует результаты
        """
        relevant_stats = {}
        question_lower = question.lower()

        # Проверяем, что нужные столбцы существуют
        if "Earnings_USD" not in self.df.columns:
            return {"error": "Столбец с доходами (Earnings_USD) не найден в данных"}

        # Анализ по способу оплаты
        if any(word in question_lower for word in ["криптовалюта", "оплата", "payment"]):
            if "Payment_Method" in self.df.columns:
                try:
                    payment_stats = (
                        self.df.groupby("Payment_Method")["Earnings_USD"]
                        .agg(
                            mean="mean",
                            median="median",
                            count="count",
                            std="std",
                        )
                        .reset_index()
                    )

                    crypto_mask = self.df["Payment_Method"].str.contains(
                        "crypto",
                        case=False,
                        na=False,
                    )
                    crypto_stats = self.df[crypto_mask]["Earnings_USD"].mean()
                    others_stats = self.df[~crypto_mask]["Earnings_USD"].mean()

                    relevant_stats["payment_stats"] = payment_stats.to_dict(orient="records")
                    relevant_stats["crypto_comparison"] = {
                        "crypto_mean": crypto_stats,
                        "others_mean": others_stats,
                        "difference": crypto_stats - others_stats,
                        "difference_pct": ((crypto_stats / others_stats) - 1) * 100
                        if others_stats != 0
                        else 0,
                    }
                except Exception as e:
                    relevant_stats["payment_error"] = str(e)

        # Анализ по региону клиента
        if any(word in question_lower for word in ["регион", "region", "client"]):
            if "Client_Region" in self.df.columns:
                try:
                    region_stats = (
                        self.df.groupby("Client_Region")["Earnings_USD"]
                        .agg(
                            mean="mean",
                            median="median",
                            count="count",
                            std="std",
                        )
                        .sort_values("mean", ascending=False)
                        .reset_index()
                    )
                    relevant_stats["region_stats"] = region_stats.to_dict(orient="records")
                except Exception as e:
                    relevant_stats["region_error"] = str(e)

        # Анализ по уровню опыта
        if any(word in question_lower for word in ["опыт", "уровень", "experience", "level"]):
            if "Experience_Level" in self.df.columns:
                try:
                    exp_stats = (
                        self.df.groupby("Experience_Level")["Earnings_USD"]
                        .agg(
                            mean="mean",
                            median="median",
                            count="count",
                            std="std",
                        )
                        .sort_index()
                        .reset_index()
                    )
                    relevant_stats["experience_stats"] = exp_stats.to_dict(orient="records")
                except Exception as e:
                    relevant_stats["experience_error"] = str(e)

        return relevant_stats

    def create_context(self, data: dict[str, Any], question: str) -> str:
        """Создание контекста для LLM на основе извлеченных данных"""
        context = "Статистические данные о фрилансерах:\n"

        if "error" in data:
            return f"Ошибка в данных: {data['error']}"

        if "payment_stats" in data:
            context += "\nДоход по способам оплаты (USD):\n"
            for item in data["payment_stats"]:
                context += (
                    f"- {item['Payment_Method']}: "
                    f"средний {item['mean']:.2f}, "
                    f"медиана {item['median']:.2f}, "
                    f"проектов {item['count']}\n"
                )

            if "crypto_comparison" in data:
                comp = data["crypto_comparison"]
                context += (
                    f"\nСравнение криптовалюты с другими способами оплаты:\n"
                    f"Средний доход при оплате криптовалютой: {comp['crypto_mean']:.2f} USD\n"
                    f"Средний доход при других способах оплаты: {comp['others_mean']:.2f} USD\n"
                    f"Разница: {comp['difference']:.2f} USD ({comp['difference_pct']:.1f}%)\n"
                )

        if "region_stats" in data:
            context += "\nДоход по регионам клиентов (USD):\n"
            for item in data["region_stats"]:
                context += (
                    f"- {item['Client_Region']}: "
                    f"средний {item['mean']:.2f}, "
                    f"медиана {item['median']:.2f}, "
                    f"проектов {item['count']}\n"
                )

        if "experience_stats" in data:
            context += "\nДоход по уровню опыта (USD):\n"
            for item in data["experience_stats"]:
                context += (
                    f"- {item['Experience_Level']}: "
                    f"средний {item['mean']:.2f}, "
                    f"медиана {item['median']:.2f}, "
                    f"проектов {item['count']}\n"
                )

        # Общая статистика
        if "Earnings_USD" in self.df.columns:
            context += (
                f"\nОбщая статистика по доходам:\n"
                f"Средний доход: {self.df['Earnings_USD'].mean():.2f} USD\n"
                f"Медианный доход: {self.df['Earnings_USD'].median():.2f} USD\n"
                f"Общее количество проектов: {len(self.df)}\n"
            )

        return context
