import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from typing import Dict
from llm import get_llm
import config


class FreelancerDataAnalyzer:
    def __init__(self, data_path: str = "freelancer_earnings_bd.csv", llm_type: str = "ollama"):
        self.data_path = data_path
        self.df = self.load_and_preprocess_data()
        if llm_type == "yandex":
            self.llm = get_llm(
                "yandex",
                api_key=config.YANDEX_API_KEY,
                folder_id=config.YANDEX_FOLDER_ID
            )
        elif llm_type == "deepseek":
            self.llm = get_llm("deepseek", api_key=config.DEEPSEEK_API_KEY)
        else:
            self.llm = get_llm("ollama", model_name=config.OLLAMA_MODEL)

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Загрузка и предварительная обработка данных из локального файла"""
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
        """Очистка данных"""
        df = df.drop_duplicates()

        for col in ['income', 'projects_completed']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        categorical_cols = ['payment_method', 'region', 'skill_level']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обогащение данных дополнительными признаками"""
        if 'skill_level' in df.columns:
            df['is_expert'] = (df['skill_level'] == 'expert').astype(int)

        if 'income' in df.columns:
            df['log_income'] = np.log1p(df['income'])

        return df

    def analyze_data(self, question: str) -> str:
        """Анализ данных и генерация ответа на вопрос"""
        relevant_data = self.retrieve_relevant_data(question)
        context = self.create_context(relevant_data, question)
        return self.llm.generate(question, context)

    def retrieve_relevant_data(self, question: str) -> Dict:
        """Извлечение релевантных данных на основе вопроса"""
        relevant_stats = {}
        question_lower = question.lower()

        # Анализ по способу оплаты
        if 'криптовалюта' in question_lower or 'оплата' in question_lower:
            if 'payment_method' in self.df.columns:
                payment_stats = self.df.groupby('payment_method')['income'].describe()
                relevant_stats['payment_stats'] = payment_stats.to_dict()

        # Анализ по региону
        if 'регион' in question_lower:
            if 'region' in self.df.columns:
                region_stats = self.df.groupby('region')['income'].describe()
                relevant_stats['region_stats'] = region_stats.to_dict()

        # Анализ экспертов
        if 'эксперт' in question_lower or 'проект' in question_lower:
            if 'is_expert' in self.df.columns and 'projects_completed' in self.df.columns:
                expert_stats = self.df[self.df['is_expert'] == 1]['projects_completed'].describe()
                relevant_stats['expert_stats'] = expert_stats.to_dict()
                less_than_100 = (self.df[self.df['is_expert'] == 1]['projects_completed'] < 100).mean()
                relevant_stats['expert_less_than_100'] = less_than_100

        return relevant_stats

    def create_context(self, data: Dict, question: str) -> str:
        """Создание контекста для LLM на основе извлеченных данных"""
        context = "Статистические данные о фрилансерах:\n"

        if 'payment_stats' in data:
            context += f"\nДоход по способам оплаты:\n{data['payment_stats']}\n"

        if 'region_stats' in data:
            context += f"\nРаспределение дохода по регионам:\n{data['region_stats']}\n"

        if 'expert_stats' in data:
            context += (f"\nСтатистика по количеству проектов у экспертов:\n{data['expert_stats']}\n"
                        f"Процент экспертов с менее чем 100 проектами: {data['expert_less_than_100']:.1%}\n")

        return context