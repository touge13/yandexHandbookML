# CityMeanRegressor

from sklearn.base import RegressorMixin
import numpy as np
import pandas as pd

class CityMeanRegressor(RegressorMixin):
    def __init__(self):
        self.city_mean_dict = None  # Словарь для хранения средних значений по городам

    def fit(self, X, y):
        # Создаем DataFrame на основе X и y для удобства обработки
        data = pd.DataFrame({'city': X['city'], 'average_bill': y})
        
        # Вычисляем средние значения average_bill для каждого города
        self.city_mean_dict = data.groupby('city')['average_bill'].mean().to_dict()
        return self

    def predict(self, X):
        # Предсказываем среднее значение по городу для каждого заведения
        predictions = [self.city_mean_dict[city] if city in self.city_mean_dict else 0 for city in X['city']]
        return np.array(predictions)