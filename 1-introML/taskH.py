# RubricCityMedianClassifier

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def __init__(self):
        self.city_median_dict = None  # Словарь для хранения медиан средних чеков по комбинации modified_rubrics и city

    def fit(self, X, y):
        # Создаем DataFrame на основе X и y для удобства обработки
        data = pd.DataFrame({'modified_rubrics': X['modified_rubrics'], 'city': X['city'], 'average_bill': y})
        
        # Вычисляем медианы средних чеков для каждой комбинации modified_rubrics и city
        self.city_median_dict = data.groupby(['modified_rubrics', 'city'])['average_bill'].median().to_dict()
        return self

    def predict(self, X):
        # Предсказываем медиану среднего чека по комбинации modified_rubrics и city для каждого заведения
        predictions = []
        for idx, row in X.iterrows():
            key = (row['modified_rubrics'], row['city'])
            if key in self.city_median_dict:
                predictions.append(self.city_median_dict[key])
            else:
                # Если для данной комбинации нет данных, можно вернуть общую медиану или другое значение по умолчанию
                predictions.append(np.median(list(self.city_median_dict.values())))
        return np.array(predictions)