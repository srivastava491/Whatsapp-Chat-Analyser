import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array.drop(columns=['sentiment']),
                train_array['sentiment'],
                test_array.drop(columns=['sentiment']),
                test_array['sentiment']
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVC": SVC()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg', 'sag']
                },
                "AdaBoost Classifier": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "SVC": {
                    'C': [0.1, 1, 10, 100,1000],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return (best_model,r2_square)



        except Exception as e:
            raise CustomException(e, sys)