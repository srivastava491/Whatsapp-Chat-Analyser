import logging
import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object
import os
from src.components.data_transformation import DataTransformation
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            data_transformation_obj=DataTransformation()
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            logging.info("Model and preprocessor loading started")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Model and preprocessor loading completed")
            data_transformation_obj.feature_column_processer(features)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,message):
        self.message = message

    def get_data_as_data_frame(self,message):
        try:
            custom_data_input_dict = {
                "message":[self.message]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

