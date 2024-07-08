import logging
import sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object
import os
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, message):
        try:
            predict_pipeline_obj=PredictPipeline()
            custom_data_obj=CustomData()
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            logging.info("Model and preprocessor loading started")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Model and preprocessor loading completed")

            message=predict_pipeline_obj.message_transformer(message)
            message_df=custom_data_obj.get_data_as_data_frame(message)
            data_scaled = preprocessor.transform(message_df)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            raise CustomException(e, sys)

    def message_transformer(self,text):
        stop_words=stopwords.words('english')
        tokens = wordpunct_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        processed_text = " ".join(filtered_tokens)
        return processed_text

class CustomData:
    def __init__(self):
        pass
    def get_data_as_data_frame(self,message):
        try:
            custom_data_input_dict = {
                "processed_text":[message]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)