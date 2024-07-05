import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exceptions import CustomException
from src.logger import logging
import os
from nltk.stem import WordNetLemmatizer
from src.utils import save_object
from nltk.tokenize import wordpunct_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation

        '''
        try:
            tfidf=TfidfVectorizer(max_features=300,ngram_range=(1,2))
            logging.info("Tfidf vectorizer initialized")
            preprocessor = ColumnTransformer(
                [
                    ("tfidf_vectorizer", tfidf, 'messages'),
                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def data_conversion(self,df):
        stop_words = set(stopwords.words('english'))
        processed_texts = []

        for text in df['message']:
            tokens = wordpunct_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token not in stop_words]
            processed_text = " ".join(filtered_tokens)
            processed_texts.append(processed_text)

        df['processed_text'] = processed_texts

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            self.data_conversion(train_df)
            self.data_conversion(test_df)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "sentiment"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
