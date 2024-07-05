import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object
from nltk.tokenize import wordpunct_tokenize
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        logging.info("Tfidf vectorizer initialized")
        preprocessor = ColumnTransformer(
            [
                ("tfidf_vectorizer", tfidf, 'processed_text'),
            ],
            remainder='passthrough'  # Keep non-text columns unchanged
        )

        return preprocessor

    def data_conversion(self,df):
        stop_words = set(stopwords.words('english'))
        processed_texts = []
        df.sentiment = df.sentiment.apply(lambda x: 2 if (x == 'positive') else (1 if x == 'neutral' else 0))
        for text in df['message']:
            tokens = wordpunct_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token not in stop_words]
            processed_text = " ".join(filtered_tokens)
            processed_texts.append(processed_text)

        df['processed_text'] = pd.Series(processed_texts)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Perform text data conversion
            self.data_conversion(train_df)
            self.data_conversion(test_df)

            # Initialize preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Separate features and targets
            target_column_name = "sentiment"
            input_column_name="processed_text"
            input_feature_train_df = train_df[[input_column_name]]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[[input_column_name]]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            # Transform features using preprocessor
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            # Concatenate features and targets into arrays
            train_arr = np.hstack((input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)))
            test_arr = np.hstack((input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)))

            # Save preprocessing object
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
