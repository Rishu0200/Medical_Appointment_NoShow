import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data Processing step")

            logger.info("Dropping the unnecessary columns")
            df.drop(columns=['Unnamed: 0', 'PatientId', 'AppointmentID','ScheduledDay','AppointmentDay'] , inplace=True)
            df.drop_duplicates(inplace=True)


            logger.info("Reading the config file for data processing")
            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Capping Ages at 100")
            df['Age_capped'] = df['Age'].apply(lambda x: min(x, 100)) 
            logger.info("Created Age_capped column")


            df['Any_condition'] = (
                    df['Hipertension'].astype(int)
                + df['Diabetes'].astype(int)
                + df['Alcoholism'].astype(int)
                + df['Handcap'].astype(int))   
            df['Has_condition'] = (df['Any_condition'] > 0).astype(int)
            logger.info("Created Has_condition column")  



            logger.info("Dropping the columns not required for modeling")
            #df.drop(columns=['Age','Any_condition'] , inplace=True)
            df.drop_duplicates(inplace=True)

            top_Neighbourhood = self.config["data_processing"]["top_Neighbourhood"]
            top_neigh = df['Neighbourhood'].value_counts().nlargest(top_Neighbourhood).index

            logger.info("Grouping rare categories in Neighbourhood column as Other")
            df['Neighbourhood_grouped'] = df['Neighbourhood'].where(
                df['Neighbourhood'].isin(top_neigh),
                other='Other')
            #df.drop(columns=['Neighbourhood'], inplace=True)
            logger.info("Created Neighbourhood_grouped column")

            logger.info("Applying log transformation to Date.diff column")
            df['Date_diff_log'] = np.log1p(df['Date.diff'])
            #df.drop(columns=['Date.diff'], inplace=True)
            logger.info("Created Date_diff_log column")

            drop_cols = ['Age','Any_condition','Neighbourhood','Date.diff']
            df_model = df.drop(columns=drop_cols)
            logger.info(f"Dropped columns: {drop_cols}")

            logger.info("Encoding categorical variables")
            df_model_encoded = df_model.copy()
            # map Gender: F=0, M=1  
            df_model_encoded['Gender'] = df_model_encoded['Gender'].map({'F': 0, 'M': 1}).astype(int)

            # convert all boolean columns to int
            bool_cols = ['Scholarship', 'Hipertension', 'Diabetes',
                        'Alcoholism', 'Handcap', 'SMS_received',
                        'Has_condition', 'Showed_up']

            df_model_encoded[bool_cols] = df_model_encoded[bool_cols].astype(int)
            df_model_encoded = pd.get_dummies(
            df_model_encoded,
            columns=['Neighbourhood_grouped'],
            drop_first=True)
            df_model_encoded = df_model_encoded.dropna(subset=['Date_diff_log']).reset_index(drop=True)
            logger.info("Categorical encoding completed")

            return df_model_encoded
                
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data", e)    

    def balance_data(self,df_model_encoded):
        try:
            logger.info("Handling Imbalanced Data")
            X = df_model_encoded.drop(columns='Showed_up')
            y = df_model_encoded['Showed_up']

            # keep only numeric columns
            X_num = X.select_dtypes(include=[np.number])

            # 1) any inf?
            np.isinf(X_num.values).any()

            # 2) which columns have inf?
            cols_with_inf = X_num.columns[np.isinf(X_num).any(axis=0)]
            print(cols_with_inf)

            # 3) show problematic rows
            rows_with_inf = np.isinf(X_num[cols_with_inf]).any(axis=1)
            X.loc[rows_with_inf, cols_with_inf]

            X[cols_with_inf] = X[cols_with_inf].replace([np.inf, -np.inf], np.nan)
            X = X.dropna().reset_index(drop=True)

            # keep y aligned if you have it
            y = y.loc[X.index].reset_index(drop=True)

            smote = SMOTE(random_state=42)
            X_res , y_res = smote.fit_resample(X,y)
                                  
            logger.info("Data balanced sucesffuly")
            
            return X_res, y_res
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e) 

    def save_processed_data(self, X_res, y_res, processed_train_path, processed_test_path, test_size=0.2):
        try:
            logger.info("Saving Processed Data")

            # Combine X_res and y_res back into a single DataFrame
            df_resampled = pd.concat([X_res, y_res], axis=1)

            # Shuffle the data
            df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

            # Split into train and test sets
            split_index = int((1 - test_size) * len(df_resampled))
            train_df = df_resampled.iloc[:split_index]
            test_df = df_resampled.iloc[split_index:]

            # Save to CSV files
            train_df.to_csv(processed_train_path, index=False)
            test_df.to_csv(processed_test_path, index=False)

            logger.info(f"Processed training data saved at: {processed_train_path}")
            logger.info(f"Processed testing data saved at: {processed_test_path}")

        except Exception as e:
            logger.error(f"Error during saving processed data step {e}")
            raise CustomException("Error while saving processed data", e)    

    def run(self):
        try:
            # Load data
            df = load_data(self.train_path)

            # Preprocess data
            df_model_encoded = self.preprocess_data(df)

            # Balance data
            X_res, y_res = self.balance_data(df_model_encoded)

            # Save processed data
            self.save_processed_data(
                X_res,
                y_res,
                PROCESSED_TRAIN_DATA_PATH,
                PROCESSED_TEST_DATA_PATH
            )

            logger.info("Data Processing Completed Successfully")

        except Exception as e:
            logger.error(f"Error in DataProcessor run method: {e}")
            raise CustomException("Error in DataProcessor run method", e)

if __name__ == "__main__":
    data_processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_processor.run() 