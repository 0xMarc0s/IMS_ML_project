from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def classify_bmi_comprehensive(row):
    """
    Classify BMI based on age and BMI value.

    Input:
    row: A Pandas row with 'weight', 'height', and 'age' columns.

    Output:
    Returns a string that classifies the individual into BMI categories.
    """
    # Check if weight and height are valid
    if row['height'] <= 0 or row['weight'] <= 0:
        print(row)
        return 'Invalid data'  

    # Calculate BMI
    bmi = row['weight'] / (row['height'] ** 2)

    # Age group: Children (2-19 years)
    if 2 <= row['age'] < 20:
        if bmi < 14:
            return 'Underweight'
        elif 14 <= bmi < 18:
            return 'Normal weight'
        elif 18 <= bmi < 21:
            return 'Overweight'
        else:
            return 'Obese'

    # Age group: Adults (20-64 years)
    elif 20 <= row['age'] < 65:
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 24.9:
            return 'Normal weight'
        elif 25 <= bmi < 29.9:
            return 'Overweight'
        else:
            return 'Obese'

class preprocesser():
    config_12 = {
                    "knn_imputer": ['age', 'weight', 'height'],
                    "add_bmi": None,
                    "scaler;standard": ['age', 'weight', 'height'],
                    "constant_imputer;None": ['physical_activity_perweek'],
                    "iterative_imputer;KNNclassifier": [
                                            'alcohol_freq',
                                            'caloric_freq',
                                            'devices_perday',
                                            'eat_between_meals',
                                            'gender',
                                            'meals_perday',
                                            'monitor_calories',
                                            'parent_overweight',
                                            'smoke',
                                            'transportation',
                                            'veggies_freq',
                                            'water_daily',
                                            ],
                    "encode_data;ordinal": [
                                            'alcohol_freq',
                                            'caloric_freq',
                                            'devices_perday',
                                            'meals_perday',
                                            'monitor_calories',
                                            'parent_overweight',
                                            'physical_activity_perweek',
                                            'transportation',
                                            'veggies_freq',
                                            'water_daily',
                                            "bmi_class",
                                            "eat_between_meals"
                                            ],
                    "encode_data;one_hot": [
                            "gender",
                            "smoke",
                                            ]
                } # Automatically defined configuration, done by doing some experiments

    def __init__(self):
        print("Preprocesser loaded")
    
    def add_bmi(self, data_train, data_test):
        """
        Adds BMI.
        """

        data_train['bmi_class'] = data_train.apply(lambda row: classify_bmi_comprehensive(row), axis=1)
        data_test['bmi_class'] = data_test.apply(lambda row: classify_bmi_comprehensive(row), axis=1)

        return data_train, data_test

    def encode_data(self, data_train, data_test, columns, type):
        """
        Encodes any type of data in onehot and ordinal, using a list of columns to specify which to encode.
        Data: your dataframe, duh
        columns: target
        type: specify whether to encode with onehot or ordinal
        
        RETVAL: Two dataframes with the encoded data. Transformed columns' names will be deleted and replaced with the encoded ones, following this naming convention:
        - If onehot, it will follow the format target_transformed_column
        - If ordinal, it will follow the format target_encoded
        """
        
        encoder = OneHotEncoder(sparse_output=False, drop="first")  # drop to avoid multicollinearity
        target_encoder = OrdinalEncoder() # ordinal

        encoded_train = data_train.copy()
        encoded_test = data_test.copy()
        
        if type == "one_hot":
            for target in columns:
                encoded_train[target] = encoded_train[target].astype("str")
                encoded_test[target] = encoded_test[target].astype("str")
                
                # Fit encoder on training data
                target_encoder = encoder.fit(encoded_train[[target]])
                
                # Transform training data
                transformed_train = target_encoder.transform(encoded_train[[target]])
                transformed_train = pd.DataFrame(transformed_train, columns=[f"{target}_{col}" for col in target_encoder.categories_[0][1:]])
                transformed_train.set_index(encoded_train.index, inplace=True)
                
                # Transform test data
                transformed_test = target_encoder.transform(encoded_test[[target]])
                transformed_test = pd.DataFrame(transformed_test, columns=[f"{target}_{col}" for col in target_encoder.categories_[0][1:]])
                transformed_test.set_index(encoded_test.index, inplace=True)
                
                # Merge columns
                encoded_train = encoded_train.drop(columns=[target]).join(transformed_train)
                encoded_test = encoded_test.drop(columns=[target]).join(transformed_test)
        
        elif type == "ordinal":
            for target in columns:
                encoded_train[target] = encoded_train[target].astype("str")
                encoded_test[target] = encoded_test[target].astype("str")
                
                # Fit encoder on training data
                target_encoder.fit(encoded_train[[target]])
                
                # Transform training data
                transformed_train = target_encoder.transform(encoded_train[[target]])
                encoded_train[f"{target}_encoded"] = transformed_train
                
                # Transform test data
                transformed_test = target_encoder.transform(encoded_test[[target]])
                encoded_test[f"{target}_encoded"] = transformed_test
                
                # Drop original columns
                encoded_train.drop(columns=[target], inplace=True)
                encoded_test.drop(columns=[target], inplace=True)
        
        return encoded_train, encoded_test

    def simp_imputer(self, data_train, data_val, columns, strategy='median'):
        """
        """           
        # Impute 
        for column in columns:
            imputer = SimpleImputer(strategy=strategy).fit(data_train[[column]])
            data_train[[column]] = imputer.transform(data_train[[column]])
            data_val[[column]] = imputer.transform(data_val[[column]])
        
        return data_train, data_val
        
    def scaler(self, data_train, data_val, columns, method='standard'):
        """
        """
        # Select the scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        
        # Scale the data

        for column in columns:
            scaler = scaler.fit(data_train[[column]])
            data_train[[column]] = scaler.transform(data_train[[column]])
            data_val[[column]] = scaler.transform(data_val[[column]])
        
        return data_train, data_val

    def knn_imputer(self, data_train, data_val, columns):
        """
        Impute missing values using KNNImputer.
        """
        
        # categorical to ordinal encodder
        encoder = OrdinalEncoder(encoded_missing_value=np.nan)
        for col in data_train.columns:
            if data_train[col].dtype == 'object':
                data_train[col] = encoder.fit_transform(data_train[[col]])
                data_val[col] = encoder.transform(data_val[[col]])
        
        # Initialize and fit the KNN imputer
        knn_imputer = KNNImputer(n_neighbors=5)
        data_train[columns] = knn_imputer.fit_transform(data_train[columns])
        data_val[columns] = knn_imputer.transform(data_val[columns])
        
        return data_train, data_val

    def constant_imputer(self, data_train, data_val, columns, filling):
        """
        Impute with an arbitrary value
        """
        data_train[columns] = data_train[columns].fillna(filling)
        data_val[columns] = data_val[columns].fillna(filling)

        return data_train, data_val
    
    def iterative_imputer(self, data_train, data_val, columns, estimator):
        """
        idea: let IterativeImputer train various regressions-
        NOTE: THIS FILLS ONLY NUMERICAL COLUMNS!
        """
        # Split explanatory variables from target variables
        if estimator == "lr":
            m_estimator = LinearRegression()

        elif estimator == "logistic":
            m_estimator = LogisticRegression()

        elif estimator == "KNNclassifier":
            m_estimator = KNeighborsClassifier()
    
        elif estimator == "none":
            m_estimator = None

        else:
            raise Exception("estimator", estimator, "not supported")
        # You can add more options...

        filler = IterativeImputer(estimator=m_estimator)
        filler.fit(
            X=data_train.loc[:, columns]
        )

        data_train[columns] = filler.transform(data_train[columns])
        data_val[columns] = filler.transform(data_val[columns])

        return data_train, data_val
    # Data preprocesser:
    def run(self, o_train, o_val, configs=config_12):
        """
        RUNS THE PREPROCESSER WITH A PREPROCESSING CONFIGURATION (PIPELINE) (literally some if-else statements)
        """
        # Preserve original data as preprocesser does everything inplace
        train = o_train.copy()
        val = o_val.copy()

        for config in configs:
            l = config.split(";")

            if len(l) == 1: # Single-argument options only
                option = l[0]            
                if option == "knn_imputer":
                    train, val = self.knn_imputer(train, val, configs[config])
                
                if option == "add_bmi":
                    train, val = self.add_bmi(train, val)
                
            if len(l) == 2: # Two options
                option = l[0]
                arg = l[1]

                if option == "encode_data":
                    train, val = self.encode_data(train, val, configs[config], type=arg)

                if option == "simp_imputer":
                    train, val = self.simp_imputer(train, val, configs[config], strategy=arg)
                
                if option == "scaler":
                    train, val = self.scaler(train, val, configs[config], method=arg)
                
                if option == "constant_imputer":
                    train, val = self.constant_imputer(train, val, configs[config], filling=arg)

                if option == "iterative_imputer":
                    train, val = self.iterative_imputer(train, val, configs[config], estimator=arg)
            
        return train, val 