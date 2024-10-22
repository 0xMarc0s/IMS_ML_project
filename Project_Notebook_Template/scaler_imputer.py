from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
import numpy as np

obesity_train_raw = pd.read_csv('../data/obesity_train.csv')
obesity_test_raw = pd.read_csv('../data/obesity_test.csv')

obesity_test_raw.info()
obesity_train_raw.info()

dummy_columns = ['alcohol_freq','caloric_freq','devices_perday','eat_between_meals','gender',
                 'monitor_calories','parent_overweight','physical_activity_perweek','smoke','transportation',
                 'veggies_freq','water_daily']

columns = ['age', 'meals_perday']

obesity_train = obesity_train_raw.drop(columns=['marrital_status', 'region', 'obese_level'])
obesity_val = obesity_test_raw.drop(columns=['marrital_status', 'region'])
obesity_train.info()

def dummer(data_train, data_val, columns):
    data_train = pd.get_dummies(data_train, columns=columns)
    data_val = pd.get_dummies(data_val, columns=columns)
    
    return data_train, data_val

def simp_imputer(data_train, data_val, columns, strategy='median'):
    """
    """           
    # Impute 
    for column in columns:
        imputer = SimpleImputer(strategy=strategy).fit(data_train[[column]])
        data_train[column] = imputer.transform(data_train[[column]])
        data_val[column] = imputer.transform(data_val[[column]])
    
    return data_train, data_val
    
def scaler(data_train, data_val, method='standard'):
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
    scaler = scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_val = scaler.transform(data_val)
    
    return data_train, data_val

def knn_imputer(data_train, data_val, columns):
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

obesity_train_knn_imputed, obesity_val_knn_imputed = knn_imputer(obesity_train, obesity_val, dummy_columns)
obesity_train_knn_imputed.info()


simp_imputer(obesity_train, obesity_val, dummy_columns, strategy='most_frequent')