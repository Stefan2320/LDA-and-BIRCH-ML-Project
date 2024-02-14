import os
import numpy as np
import pandas as pd
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

'''
Load dataset
'''
nltk.download('punkt')
nltk.download('stopwords')

def dataset_details(dataframe):
    print(f"Labels:{dataframe.columns}")
    print(f"See dataset balance:\n{dataframe['Label'].value_counts()}")
    print(dataframe.head())
    print(dataframe.info(),end='\n')
    print(dataframe.isnull().sum())


def load_and_describe_raw_data(path="./dataset"):

    train_path = os.path.join(path,'train.csv')
    X_train = pd.read_csv(train_path)


    test_path = os.path.join(path, 'test.csv')
    X_test = pd.read_csv(test_path)


    valid_path = os.path.join(path, 'valid.csv')
    X_valid = pd.read_csv(valid_path)

    dataset_details(X_train)
    return X_train,X_valid,X_test

