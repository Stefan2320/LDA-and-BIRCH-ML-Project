import re
import nltk
import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from utils import load_and_describe_raw_data

class ProcessData:
    def __init__(self,dataset):
        self.dataframe = dataset

    def eliminate_labels(self):
        self.dataframe.drop(columns = ['Id','Score','ViewCount','LabelNum'])

    def clean_text(self, text):
        pattern = re.compile('<[^>]*>|\'|\(|\)|\"|”|“|\?|\.|,|:|;|&|[|]|-|\\\\')
        text = text.lower()
        text = re.sub(pattern, " ", text);
        text = nltk.word_tokenize(text)
        stop_words = stopwords.words('english')
        text = [word for word in text if word not in stop_words]

        stemmer = PorterStemmer()
        try:
            text = [stemmer.stem(word) for word in text]
            text = [word for word in text if len(word) > 1]
        except IndexError:
            pass
        text = " ".join([word for word in text])
        return text

    def merge_text_labels(self):
        self.dataframe['Content'] = self.dataframe['Title'] + self.dataframe['Body']
        self.dataframe['Content'] = self.dataframe['Content'].apply(self.clean_text)


X_train,X_valid,X_test = load_and_describe_raw_data()
dataset_train = ProcessData(X_train)
dataset_train.eliminate_labels()
dataset_train.merge_text_labels()



vect = TfidfVectorizer(max_features=20000)
vect_text = vect.fit_transform(dataset_train.dataframe['Content'])
clf = RandomForestClassifier()
clf.fit(vect_text, dataset_train.dataframe['LabelNum'])


dataset_test = ProcessData(X_test)
dataset_test.eliminate_labels()
dataset_test.merge_text_labels()
vect_text_test = vect.fit_transform(dataset_test.dataframe['Content'])
y_pred_test = clf.predict(vect_text_test)

print(accuracy_score(dataset_test.dataframe['LabelNum'], y_pred_test))

def random(X):
    random_labels = np.random.randint(0, 1, len(X['LabelNum']))
    random_accuracy = accuracy_score(X['LabelNum'], random_labels)
    print(random_accuracy)

random(X_test)