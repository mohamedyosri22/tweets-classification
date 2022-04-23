import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.impute import SimpleImputer
import time

start_time = time.time()

path = "train.tsv"
dataset = pd.read_csv(path, sep='\t')
path1 = "unlabeled_test_with_noise.tsv"
dataset1 = pd.read_csv(path1, sep='\t')

# print("dataset shape = ", dataset.shape)
# print("dataset1 shape = ", dataset1.shape)


# label encoder
le = LabelEncoder()
le.fit(dataset['Label'])


# assigning the label with and 1 INFORMATIVE  or 0 UNINFORMATIVE
dataset['Label Index'] = le.transform(dataset['Label'])


# print("class found = ",list(le.classes_))
# print('original data : \n',list(le.inverse_transform([1,0])))
#dataset['Label Index'] = dataset['Label'].map({'UNINFORMATIVE':0,'INFORMATIVE':1})

# deleting unneeded columns


del dataset['Label']
del dataset['Id']

# X & y data

X = dataset['Text']
y = dataset['Label Index']

# tokenizing the text data

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X = tfidf.fit_transform(X)

y = dataset['Label Index']

#print(X.shape)
#print(y.shape)

# spliting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=44, test_size=0.33)

# implementing the model

NBmodel = BernoulliNB(alpha=0.7)

NBmodel.fit(X_train, y_train)

print("Train data model score = ", NBmodel.score(X_train, y_train))
print("Test data model score = ", NBmodel.score(X_test, y_test))

y_pred = NBmodel.predict(X_test)


#print("Confusion Matrix :")
#print(confusion_matrix(y_test, y_pred))

print("accuracy (%) = ",accuracy_score(y_test, y_pred, normalize=True) * 100)

# for the unlabeled data
X_test_unlabeld = tfidf.transform(dataset1['Text'])

y_pred_unlabeled = NBmodel.predict(X_test_unlabeld)
# print("y pred unlabeld shape :",y_pred_unlabeled.shape)


df = pd.DataFrame(dataset1, columns=['Id', 'Text'])
# print(df.head())


df['Label'] = y_pred_unlabeled
df['Label'] = df['Label'].map({0:'INFORMATIVE',1:'UNINFORMATIVE'})

df.to_csv('unlabeled_data_classification.txt', sep='\t')

end_time = time.time()

print(f"Run time {end_time - start_time}")

# test
print(df.Text[11985], "\t", df.Label[11985])


# test

test = input()
test = tfidf.transform(test)

result = NBmodel.predict(test)

print(result)
