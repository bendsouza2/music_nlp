import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


df = pd.read_csv('ld_clean')

x = df['lyrics']  # features
y = df['genre']  # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)
y_train.value_counts(normalize=True)  # imbalanced dataset (83% rock, 17% rap)

# Handling imbalanced dataset
oversampled = RandomOverSampler(sampling_strategy=1.0, random_state=1)  # minority oversampling
x_over, y_over = oversampled.fit_resample(pd.DataFrame(x_train), y_train)
x_over = x_over.squeeze()

# Random forest classifier using count vectorizer
cv_forest = Pipeline([('cv', CountVectorizer()), ('rf', RandomForestClassifier(random_state=1))])
cv_forest.fit(x_over, y_over)
print(cv_forest.score(x_over, y_over), cv_forest.score(x_test, y_test))  # 94.7% accuracy on test

ConfusionMatrixDisplay.from_estimator(cv_forest, x_test, y_test, normalize='true', cmap='BuGn')
plt.title('Random Forest Classifier')
plt.savefig(os.getcwd() + '/Visualisations/Model Accuracy/rfc_cv.jpeg')


# Support vector machines using tfidf vectorizer
support_vect = Pipeline([('tfidf', TfidfVectorizer()), ('svc', svm.SVC(random_state=1, kernel='linear'))])
support_vect.fit(x_over, y_over)
print(support_vect.score(x_over, y_over), support_vect.score(x_test, y_test))

ConfusionMatrixDisplay.from_estimator(support_vect, x_test, y_test, normalize='true', cmap='BuGn')
plt.title('Support Vector Machine')
plt.savefig(os.getcwd() + '/Visualisations/Model Accuracy/svm.jpeg')
