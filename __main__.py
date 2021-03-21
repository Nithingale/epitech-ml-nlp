#!/bin/env python3

import nltk
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from source.blogParser import parseBlogs

# -------------------------------------------------------------------------------------------------------------------- #

TRAINING_TEST_RATIO = 0.8

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    stopWords: set = set(nltk.corpus.stopwords.words('english'))

    blogData: list = parseBlogs()
    trainingTestLimit: int = int(len(blogData) * TRAINING_TEST_RATIO)
    trainingData: list = blogData[:trainingTestLimit]
    testData: list = blogData[trainingTestLimit:]

    vectorizer = TfidfVectorizer()
    classifier = Perceptron()
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    trainingX = [userData.data for userData in trainingData]
    trainingY = [userData.gender for userData in trainingData]
    pipeline.fit(trainingX, trainingY)

    testX = [userData.data for userData in testData]
    testY = [userData.gender for userData in testData]
    testYPredicted = pipeline.predict(testX)

    print('Accuracy:', accuracy_score(testY, testYPredicted))
    print('Precision:', precision_score(testY, testYPredicted, average = 'micro'))
    print('Recall:', recall_score(testY, testYPredicted, average = 'micro'))
    print('F-score:', f1_score(testY, testYPredicted, average = 'micro'))

    sys.exit(0)