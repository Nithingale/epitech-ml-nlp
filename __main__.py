#!/bin/env python3

import os
import sys
import time

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from source.parser import parseBlogs
from source.benchmark import BenchmarkVectorizer, BenchmarkClassifier, BenchmarkBaseline, BenchmarkNaiveBayes

# -------------------------------------------------------------------------------------------------------------------- #

DATASET_PATH = './blog.csv'
USERS_COUNT: int = 10000 # Total number of users to use for training and testing (-1 = all users available)
FEATURES_COUNT: any = 5000 # Maximum number of features to use in the vectorizers (None = all features available)
TRAINING_TEST_RATIO: float = 0.8
USE_NTLK_TOKENIZER: bool = True

if __name__ == '__main__':
    # Create result folder and retrieve current time for file naming
    if not os.path.exists('result'):
        os.makedirs('result')
    timeStr: str = time.strftime("%d-%m-%Y.%H-%M-%S", time.gmtime())

    print('INFO: Parsing blogs data...', flush = True)

    blogData: list = parseBlogs(blogPath = DATASET_PATH, blogCount = USERS_COUNT)

    print('INFO: Sorting data...', flush = True)

    # Data split between training and testing
    trainingTestLimit: int = int(len(blogData) * TRAINING_TEST_RATIO)
    trainingData: list = blogData[:trainingTestLimit]
    testData: list = blogData[trainingTestLimit:]

    # Extracting data from our parser output and apply tokenizing if enabled
    if USE_NTLK_TOKENIZER:
        stopWords: set = set(stopwords.words('english'))
        trainingX: list = [' '.join(word for word in word_tokenize(userData.data) if word not in stopWords) for userData in trainingData]
        testX: list = [' '.join(word for word in word_tokenize(userData.data) if word not in stopWords) for userData in testData]
    else:
        trainingX: list = [userData.data for userData in trainingData]
        testX: list = [userData.data for userData in testData]

    trainingYGender: list = [userData.gender for userData in trainingData]
    testYGender: list = [userData.gender for userData in testData]

    trainingYAge: list = [userData.age for userData in trainingData]
    testYAge: list = [userData.age for userData in testData]

    defaultClassifier = Perceptron()

    print('INFO: Benchmarking Count Vectorizer (1/2)...', flush = True)

    BenchmarkVectorizer('CountVectorizer.Ngram12.Word', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (1, 2), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('CountVectorizer.Ngram23.Word', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (2, 3), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('CountVectorizer.Ngram34.Word', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (3, 4), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)

    print('INFO: Benchmarking Count Vectorizer (2/2)...', flush = True)

    BenchmarkVectorizer('CountVectorizer.Ngram34.Char', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (3, 4), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('CountVectorizer.Ngram45.Char', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (4, 5), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('CountVectorizer.Ngram56.Char', CountVectorizer(max_features = FEATURES_COUNT, ngram_range = (5, 6), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)

    print('INFO: Benchmarking TFIDF Vectorizer (1/2)...', flush = True)

    BenchmarkVectorizer('TfidfVectorizer.Ngram12.Word', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (1, 2), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('TfidfVectorizer.Ngram23.Word', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (2, 3), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('TfidfVectorizer.Ngram34.Word', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (3, 4), analyzer = 'word'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)

    print('INFO: Benchmarking TFIDF Vectorizer (2/2)...', flush = True)

    BenchmarkVectorizer('TfidfVectorizer.Ngram34.Char', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (3, 4), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('TfidfVectorizer.Ngram45.Char', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (4, 5), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)
    BenchmarkVectorizer('TfidfVectorizer.Ngram56.Char', TfidfVectorizer(max_features = FEATURES_COUNT, ngram_range = (5, 6), analyzer = 'char'), defaultClassifier).run(timeStr, trainingX, trainingYGender, testX, testYGender)

    print('INFO: Preparing classifier benchmarks...', flush = True)

    # Pre-vectorize the training and testing data to save that step from the classifier benchmarks
    defaultVectorizer = TfidfVectorizer(max_features = FEATURES_COUNT, lowercase = not USE_NTLK_TOKENIZER)
    trainingXVectorized = defaultVectorizer.fit_transform(trainingX)
    testXVectorized = defaultVectorizer.transform(testX)

    print('INFO: Benchmarking baselines...', flush = True)

    BenchmarkBaseline('Baseline.Gender').run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkBaseline('Baseline.Age').run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)

    print('INFO: Benchmarking Perceptron...', flush = True)

    BenchmarkClassifier('Perceptron.MaxIter1', Perceptron(max_iter = 1)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('Perceptron.MaxIter5', Perceptron(max_iter = 5)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('Perceptron.MaxIter9', Perceptron(max_iter = 9)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('Perceptron.MaxIter13', Perceptron(max_iter = 13)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('Perceptron.MaxIter17', Perceptron(max_iter = 17)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking K-Nearest Neighbors...', flush = True)

    BenchmarkClassifier('KNearestNeighbors.NNeighbors1', KNeighborsClassifier(n_neighbors = 1)).run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)
    BenchmarkClassifier('KNearestNeighbors.NNeighbors3', KNeighborsClassifier(n_neighbors = 3)).run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)
    BenchmarkClassifier('KNearestNeighbors.NNeighbors5', KNeighborsClassifier(n_neighbors = 5)).run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)
    BenchmarkClassifier('KNearestNeighbors.NNeighbors7', KNeighborsClassifier(n_neighbors = 7)).run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)
    BenchmarkClassifier('KNearestNeighbors.NNeighbors9', KNeighborsClassifier(n_neighbors = 9)).run(timeStr, trainingXVectorized, trainingYAge, testXVectorized, testYAge)

    print('INFO: Benchmarking Naive Bayes...', flush = True)

    # For Naive Bayes we actually use more testing data than normally to improve our per-age predictions
    testXAge10: list = [data for data, userData in zip(testX, testData) if userData.age < 20]
    testXAge20: list = [data for data, userData in zip(testX, testData) if 20 <= userData.age < 30]
    testXAge30: list = [data for data, userData in zip(testX, testData) if 30 <= userData.age]

    testYAge10: list = [userData.age for userData in testData if userData.age < 20]
    testYAge20: list = [userData.age for userData in testData if 20 <= userData.age < 30]
    testYAge30: list = [userData.age for userData in testData if 30 <= userData.age]

    BenchmarkNaiveBayes(('NaiveBayes.Age10', 'NaiveBayes.Age20', 'NaiveBayes.Age30')).run(timeStr, trainingXVectorized, trainingYAge, (
        defaultVectorizer.transform(testXAge10),
        defaultVectorizer.transform(testXAge20),
        defaultVectorizer.transform(testXAge30)
    ), (
        testYAge10,
        testYAge20,
        testYAge30
    ))

    print('INFO: Benchmarking Decision Tree (1/5)...', flush = True)

    BenchmarkClassifier('DecisionTree.MaxDepth1', DecisionTreeClassifier(max_depth = 1)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MaxDepth10', DecisionTreeClassifier(max_depth = 10)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MaxDepth100', DecisionTreeClassifier(max_depth = 100)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Decision Tree (2/5)...', flush = True)

    BenchmarkClassifier('DecisionTree.MinSamplesSplit5', DecisionTreeClassifier(min_samples_split = 5)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinSamplesSplit10', DecisionTreeClassifier(min_samples_split = 10)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinSamplesSplit20', DecisionTreeClassifier(min_samples_split = 20)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Decision Tree (3/5)...', flush = True)

    BenchmarkClassifier('DecisionTree.MinSamplesLeaf5', DecisionTreeClassifier(min_samples_leaf = 5)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinSamplesLeaf10', DecisionTreeClassifier(min_samples_leaf = 10)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinSamplesLeaf20', DecisionTreeClassifier(min_samples_leaf = 20)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Decision Tree (4/5)...', flush = True)

    BenchmarkClassifier('DecisionTree.MinWeightFractionLeaf0001', DecisionTreeClassifier(min_weight_fraction_leaf = 0.001)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinWeightFractionLeaf001', DecisionTreeClassifier(min_weight_fraction_leaf = 0.01)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MinWeightFractionLeaf01', DecisionTreeClassifier(min_weight_fraction_leaf = 0.1)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Decision Tree (5/5)...', flush = True)

    BenchmarkClassifier('DecisionTree.MaxFeatures5', DecisionTreeClassifier(max_features = 5)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MaxFeatures10', DecisionTreeClassifier(max_features = 10)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('DecisionTree.MaxFeatures20', DecisionTreeClassifier(max_features = 20)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Support Vector Machines (1/3)...', flush = True)

    BenchmarkClassifier('SupportVectorMachine.Linear.C1', SVC(kernel = 'linear', C = 1.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.Linear.C100', SVC(kernel = 'linear', C = 100.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.Linear.C10000', SVC(kernel = 'linear', C = 10000.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Support Vector Machines (2/3)...', flush = True)

    BenchmarkClassifier('SupportVectorMachine.RBF.C1', SVC(kernel = 'rbf', C = 1.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.RBF.C100', SVC(kernel = 'rbf', C = 100.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.RBF.C10000', SVC(kernel = 'rbf', C = 10000.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Benchmarking Support Vector Machines (3/3)...', flush = True)

    BenchmarkClassifier('SupportVectorMachine.Linear.Gamma001', SVC(kernel = 'rbf', gamma = 0.01)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.Linear.Gamma1', SVC(kernel = 'rbf', gamma = 1.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)
    BenchmarkClassifier('SupportVectorMachine.Linear.Gamma100', SVC(kernel = 'rbf', gamma = 100.0)).run(timeStr, trainingXVectorized, trainingYGender, testXVectorized, testYGender)

    print('INFO: Success, the results are available in the \'result\' directory.', flush = True)

    sys.exit(0)