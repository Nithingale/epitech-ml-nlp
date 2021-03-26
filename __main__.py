#!/bin/env python3

import nltk
import os
import sys
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from source.blogParser import parseBlogs
from source.experimentRunner import runExperiment

# -------------------------------------------------------------------------------------------------------------------- #

TRAINING_TEST_RATIO = 0.8

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    stopWords: set = set(nltk.corpus.stopwords.words('english'))

    print('INFO: Parsing blogs data, this might take a while...', flush = True)

    blogData: list = parseBlogs()

    print('INFO: Generating data groups...', flush = True)

    trainingTestLimit: int = int(len(blogData) * TRAINING_TEST_RATIO)
    trainingData: list = blogData[:trainingTestLimit]
    testData: list = blogData[trainingTestLimit:]

    trainingX: list = [userData.data for userData in trainingData]
    trainingX10: list = [userData.data for userData in trainingData if userData.age < 20]
    trainingX20: list = [userData.data for userData in trainingData if 20 <= userData.age < 30]
    trainingX30: list = [userData.data for userData in trainingData if 30 <= userData.age < 40]
    trainingX40: list = [userData.data for userData in trainingData if 40 <= userData.age]
    trainingYGender: list = [userData.gender for userData in trainingData]
    trainingYAge: list = [userData.age for userData in trainingData]
    trainingYActivity: list = [userData.activity for userData in trainingData]
    trainingYAstro: list = [userData.astro for userData in trainingData]

    testX: list = [userData.data for userData in testData]
    testX10: list = [userData.data for userData in testData if userData.age < 20]
    testX20: list = [userData.data for userData in testData if 20 <= userData.age < 30]
    testX30: list = [userData.data for userData in testData if 30 <= userData.age < 40]
    testX40: list = [userData.data for userData in testData if 40 <= userData.age]
    testYGender: list = [userData.gender for userData in testData]
    testYAge: list = [userData.age for userData in testData]
    testYActivity: list = [userData.activity for userData in testData]
    testYAstro: list = [userData.astro for userData in testData]

    print('INFO: Initializing experiments...', flush = True)

    experimentList: tuple = (
        # Test CountVectorizer with ngram_range on words
        (
            'CounterVectorizer.Ngram11.Word',
            CountVectorizer(ngram_range = (1, 1), analyzer = 'word'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'CounterVectorizer.Ngram12.Word',
            CountVectorizer(ngram_range = (1, 2), analyzer = 'word'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test CountVectorizer with ngram_range on characters
        (
            'CounterVectorizer.Ngram12.Char',
            CountVectorizer(ngram_range = (1, 2), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'CounterVectorizer.Ngram22.Char',
            CountVectorizer(ngram_range = (2, 2), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'CounterVectorizer.Ngram23.Char',
            CountVectorizer(ngram_range = (2, 3), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test TfidfVectorizer with ngram_range on words
        (
            'TfidfVectorizer.Ngram11.Word',
            TfidfVectorizer(ngram_range = (1, 1), analyzer = 'word'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'TfidfVectorizer.Ngram12.Word',
            TfidfVectorizer(ngram_range = (1, 2), analyzer = 'word'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test TfidfVectorizer with ngram_range on characters
        (
            'TfidfVectorizer.Ngram12.Char',
            TfidfVectorizer(ngram_range = (1, 2), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'TfidfVectorizer.Ngram22.Char',
            TfidfVectorizer(ngram_range = (2, 2), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'TfidfVectorizer.Ngram23.Char',
            TfidfVectorizer(ngram_range = (2, 3), analyzer = 'char'),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test Perceptron with max_iter
        (
            'Perceptron.Iter.10',
            TfidfVectorizer(),
            Perceptron(max_iter = 10),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'Perceptron.Iter.50',
            TfidfVectorizer(),
            Perceptron(max_iter = 50),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'Perceptron.Iter.100',
            TfidfVectorizer(),
            Perceptron(max_iter = 100),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test KNeighborsClassifier with n_neighbors
        (
            'KNeighborsClassifier.Neighbors.1',
            TfidfVectorizer(),
            KNeighborsClassifier(n_neighbors = 1),
            trainingX,
            trainingYAge,
            testX,
            testYAge
        ),
        (
            'KNeighborsClassifier.Neighbors.5',
            TfidfVectorizer(),
            KNeighborsClassifier(n_neighbors = 5),
            trainingX,
            trainingYAge,
            testX,
            testYAge
        ),
        (
            'KNeighborsClassifier.Neighbors.10',
            TfidfVectorizer(),
            KNeighborsClassifier(n_neighbors = 10),
            trainingX,
            trainingYAge,
            testX,
            testYAge
        ),
        (
            'KNeighborsClassifier.Neighbors.50',
            TfidfVectorizer(),
            KNeighborsClassifier(n_neighbors = 50),
            trainingX,
            trainingYAge,
            testX,
            testYAge
        ),
        (
            'KNeighborsClassifier.Neighbors.100',
            TfidfVectorizer(),
            KNeighborsClassifier(n_neighbors = 100),
            trainingX,
            trainingYAge,
            testX,
            testYAge
        ),

        # Test MultinomialNB with classes
        (
            'MultinomialNB.Class.10',
            TfidfVectorizer(),
            MultinomialNB(),
            trainingX10,
            trainingYAge,
            testX10,
            testYAge
        ),
        (
            'MultinomialNB.Class.20',
            TfidfVectorizer(),
            MultinomialNB(),
            trainingX20,
            trainingYAge,
            testX20,
            testYAge
        ),
        (
            'MultinomialNB.Class.30',
            TfidfVectorizer(),
            MultinomialNB(),
            trainingX30,
            trainingYAge,
            testX30,
            testYAge
        ),
        (
            'MultinomialNB.Class.40',
            TfidfVectorizer(),
            MultinomialNB(),
            trainingX40,
            trainingYAge,
            testX40,
            testYAge
        ),

        # Test DecisionTreeClassifier with max_depth
        (
            'DecisionTreeClassifier.MaxDepth.5',
            TfidfVectorizer(),
            DecisionTreeClassifier(max_depth = 5),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'DecisionTreeClassifier.MaxDepth.25',
            TfidfVectorizer(),
            DecisionTreeClassifier(max_depth = 25),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test DecisionTreeClassifier with min_samples_split
        (
            'DecisionTreeClassifier.MinSamplesSplit.2',
            TfidfVectorizer(),
            DecisionTreeClassifier(min_samples_split = 2),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'DecisionTreeClassifier.MinSamplesSplit.4',
            TfidfVectorizer(),
            DecisionTreeClassifier(min_samples_split = 4),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test DecisionTreeClassifier with min_samples_leaf
        (
            'DecisionTreeClassifier.MinSamplesLeaf.1',
            TfidfVectorizer(),
            DecisionTreeClassifier(min_samples_leaf = 1),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'DecisionTreeClassifier.MinSamplesLeaf.2',
            TfidfVectorizer(),
            DecisionTreeClassifier(min_samples_leaf = 2),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test SVC linear kernel
        (
            'SvmLinear.C.1',
            TfidfVectorizer(),
            SVC(kernel = 'linear', C = 1.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'SvmLinear.C.10',
            TfidfVectorizer(),
            SVC(kernel = 'linear', C = 10.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test SVC RBF with C
        (
            'SvmRbf.C.1',
            TfidfVectorizer(),
            SVC(kernel = 'rbf', C = 1.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'SvmRbf.C.10',
            TfidfVectorizer(),
            SVC(kernel = 'rbf', C = 10.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),

        # Test SVC RBF with
        (
            'SvmRbf.Gamma.1',
            TfidfVectorizer(),
            SVC(kernel = 'rbf', gamma = 1.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
        (
            'SvmRbf.Gamma.10',
            TfidfVectorizer(),
            SVC(kernel = 'rbf', gamma = 10.0),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        ),
    )

    print('INFO: Executing the experiments pipelines, this might take a while...', flush = True)

    if not os.path.exists('result'):
        os.makedirs('result')
    experimentTime: str = time.strftime("%d-%m-%Y.%H-%M-%S", time.gmtime())
    for experiment in experimentList:
        runExperiment(experimentTime, *experiment)

    print('INFO: Success, the results are available in the \'result\' directory.', flush = True)

    sys.exit(0)