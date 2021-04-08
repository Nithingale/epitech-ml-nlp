#!/bin/env python3

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------------------------------------------------- #

class Benchmark(object):
    def __init__(self, name: str):
        self._name = name

    def _register(self, fileName: str, testY: list, testYPredicted: list) -> None:
        with open(f'result/{fileName}.txt', 'w') as resultFile:
            print(
                f'Accuracy: {accuracy_score(testY, testYPredicted)}',
                'Micro:',
                f'\tPrecision: {precision_score(testY, testYPredicted, average = "weighted", zero_division = 0)}',
                f'\tRecall: {recall_score(testY, testYPredicted, average = "weighted", zero_division = 0)}',
                f'\tF-score: {f1_score(testY, testYPredicted, average = "weighted", zero_division = 0)}',
                'Macro:',
                f'\tPrecision: {precision_score(testY, testYPredicted, average = "macro", zero_division = 0)}',
                f'\tRecall: {recall_score(testY, testYPredicted, average = "macro", zero_division = 0)}',
                f'\tF-score: {f1_score(testY, testYPredicted, average = "macro", zero_division = 0)}',
                sep = '\n', file = resultFile
            )

class BenchmarkVectorizer(Benchmark):
    def __init__(self, name: str, vectorizer: any, classifier: any):
        super(BenchmarkVectorizer, self).__init__(name)
        self._vectorizer = vectorizer
        self._classifier = classifier

    def run(self, timeStr: str, trainingX: list, trainingY: list, testX: list, testY: list) -> None:
        pipeline = Pipeline([
            ('vectorizer', self._vectorizer),
            ('classifier', self._classifier)
        ])
        pipeline.fit(trainingX, trainingY)
        testYPredicted = pipeline.predict(testX)
        del self._vectorizer
        del self._classifier
        del pipeline
        self._register(f'{timeStr}.{self._name}', testY, testYPredicted)
        del testYPredicted
        del self

class BenchmarkClassifier(Benchmark):
    def __init__(self, name: str, classifier: any):
        super(BenchmarkClassifier, self).__init__(name)
        self._classifier = classifier

    def run(self, timeStr: str, trainingXVectorized: list, trainingY: list, testXVectorized: list, testY: list) -> None:
        self._classifier.fit(trainingXVectorized, trainingY)
        testYPredicted = self._classifier.predict(testXVectorized)
        del self._classifier
        self._register(f'{timeStr}.{self._name}', testY, testYPredicted)
        del testYPredicted
        del self

class BenchmarkBaseline(BenchmarkClassifier):
    def __init__(self, name: str):
        super(BenchmarkBaseline, self).__init__(name, DummyClassifier())

class BenchmarkNaiveBayes(BenchmarkClassifier):
    def __init__(self, nameList: tuple):
        super(BenchmarkNaiveBayes, self).__init__('+'.join(nameList), MultinomialNB())

    def run(self, timeStr: str, trainingXVectorized: list, trainingY: list, testXVectorizedList: tuple, testYList: tuple) -> None:
        self._classifier.fit(trainingXVectorized, trainingY)
        for name, testXVectorized, testY in zip(self._name.split('+'), testXVectorizedList, testYList):
            testYPredicted = self._classifier.predict(testXVectorized)
            self._register(f'{timeStr}.{name}', testY, testYPredicted)
            del testYPredicted
        del self._classifier
        del self