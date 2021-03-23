#!/bin/env python3

import nltk
import os
import sys

from multiprocessing import Process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron

from source.blogParser import parseBlogs
from source.pipelineProcess import executePipeline

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
    trainingYGender: list = [userData.gender for userData in trainingData]
    trainingYAge: list = [userData.age for userData in trainingData]
    trainingYActivity: list = [userData.activity for userData in trainingData]
    trainingYAstro: list = [userData.astro for userData in trainingData]

    testX: list = [userData.data for userData in testData]
    testYGender: list = [userData.gender for userData in testData]
    testYAge: list = [userData.age for userData in testData]
    testYActivity: list = [userData.activity for userData in testData]
    testYAstro: list = [userData.astro for userData in testData]

    print('INFO: Initializing experiments...', flush = True)

    experimentList: list = [
        (
            TfidfVectorizer(),
            Perceptron(),
            trainingX,
            trainingYGender,
            testX,
            testYGender
        )
    ]

    print('INFO: Preparing result directory...', flush = True)

    if os.path.exists('result'):
        for file in os.listdir('result'):
            os.remove(f'result/{file}')
    else:
        os.makedirs('result')

    print('INFO: Executing the experiments pipelines, this might take a while...', flush = True)

    processList: list = []
    for i, experiment in enumerate(experimentList[1:]):
        processList.append(Process(target = executePipeline, args = (i + 1, *experiment)))
        processList[i].start()
    executePipeline(0, *experimentList[0])
    for process in processList:
        process.join()

    print('INFO: Success, the results are available in the \'result\' directory.', flush = True)

    sys.exit(0)