#!/bin/env python3

import nltk
import sys

from source.blogParser import BlogUser, parseBlogs

# -------------------------------------------------------------------------------------------------------------------- #

TRAINING_TEST_RATIO = 0.8

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    blogData: list = parseBlogs(blogCount = 100)
    trainingTestLimit: int = int(len(blogData) * TRAINING_TEST_RATIO)
    trainingData: list = blogData[:trainingTestLimit]
    testData: list = blogData[trainingTestLimit:]

    sys.exit(0)