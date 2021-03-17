#!/bin/env python3

import nltk
import sys

from source.blogParser import parseBlogs

# -------------------------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    parseBlogs()
    sys.exit(0)