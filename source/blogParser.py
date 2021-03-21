#!/bin/env python3

import os
import xml.etree.ElementTree as ET
import nltk

# -------------------------------------------------------------------------------------------------------------------- #

class BlogUser(object):
    def __init__(self, id: str, gender: str, age: str, activity: str, astro: str, data: list):
        self.id = int(id)
        self.gender = gender
        self.age = int(age)
        self.activity = activity
        self.astro = astro
        self.data = data

    def ngram(self, n: int) -> list:
        return [self.data[i:i + n] for i in range(len(self.data) - n + 1)]

def parseBlogs(blogDir: str = 'blogs', blogCount: int = -1, lowerWords: bool = True, removeStopWords: bool = True) -> list:
    blogUserList: list = []
    stopWords: list = nltk.corpus.stopwords.words('english')
    for blogFile in os.listdir(blogDir)[:blogCount] if blogCount >= 0 else os.listdir(blogDir):
        blogPath: str = f'{blogDir}/{blogFile}'
        try:
            blogData: list = nltk.word_tokenize(''.join(post.text for post in ET.parse(blogPath).getroot().findall('post')))
            if lowerWords:
                for word in blogData:
                    word.lower()
            if removeStopWords:
                blogData = [word for word in blogData if (word if lowerWords else word.lower()) not in stopWords]
            blogUserList.append(BlogUser(*blogFile.split('.')[:-1], blogData))
        except:
            continue
    return blogUserList