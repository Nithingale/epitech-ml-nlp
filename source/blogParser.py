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

def parseBlogXML(blogPath: str) -> list:
    tree = ET.parse(blogPath)
    return nltk.word_tokenize(''.join(post.text for post in tree.getroot().findall('post')))

def parseBlogs(blogDir: str = 'blogs') -> list:
    blogUserList = []
    stopWords = set(nltk.corpus.stopwords.words('english'))
    for blogFile in os.listdir(blogDir):
        blogPath = f'{blogDir}/{blogFile}'
        try:
            blogData = [word for word in parseBlogXML(blogPath) if word not in stopWords]
            blogUserList.append(BlogUser(*blogFile.split('.')[:-1], blogData))
        except:
            continue
    return blogUserList