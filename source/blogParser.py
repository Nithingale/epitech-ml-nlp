#!/bin/env python3

import csv
import sys

# -------------------------------------------------------------------------------------------------------------------- #

class BlogUser(object):
    def __init__(self, gender: str, age: str, activity: str, astro: str, data: str):
        self.gender: str = gender
        self.age: int = int(age)
        self.activity: str = activity
        self.astro: str = astro
        self.data: str = data

    def __repr__(self):
        return f'BlogUser({self.__dict__})'

def parseBlogs(blogPath: str = 'blog.csv', blogCount: int = -1) -> list:
    blogUserDict: dict = {}
    with open(blogPath, errors = 'ignore', newline = '') as blogFile:
        csv.field_size_limit(min(sys.maxsize, 1000000))
        for row in csv.reader(line.replace('\0', '') for line in blogFile if line):
            if 0 <= blogCount == len(blogUserDict.keys()):
                break
            try:
                userId: int = int(row[0])
                postText: str = row[6]
                if userId not in blogUserDict:
                    blogUserDict[userId] = BlogUser(*row[1:5], postText)
                else:
                    blogUserDict[userId].data += postText
            except:
                continue
    return list(blogUserDict.values())