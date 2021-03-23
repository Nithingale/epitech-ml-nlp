#!/bin/env python3

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------------------------------------------------- #

def executePipeline(pipelineId: int, vectorizer: any, classifier: any, trainingX: list, trainingY: list, testX: list, testY: list) -> None:
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    pipeline.fit(trainingX, trainingY)
    testYPredicted = pipeline.predict(testX)

    with open(f'result/{pipelineId}.txt', 'w') as resultFile:
        print('Accuracy:', accuracy_score(testY, testYPredicted), file = resultFile)
        print('Precision:', precision_score(testY, testYPredicted, average = 'micro'), file = resultFile)
        print('Recall:', recall_score(testY, testYPredicted, average = 'micro'), file = resultFile)
        print('F-score:', f1_score(testY, testYPredicted, average = 'micro'), file = resultFile)