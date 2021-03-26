#!/bin/env python3

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------------------------------------------------------- #

def runExperiment(experimentTime: str, experimentName: str, vectorizer: any, classifier: any, trainingX: list, trainingY: list, testX: list, testY: list) -> None:
    print(f'INFO: Running the experiment \'{experimentName}\'...', flush = True)

    # Run the training and the predictions
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    pipeline.fit(trainingX, trainingY)
    testYPredicted = pipeline.predict(testX)

    # Clear the training model of the memory
    del vectorizer
    del classifier
    del pipeline

    # Calculate and write the results in a file
    with open(f'result/{experimentTime}.{experimentName}.txt', 'w') as resultFile:
        print('Accuracy:', accuracy_score(testY, testYPredicted), file = resultFile)
        print('Precision:', precision_score(testY, testYPredicted, average = 'micro'), file = resultFile)
        print('Recall:', recall_score(testY, testYPredicted, average = 'micro'), file = resultFile)
        print('F-score:', f1_score(testY, testYPredicted, average = 'micro'), file = resultFile)

    # Clear the predictions of the memory
    del testYPredicted