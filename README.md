# Machine Learning for NLP
## Setup
- Install the dependencies by running ```pip install -r requirements.txt``` in a terminal.
- Download NLTK data using ```python -m nltk.downloader all```.
- Place the dataset (downloadable here https://www.kaggle.com/rtatman/blog-authorship-corpus) where you want it and make sure that the path matches with what is indicated in the variable ```DATASET_PATH``` of the ```__main__.py``` file.
- Run the program by entering ```python .``` from its root directory.

## Notes
- The program runs a lot of benchmarks, and some take a while to finish. Plan at least an hour to run the whole program.
- The program was developed and tested on a computer equipped with 32GB of RAM. If you encounter memory errors, reduce the ```USERS_COUNT``` and the ```FEATURES_COUNT``` variables of the ```__main__.py``` file.