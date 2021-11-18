 
import os
os.system('pip install pickle5')
# install pickle5 > pip install pickle5  

import pickle
import csv
import pickle5
import nltk
# nltk.download('stopwords')  
nltk.download('punkt')
nltk.download('wordnet')

import glob
import pandas as pd
import re
import string
from time import time
import numpy as np
import collections
from sklearn.feature_extraction.text import CountVectorizer
from numpy import random
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
from gensim import models
import gensim