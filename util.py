
'''
   utility functions for processing terms
'''

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def isStopWord(word):
    ''' using the NLTK functions, return true/false'''
    st_words = set(stopwords.words("english"))
    if word in st_words:
        return True
    else:
        return False

def stemming(word):
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''
    porstem = PorterStemmer()
    return porstem.stem(word)