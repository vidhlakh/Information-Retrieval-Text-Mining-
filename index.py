'''
Index structure:
    The Index class contains a list of IndexItems, stored in a dictionary type for easier access
    each IndexItem contains the term and a set of PostingItems
    each PostingItem contains a document ID and a list of positions that the term occurs
'''
from util import isStopWord
from util import stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import jsonpickle
import json
import sys
import os
from parsedocs import parse_docs

class Posting:

    def __init__(self, docID, class_name):
        self.docID = docID
        self.class_name = class_name
        self.positions = []
        self.termfreq = 0;

    def append(self, pos):
        self.positions.append(pos)
        # adding term frequency here
        self.termfreq = self.termfreq + 1;

class IndexItem:

    def __init__(self, term):
        self.term = term
        self.posting = {}  # postings are stored in a python dict for easier index building
        self.idf = 0;

    def add(self, docid, pos, class_name):
        ''' add a posting'''
        if docid not in self.posting:
            self.posting[docid] = Posting(docid, class_name)
        self.posting[docid].append(pos)

class InvertedIndex:

    def __init__(self):
        self.items = {}  # list of IndexItems

    def indexDoc(self, doc):  # indexing a Document object
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''

        # ToDo: indexing only title and body; use some functions defined in util.py

        titletoken = word_tokenize(doc.subject)
        bodytoken = word_tokenize(doc.body)
        tokens = titletoken + bodytoken
        
        for counter,token in enumerate(tokens):
            #remove stop words from token
            token_is_stopword=isStopWord(token)
            if(token_is_stopword):
                tokens.pop(counter)
                continue
            #perform stemming
            stemmedToken = stemming(token)
            positionindoc=counter
            tokens[counter]=stemmedToken
            tempindexitem = IndexItem(tokens[counter])
            if (stemmedToken in self.items):
                self.items.get(stemmedToken).add(doc.docID, positionindoc,doc.class_name)
            else:
                tempindexitem.add(doc.docID, positionindoc,doc.class_name)
                self.items[stemmedToken] = tempindexitem
            positionindoc = positionindoc + len(tokens[counter]) + 1;

    def find(self, term):
        return self.items[term]

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        fh = open(filename, 'w')
        jsonEncoded = jsonpickle.encode(self)
        fh.write(jsonEncoded)
        #print("item length",len(self.items))
        
    def load(self, filename):
        ''' load from disk'''
        f = open(filename, "r")
        jsonString = f.read()
        # print(jsonString)
        self = jsonpickle.decode(jsonString)
        return self

    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        # ToDo: return the IDF of the term
        rawvalue = 2000 / (len(list(self.items[term].posting.keys())))
        self.items[term].idf = math.log(rawvalue, 10)

    def indexingCranfield(self, collectionname):

        # ToDo: indexing the Cranfield dataset and save the index to a file
        # command line usage: "python index.py cran.all index_file"
        # the index is saved to index_file
        pd = parse_docs(collectionname)
        iindex = InvertedIndex()
        for doc in pd.docs:
            iindex.indexDoc(doc)
        for terms in iindex.items:
            # print(terms)
            iindex.idf(terms)

            # iindex.save(indexfilename)
        print("Total Tokens " + str(len(iindex.items)))
        print("Index builded successfully")
        return iindex

def test():
    print("***************************** testing ****************************")
    ''' test your code thoroughly. put the testing cases here'''
    try:
        statinfo = os.stat('index_file')
        if statinfo.st_size > 25000:
            print('index generated successfully')
        else:
            print('index is incomplete')
    except:
        print('index file not found')
        sys.exit()

    testing_index = InvertedIndex()
    testing_index = testing_index.load("index_file")

    # Stopword Removal Test
    stopword_check = "found"
    for key, values in testing_index.items.items():
        # print(key)
        if ('and' not in str(key)):
            stopword_check = "not found"
    print("Stopword " + stopword_check)

    # Stemming Done?
    stemming_check = "not done"
    ps = PorterStemmer()
    stem_temp = ps.stem("approximation")
    print("After stemming 'approximation' should be " + stem_temp)
    for key, values in testing_index.items.items():
        if ('approximation' not in str(key)):
            stemming_check = "done"
    print("Stemming is " + stemming_check)

    # Check TF and TDF computed value?
    if testing_index.items['experiment'].idf == 0.619788758288394:
        print("idf calculated properly")
    else:
        print('idf is incorrect')

    if testing_index.items['experiment'].posting.get('1').termfreq == 3:
        print('term frequency is  correct')
    else:
        print('term frequency is incorrect')
    print('Pass')
    print("***************************** testing ****************************")

if __name__ == '__main__':

    iindex = InvertedIndex()
    iindex.indexingCranfield("mini_newsgroups")
    #iindex.indexingCranfield("sample_newsgroup")
    #test()