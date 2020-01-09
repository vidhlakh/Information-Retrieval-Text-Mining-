import json
from index import InvertedIndex

def class_defn_file(class_file):

    class_dict = {1: ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                      'comp.windows.x'],
                  2: ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
                  3: ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
                  4: ['misc.forsale'],
                  5: ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
                  6: ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']}

    with open(class_file, "w") as write_file:
        json.dump(class_dict, write_file, indent=2, sort_keys=True)
        print("class definition file generated successfully")

def feature_defn_file(iindex,feature_file):
    feature_dict = {}
    with open(feature_file, "w") as write_file:
        for counter, token in enumerate(iindex.items.keys()):
            feature_dict.update({counter: token})
        json.dump(feature_dict, write_file, indent=2, sort_keys=True)
        print("feature definition file generated successfully")

def training_file(feature_file, class_file, training_file_tf, training_file_idf, training_file_tfidf, iindex):

    feature_obj = open(feature_file, "r")
    fea_dict = json.load(feature_obj)
    # print("data type",type(data),"len",len(data))

    class_obj = open(class_file, "r")
    class_dict = json.load(class_obj)
    # print("data type",type(data),"len",len(data))

    '''{term:[idf,{docid:[positions],tf}]}                   
        {docid: [ class_label,{feat_id1:feat_value1},{feaid2:feval2}]}'''

    #generate training file for feature value TF
    train_dict = {}
    docids = []
    # iterate through index
    for term, values in iindex.items.items():
        # get posting list
        docids = (iindex.items.get(term).posting.keys())
        for docid in docids:
            term_id = list(fea_dict.keys())[list(fea_dict.values()).index(term)]
            term_val = iindex.items.get(term).posting.get(docid).termfreq
            class_name = iindex.items.get(term).posting.get(docid).class_name
            # if docid + class_name already present in train_dict
            if (docid+class_name) in train_dict:
                train_dict.get(docid+class_name).append({term_id: term_val})
            # If docid + class_name is not present
            else:
                for k, v in class_dict.items():
                    # print(v)
                    # print(doc.class_name)
                    if class_name in v:
                        class_label = k
                train_dict.update({(docid+class_name): [class_label, {term_id: term_val}]})

    #write to file
    with open (training_file_tf, "w") as train_obj:
        for doc,val in train_dict.items():
            x = ''
            classid = str(train_dict[doc][0])
            for i in val[1:]:
                for k,v in i.items():
                    x=x + " "+str(k)+":"+str(v)
            tfdata = classid + "\t" + x+"\n"
            train_obj.write(tfdata)
    print("training data file.tf generated successfully")

    #generating training file for feature value IDF
    train_dict = {}
    docids = []
    # iterate through index
    for term, values in iindex.items.items():
        # get posting list
        docids = (iindex.items.get(term).posting.keys())
        for docid in docids:
            term_id = list(fea_dict.keys())[list(fea_dict.values()).index(term)]
            term_val = iindex.items.get(term).idf
            class_name = iindex.items.get(term).posting.get(docid).class_name
            # if docid + class_name already present in train_dict
            if (docid + class_name) in train_dict:
                train_dict.get(docid + class_name).append({term_id: term_val})
            # If docid + class_name is not present
            else:
                for k, v in class_dict.items():
                    # print(v)
                    # print(doc.class_name)
                    if class_name in v:
                        class_label = k
                train_dict.update({(docid + class_name): [class_label, {term_id: term_val}]})

    # write to file
    with open(training_file_idf, "w") as train_obj:
        for doc, val in train_dict.items():
            x = ''
            classid = str(train_dict[doc][0])
            for i in val[1:]:
                for k, v in i.items():
                    x = x + " " + str(k) + ":" + str(v)
            idfdata = classid + "\t" + x + "\n"
            train_obj.write(idfdata)
    print("training data file.idf generated successfully")

    #generate training file for feature value TF-IDF
    train_dict = {}
    docids = []
    # iterate through index
    for term, values in iindex.items.items():
        # get posting list
        docids = (iindex.items.get(term).posting.keys())
        for docid in docids:
            term_id = list(fea_dict.keys())[list(fea_dict.values()).index(term)]
            term_val = iindex.items.get(term).posting.get(docid).termfreq * iindex.items.get(term).idf
            class_name = iindex.items.get(term).posting.get(docid).class_name
            # if docid + class_name already present in train_dict
            if (docid + class_name) in train_dict:
                train_dict.get(docid + class_name).append({term_id: term_val})
            # If docid + class_name is not present
            else:
                for k, v in class_dict.items():
                    # print(v)
                    # print(doc.class_name)
                    if class_name in v:
                        class_label = k
                train_dict.update({(docid + class_name): [class_label, {term_id: term_val}]})

    # write to file
    with open(training_file_tfidf, "w") as train_obj:
        for doc, val in train_dict.items():
            x = ''
            classid = str(train_dict[doc][0])
            for i in val[1:]:
                for k, v in i.items():
                    x = x + " " + str(k) + ":" + str(v)
            tfidfdata = classid + "\t" + x + "\n"
            train_obj.write(tfidfdata)
    print("training data file.tfidf generated successfully")

if __name__ == '__main__':

    '''class_defn_file("class_sample_file")
    index_obj = InvertedIndex()
    iindex = index_obj.indexingCranfield("sample_newsgroup")
    feature_defn_file(iindex,"feature_sample_file")
    training_file_idf("feature_sample_file", "class_sample_file", "training_sample_file.tf", "training_sample_file.idf", "training_sample_file.tfidf", iindex)'''

    class_defn_file("class_definition_file")
    index_obj = InvertedIndex()
    iindex = index_obj.indexingCranfield("mini_newsgroups")
    feature_defn_file(iindex, "feature_definition_file")
    training_file("feature_definition_file", "class_definition_file", "training_data_file.tf", "training_data_file.idf", "training_data_file.tfidf", iindex)