import os

class Document:
    def __init__(self, docid, class_name, subject, body):
        self.docID = docid
        self.class_name = class_name
        self.subject = subject
        self.body = body

class parse_docs:
    def __init__(self, dataset):

        self.docs = []
        subject = ''
        #Scan through the directory
        dir = os.scandir(dataset)
        #Iteratory each folder
        for each_dir in dir:
            for file in os.scandir(each_dir):
                docid = file.name
                class_name = each_dir.name
                f= open(file, 'r') 
                body=[]
                buffer = False
                count_line = 0
                for line in f:
                    count_line = count_line + 1
                    if 'Subject:' in line:
                        subject=line[9:]
                    elif line.startswith('Lines:'):
                        buffer = True
                        break
                with open(file,'r') as fb:
                    body = fb.readlines()[count_line:]
                body= '\n'.join(body)
                self.docs.append(Document(docid,class_name,subject,body))

if __name__ == '__main__':
    pd=parse_docs('mini_newsgroups')
    #pd=parse_docs('sample_newsgroup')
    with open("body","a") as fb:
        for doc in pd.docs:
            fb.write(doc.docID + "\n" + doc.body)