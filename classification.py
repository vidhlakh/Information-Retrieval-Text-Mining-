from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def classif(training_file_tf,training_file_idf,training_file_tfidf):

    clfn = MultinomialNB()
    feature_vectors, targets = load_svmlight_file(training_file_tf)
    scores = cross_val_score(clfn, feature_vectors, targets, cv=5, scoring='f1_macro')
    print("FI Macro Metric: Accuracy for MultinomialNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfn, feature_vectors, targets, cv=5, scoring='precision_macro')
    print("Precision Macro metric: Accuracy for MultinomialNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfn, feature_vectors, targets, cv=5, scoring='recall_macro')
    print("Recall macro metric: Accuracy for MultinomialNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clfB = BernoulliNB()
    feature_vectors, targets = load_svmlight_file(training_file_idf)
    scores = cross_val_score(clfB, feature_vectors, targets, cv=5, scoring='f1_macro')
    print("FI macro metric: Accuracy for BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfB, feature_vectors, targets, cv=5, scoring='precision_macro')
    print("Precision macro metric: Accuracy for BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfB, feature_vectors, targets, cv=5, scoring='recall_macro')
    print("Recall macro metric: Accuracy for BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clfk = KNeighborsClassifier()
    feature_vectors, targets = load_svmlight_file(training_file_tfidf)
    scores = cross_val_score(clfk, feature_vectors, targets, cv=5, scoring='f1_macro')
    print("Fi macro metric: Accuracy for K Neighbor: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfk, feature_vectors, targets, cv=5, scoring='precision_macro')
    print("Precsion macro metric: Accuracy for K Neighbor: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfk, feature_vectors, targets, cv=5, scoring='recall_macro')
    print("Recall macro metric: Accuracy for K Neighbor: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clfsvc = SVC(gamma='auto')
    feature_vectors, targets = load_svmlight_file(training_file_tfidf)
    scores = cross_val_score(clfsvc, feature_vectors, targets, cv=5, scoring='f1_macro')
    print("F1 macro metric: Accuracy for SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfsvc, feature_vectors, targets, cv=5, scoring='precision_macro')
    print("precision macro metric: Accuracy for SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clfsvc, feature_vectors, targets, cv=5, scoring='recall_macro')
    print("Recall macro metric: Accuracy for SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    classif("training_data_file.tf","training_data_file.idf","training_data_file.tfidf")