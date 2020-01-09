from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def feature_select(training_file_tf, training_file_idf, training_file_tfidf):

    k_val = [100, 500, 1000, 5000, 10000]
    chi_scores_MNB = []
    chi_scores_BNB = []
    chi_scores_KNN = []
    chi_scores_SVC = []

    mi_scores_MNB = []
    mi_scores_BNB = []
    mi_scores_KNN = []
    mi_scores_SVC = []

    #chi2
    for k in k_val:

        print("for k value " + str(k))
        # Apply MNB
        feature_vectors, targets = load_svmlight_file(training_file_tf)
        X = feature_vectors
        y = targets
        X_new1 = SelectKBest(chi2, k).fit_transform(X, y)
        clfn = MultinomialNB()

        chi_score = cross_val_score(clfn, X_new1, targets, cv=5, scoring='f1_macro')
        print("Accuracy for MultinomialNB with CHI2: %0.2f (+/- %0.2f)" % (chi_score.mean(), chi_score.std() * 2))
        chi_scores_MNB.append(chi_score.mean())

        # Apply Bernoulli NB
        feature_vectors, targets = load_svmlight_file(training_file_idf)
        X = feature_vectors
        y = targets
        X_new1 = SelectKBest(chi2, k).fit_transform(X, y)
        clfB = BernoulliNB()

        chi_score = cross_val_score(clfB, X_new1, targets, cv=5, scoring='f1_macro')
        print("Accuracy for BernoulliNB with CHI2: %0.2f (+/- %0.2f)" % (chi_score.mean(), chi_score.std() * 2))
        chi_scores_BNB.append(chi_score.mean())

        # Apply KNN
        feature_vectors, targets = load_svmlight_file(training_file_tfidf)
        X = feature_vectors
        y = targets
        X_new1 = SelectKBest(chi2, k).fit_transform(X, y)
        clfk = KNeighborsClassifier()

        chi_score = cross_val_score(clfk, X_new1, targets, cv=5, scoring='f1_macro')
        print("Accuracy for KNN with CHI2: %0.2f (+/- %0.2f)" % (chi_score.mean(), chi_score.std() * 2))
        chi_scores_KNN.append(chi_score.mean())

        # Apply SVC
        feature_vectors, targets = load_svmlight_file(training_file_tfidf)
        X = feature_vectors
        y = targets
        X_new1 = SelectKBest(chi2, k).fit_transform(X, y)
        clfsvc = SVC(gamma='auto')

        chi_score = cross_val_score(clfsvc, X_new1, targets, cv=5, scoring='f1_macro')
        print("Accuracy for SVC with CHI2: %0.2f (+/- %0.2f)" % (chi_score.mean(), chi_score.std() * 2))
        chi_scores_SVC.append(chi_score.mean())

    #MI

    for k in k_val:

        print("for k value " + str(k))
        # Apply MNB
        feature_vectors, targets = load_svmlight_file(training_file_tf)
        X = feature_vectors
        y = targets
        X_new2 = SelectKBest(mutual_info_classif, k).fit_transform(X, y)
        mi_score = cross_val_score(clfn, X_new2, targets, cv=5, scoring='f1_macro')
        print("Accuracy for MultinomialNB with MI: %0.2f (+/- %0.2f)" % (mi_score.mean(), mi_score.std() * 2))
        mi_scores_MNB.append(mi_score.mean())

        # Apply Bernoulli NB
        feature_vectors, targets = load_svmlight_file(training_file_idf)
        X = feature_vectors
        y = targets
        X_new2 = SelectKBest(mutual_info_classif, k).fit_transform(X, y)
        clfB = BernoulliNB()
        mi_score = cross_val_score(clfB, X_new2, targets, cv=5, scoring='f1_macro')
        print("Accuracy for BernoulliNB with MI: %0.2f (+/- %0.2f)" % (mi_score.mean(), mi_score.std() * 2))
        mi_scores_BNB.append(mi_score.mean())

        # Apply KNN
        feature_vectors, targets = load_svmlight_file(training_file_tfidf)
        X = feature_vectors
        y = targets
        X_new2 = SelectKBest(mutual_info_classif, k).fit_transform(X, y)
        clfk = KNeighborsClassifier()
        mi_score = cross_val_score(clfk, X_new2, targets, cv=5, scoring='f1_macro')
        print("Accuracy for KNN with MI: %0.2f (+/- %0.2f)" % (mi_score.mean(), mi_score.std() * 2))
        mi_scores_KNN.append(mi_score.mean())

        # Apply SVC
        feature_vectors, targets = load_svmlight_file(training_file_tfidf)
        X = feature_vectors
        y = targets
        X_new2 = SelectKBest(mutual_info_classif, k).fit_transform(X, y)
        clfsvc = SVC(gamma='auto')

        mi_score = cross_val_score(clfsvc, X_new2, targets, cv=5, scoring='f1_macro')
        print("Accuracy for SVC with MI: %0.2f (+/- %0.2f)" % (mi_score.mean(), mi_score.std() * 2))
        mi_scores_SVC.append(mi_score.mean())

    f,axarr=plt.subplots(2, sharex=True)
    # Plotting CHI2 graph
    axarr[0].plot(k_val, chi_scores_MNB, label="Multinomial Naive Bayes")
    axarr[0].plot(k_val, chi_scores_BNB, label="Bernoulli Naive Bayes")
    axarr[0].plot(k_val, chi_scores_KNN, label="KNN")
    axarr[0].plot(k_val, chi_scores_SVC, label="SVM")
    axarr[0].set_ylabel("Y-axis: mean score for F1 Macro metric")
    axarr[0].set_title("chi2 evaluation measure")


    # Plotting MI graph
    axarr[1].plot(k_val, mi_scores_MNB, label = "Multinomial Naive Bayes")
    axarr[1].plot(k_val, mi_scores_BNB, label = "Bernoulli Naive Bayes")
    axarr[1].plot(k_val, mi_scores_KNN, label = "KNN")
    axarr[1].plot(k_val, mi_scores_SVC, label = "SVM")
    axarr[1].set_ylabel("Y-axis: mean score for F1 Macro metric")
    axarr[1].set_title("MI evaluation measure")
    axarr[1].set_xlabel("X-axis: Number of features")
    f.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    feature_select("training_data_file.tf", "training_data_file.idf", "training_data_file.tfidf")