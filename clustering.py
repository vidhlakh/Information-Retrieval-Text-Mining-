from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def cluster(training_file):

    feature_vectors, targets = load_svmlight_file(training_file)
    X = feature_vectors
    y = targets
    X_new1 = SelectKBest(chi2, k=1000).fit_transform(X, y)
    X_new1 = X_new1.toarray()

    range_clusters = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    silhouette_score_kmeans = []
    mutual_information_score_kmeans = []
    silhouette_score_agglomerative = []
    mutual_information_score_agglomerative = []

    for n in range_clusters:

        print("for value " + str(n))
        kmeans_model = KMeans(n_clusters = n).fit(X_new1)
        clustering_labels = kmeans_model.labels_
        silhouette_score = metrics.silhouette_score(X_new1, clustering_labels, metric='euclidean')
        silhouette_score_kmeans.append(silhouette_score)
        print("Kmeans clustering")
        print("silhouette score is " + str(silhouette_score))

        single_linkage_model = AgglomerativeClustering(n_clusters = n, linkage='ward').fit(X_new1)
        clustering_labels = single_linkage_model.labels_
        silhouette_score = metrics.silhouette_score(X_new1, clustering_labels, metric='euclidean')
        silhouette_score_agglomerative.append(silhouette_score)
        print("Hierarchical clustering")
        print("silhouette score is " + str(silhouette_score))

    for n in range_clusters:

        print("for value " + str(n))
        kmeans_model = KMeans(n_clusters = n).fit(X_new1)
        clustering_labels = kmeans_model.labels_
        mutual_information_score = metrics.normalized_mutual_info_score(y, clustering_labels)
        mutual_information_score_kmeans.append(mutual_information_score)
        print("Kmeans clustering")
        print("mutual information score is " + str(mutual_information_score))

        single_linkage_model = AgglomerativeClustering(n_clusters = n, linkage='ward').fit(X_new1)
        clustering_labels = single_linkage_model.labels_
        mutual_information_score = metrics.normalized_mutual_info_score(y, clustering_labels)
        mutual_information_score_agglomerative.append(mutual_information_score)
        print("Hierarchical clustering")
        print("mutual information score is " + str(mutual_information_score))

    f,axarr=plt.subplots(2,sharex = True)
    axarr[0].plot(range_clusters, silhouette_score_kmeans, label="kmeans")
    axarr[0].plot(range_clusters, silhouette_score_agglomerative, label="agglomerative")
    axarr[1].plot(range_clusters, mutual_information_score_kmeans, label="kmeans")
    axarr[1].plot(range_clusters, mutual_information_score_agglomerative, label="agglomerative")
    axarr[1].set_xlabel("X-axis: Number of clusters")
    axarr[0].set_ylabel("Y-axis: silhouette score")
    axarr[1].set_ylabel("Y-axis: Mutual information score")
    axarr[0].set_title("silhouette score")
    axarr[1].set_title("mutual information score")
    f.legend(loc = "upper right")
    plt.show()

if __name__ == '__main__':
    cluster("training_data_file.tfidf")