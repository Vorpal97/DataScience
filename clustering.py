import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

def dbscan(df, save_path):
    std_slc = StandardScaler()
    X_std = std_slc.fit_transform(df)
    dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    model = dbscan.fit(X_std)
    clusters = pd.DataFrame(model.fit_predict(X_std))
    df["Cluster"] = clusters
    print(df)
    fig = plt.figure(figsize=(10, 10));
    ax = fig.add_subplot(111)
    scatter = ax.scatter(df['catch_rate'], df['weight_kg'], c=df["Cluster"], s=50)
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Defense")
    plt.colorbar(scatter)
    if save_path != '':
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)
    plt.show()

def elbow_method(data):
    plt.clf()
    wcss = []
    for i in range (1,11):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        km.fit(data)
        wcss.append(km.inertia_)
    plt.figure(figsize=(20,8))
    plt.plot(range(1,11), wcss)
    plt.title('The Elbow Method', fontsize = 20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('wcss')
    plt.savefig('./grafici/elbow_method', dpi=356, bbox_inches='tight')
    plt.show()


def kmeans_clustering(data, n, title, xlab, ylab, label1= "", label2="", label3="", label4="", label5 = ""):
    km = KMeans(n_clusters = n, init = 'k-means++', max_iter = 300, n_init = 10, random_state= 0)
    y_means = km.fit_predict(data)
    plt.clf()
    plt.figure(figsize=(15, 8))
    plt.scatter(data[y_means == 0,0], data[y_means == 0, 1], s = 100, c = 'pink', label=label1)
    plt.scatter(data[y_means == 1,0], data[y_means == 1, 1], s = 100, c = 'yellow', label=label2)
    plt.scatter(data[y_means == 2,0], data[y_means == 2, 1], s = 100, c = 'magenta', label=label3)
    plt.scatter(data[y_means == 3,0], data[y_means == 3, 1], s = 100, c = 'cyan', label=label4)
    plt.scatter(data[y_means == 4,0], data[y_means == 4, 1], s = 100, c = 'red', label=label5)
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue')
    plt.title('KMeans Clustering', fontsize = 20)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig('./grafici/' + title, dpi=356, bbox_inches='tight')
    plt.show()


def nn(data):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    plt.clf()
    distances = np.sort(distances, axis=0)
    plt.figure(figsize=(12, 8))
    plt.plot(distances[:, 1])
    plt.savefig('./grafici/knn', dpi=356, bbox_inches='tight')
    plt.show()


def dbscan(x, title, xlab='', ylab=''):
    db = DBSCAN(eps=21, min_samples=5).fit(x)
    ymeans = db.labels_
    plt.clf()
    plt.scatter(x[ymeans == -1,0], x[ymeans == -1,1], s = 100, c = 'black')
    plt.scatter(x[ymeans == 0,0], x[ymeans == 0,1], s = 100, c = 'pink')
    plt.scatter(x[ymeans == 1,0], x[ymeans == 1,1], s = 100, c = 'magenta')
    plt.scatter(x[ymeans == 2,0], x[ymeans == 2,1], s = 100, c = 'cyan')
    plt.scatter(x[ymeans == 3,0], x[ymeans == 3,1], s = 100, c = 'yellow')
    plt.scatter(x[ymeans == 4,0], x[ymeans == 4,1], s = 100, c = 'blue')

    if not(xlab == '') and not(ylab == ''):
        plt.xlabel(xlab)
        plt.ylabel(ylab)
    else:
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
    plt.legend()
    plt.savefig('./grafici/' + title, dpi=356, bbox_inches='tight')
    plt.show()


def pca(data):
    pca = PCA(n_components=2, svd_solver='auto').fit(data)
    pca_x = pca.transform(data)
    return pca_x
