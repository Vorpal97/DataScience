from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

def kmean_clustering(data, n, title):
    km = KMeans(n_clusters = n, init = 'k-means++', max_iter = 300, n_init = 10, random_state= 0)
    y_means = km.fit_predict(data)
    plt.clf()
    plt.figure(figsize=(15, 8))
    plt.scatter(data[y_means == 0,0], data[y_means == 0, 1], s = 100, c = 'pink')
    plt.scatter(data[y_means == 1,0], data[y_means == 1, 1], s = 100, c = 'yellow')
    plt.scatter(data[y_means == 2,0], data[y_means == 2, 1], s = 100, c = 'magenta')
    plt.scatter(data[y_means == 3,0], data[y_means == 3, 1], s = 100, c = 'cyan')
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue')
    plt.title('KMeans Clustering', fontsize = 20)
    plt.xlabel('Total points')
    plt.ylabel('Base experience')
    plt.legend()
    plt.savefig('./grafici/' + title, dpi=356, bbox_inches='tight')
    plt.show()
