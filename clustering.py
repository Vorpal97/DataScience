from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
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