import json
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import shutil
import os

def cluster_policies(num_clusters = 6):
    if not os.path.exists("models"):
        os.makedirs("models")

    with open('candidate_models/model_params.json', 'r') as infile:
        model_params = json.load(infile)

    all_results = [[], []]
    model_names = [[], []]
    for m in model_params:
        all_results[m["mask_option"]].append(m["all_val_results"])
        model_names[m["mask_option"]].append(m["name"])

    n_clusters_per = int(num_clusters/2)
    final_policies = []
    for mask_option in range(2):
        X = np.array(all_results[mask_option])
        n_clusters = n_clusters_per

        if len(all_results[mask_option])< n_clusters_per:
            n_clusters = len(all_results[mask_option])

        kmeans = KMeans(n_clusters=n_clusters, n_init = 10)
        kmeans.fit(X)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        closest_indexs = []

        for j in range(n_clusters):
            cluster_index = j

            cluster_indices = np.where(labels == cluster_index)[0]
            distances = [euclidean(X[i], centers[cluster_index]) for i in cluster_indices]
            closest_index = cluster_indices[np.argmin(distances)]
            closest_indexs.append(closest_index)

        for index in closest_indexs:
            final_policies.append(model_names[mask_option][index])

    final_model_params = []
    for m in model_params:
        if m["name"] in final_policies:
            del m["all_val_results"]
            final_model_params.append(m)
            shutil.copy2('candidate_models/'+m["name"], 'models/'+m["name"])

    with open('models/model_params.json', 'w') as outfile:
        json.dump(final_model_params, outfile)