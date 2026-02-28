import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from backend.config.settings import feature_columns, TARGET_COLUMN, TOP_K_NEIGHBORS


def find_similar_patients(df, patient_index):
    """
    Compute nearest neighbors using Euclidean distance.
    Returns neighbor dataframe and sepsis prevalence.
    """

    X = df[feature_columns].values
    target_vector = X[patient_index].reshape(1, -1)

    distances = euclidean_distances(target_vector, X)[0]
    neighbor_indices = np.argsort(distances)[1:TOP_K_NEIGHBORS + 1]

    neighbors = df.iloc[neighbor_indices]
    sepsis_rate = neighbors[TARGET_COLUMN].mean()

    return neighbors, sepsis_rate
