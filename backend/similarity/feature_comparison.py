from backend.config.settings import feature_columns


def compare_patient_to_neighbors(patient_row, neighbor_df):
    """
    Compare each feature to neighborhood mean.
    Produces qualitative comparison summary.
    """

    comparison = {}

    for feature in feature_columns:
        patient_val = patient_row[feature]
        neighbor_mean = neighbor_df[feature].mean()

        if patient_val > neighbor_mean * 1.1:
            relation = "higher"
        elif patient_val < neighbor_mean * 0.9:
            relation = "lower"
        else:
            relation = "similar"

        comparison[feature] = relation

    return comparison
