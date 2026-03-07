from unittest import result

from backend.data_processing.load_data import load_feature_dataset
from backend.similarity.similarity import find_similar_patients
from backend.similarity.feature_comparison import compare_patient_to_neighbors
from backend.llm_agent.explainer import generate_explanation
import json


def run(patient_index=0):

    print("\nLoading dataset...")
    df = load_feature_dataset()

    print("Finding similar patients...")
    neighbors, sepsis_rate = find_similar_patients(df, patient_index)

    print("Comparing physiologic features...")
    comparison = compare_patient_to_neighbors(
        df.iloc[patient_index],
        neighbors
    )

    print("Generating clinical explanation...")
    explanation = generate_explanation(comparison, sepsis_rate)

    result = {
    "patient_index": int(patient_index),
    "neighbor_indices": neighbors.index.tolist(),
    "sepsis_prevalence": float(sepsis_rate),
    "feature_comparison": comparison,
    "clinical_explanation": explanation
}

# optional: save output
with open(f"outputs/patient_{patient_index}_explanation.json", "w") as f:
    json.dump(result, f, indent=2)

return result

if __name__ == "__main__":
    run(patient_index=0)
