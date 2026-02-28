from backend.data_processing.load_data import load_feature_dataset
from backend.similarity.similarity import find_similar_patients
from backend.similarity.feature_comparison import compare_patient_to_neighbors
from backend.llm_agent.explainer import generate_explanation


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

    print("\n========== RESULT ==========")
    print("Neighborhood sepsis prevalence:", round(sepsis_rate, 3))
    print("\nClinical Explanation:")
    print(explanation)
    print("============================")


if __name__ == "__main__":
    run(patient_index=0)
