from flask import Blueprint, request, jsonify
from backend.run_pipeline import run

api_routes = Blueprint('api_routes', __name__)

@api_routes.route('/explain_patient', methods=['POST'])
def explain_patients():
    try:
        data = request.get_json()
        patient_index = data.get('patient_index',0)
        if patient_index is None:
            return jsonify({"error": "Missing patient_index"}), 400
        result = run(patient_index=patient_index)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500