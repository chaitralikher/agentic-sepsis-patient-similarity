import React, { useState } from "react";

function App() {

  const [patientIndex, setPatientIndex] = useState("");
  const [result, setResult] = useState(null);

  const fetchExplanation = async () => {

    const response = await fetch("http://127.0.0.1:5000/explain_patient", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ patient_index: Number(patientIndex) })
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div style={{padding:40}}>

      <h2>ICU Patient Similarity Explorer</h2>

      <input
        type="number"
        placeholder="Enter patient index"
        value={patientIndex}
        onChange={(e)=>setPatientIndex(e.target.value)}
      />

      <button onClick={fetchExplanation}>
        Analyze Patient
      </button>

      {result && (
        <div style={{marginTop:30}}>
          <h3>Results</h3>

          <p>Sepsis prevalence: {result.sepsis_prevalence}</p>

          <p>{result.clinical_explanation}</p>

        </div>
      )}

    </div>
  );
}

export default App;
