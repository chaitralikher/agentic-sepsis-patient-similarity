# agentic-sepsis-patient-similarity
Agentic LLM system for ICU sepsis patient similarity using multimodal clinical embeddings and vector search
# Agentic LLM-Powered Patient Similarity & Cohort Intelligence for ICU Sepsis

## Disclaimer  ⚠️
This project was developed independently on a personal computer using publicly available datasets and synthetic data.  
It does not use, reference, or derive from any proprietary systems, workflows, schemas, or patient data.  
This repository is intended solely for educational and portfolio purposes and is not designed for clinical use.
# Agentic LLM Patient Similarity for ICU Sepsis

## Overview
This is a personal project I’ve been working on to explore how **AI and LLMs can help analyze ICU sepsis patients**.  
The goal was to build a system that can find similar patients based on both **structured ICU data** (vitals, labs) and **unstructured clinical notes**.  

I wanted to see how LLMs could orchestrate multiple tools, handle long clinical notes, and provide some level of **explainability**, all while keeping everything compliant and reproducible.

The system uses a local LLM via Ollama to generate clinically interpretable explanations of patient similarity patterns without sending data to external APIs.
Using a local LLM ensures that no clinical data is transmitted to external APIs, aligning with common healthcare privacy constraints.

---

## What This Project Does
- Finds **similar patients** in ICU sepsis datasets  
- Combines **structured embeddings** (labs, vitals) with **text embeddings** (clinical notes)  
- Uses **vector search** for fast similarity lookups  
- Includes a simple **agentic LLM layer** to interpret queries and provide explanations  
- Handles **long clinical notes** by chunking and embedding the relevant parts  

This isn’t meant for real clinical decisions — it’s a **learning & portfolio project**, but I tried to make it as realistic as possible.

---

## Why Sepsis?
Sepsis is a **complex, time-sensitive condition**.  
It’s a perfect test case for:
- Multivariate time-series data  
- Missing or delayed measurements  
- Need for explainable AI  

Working with sepsis data gave me a chance to experiment with **patient trajectories, early-warning signals, and multi-modal embeddings**, which is what real hospital AI projects often deal with.

---

## Datasets & Data Description
- **Structured data:** [Kaggle – Sepsis Prediction in ICU](https://www.kaggle.com/datasets/salikhussaini49/predict-sepsis)  

Each row represents a time-stamped clinical observation recorded during a single
ICU stay. Patients therefore have multiple observations over time, forming
longitudinal trajectories rather than multiple encounters or visits.

- **Clinical notes:** fully **synthetic**, created to simulate real ICU notes (HPI, Assessment, Plan)  

All synthetic notes are labeled as such and are only for experimentation with embeddings, chunking, and retrieval.

---

## Project Roadmap

This project is being developed in iterative phases to reflect real-world clinical AI system development.
### Version 1 — Structured Patient Similarity Engine (Current)

1. Feature engineering from ICU time-series data
2. Patient similarity modeling using clinical vitals and labs
3. Neighborhood-based sepsis risk estimation
4. LLM-generated clinical similarity explanations
5. Interactive exploration of patient risk profiles
This version establishes the core similarity and interpretability framework using structured clinical data.

### Version 2 - Multimodal Clinical Intelligence (Planned)

1. Integration of unstructured clinical notes
2. Clinical text chunking and embedding
3. Fusion of structured and unstructured patient representations
4. Enhanced similarity modeling using multimodal data
5. Context-aware LLM explanations using both numeric and narrative evidence
This phase extends the system toward real-world clinical decision support by incorporating narrative documentation.

## Methodology
### Week 1 Summary: Data Exploration

Week 1 focused on understanding the longitudinal ICU dataset: patients exhibit variable-length trajectories with septic patient timelines truncated around sepsis onset, missingness is evenly distributed across sepsis labels, and these observations establish a solid foundation for subsequent time-windowed feature engineering and patient similarity analysis.
(See `notebooks/exploration.ipynb` for full analysis and visualizations.)


### Week 2 Summary: Patient-Level Feature Construction

Patient representations were constructed using vital sign measurements from
the first 24 hours of hospital admission. Laboratory features were excluded
due to high early-window missingness (>90%). For each vital sign, statistical aggregates (mean, min, max, standard deviation) and missingness indicators were computed, followed by feature normalization to support similarity-based modeling. Sepsis labels were derived from the full ICU stay rather than the 24-hour window to avoid mislabeling patients whose sepsis onset occurred later.
(See `notebooks/feature_engineering_time_windows.ipynb` for full analysis and code.)


### Week 3 Summary: Patient Similarity and Neighborhood Analysis

Patient similarity was evaluated using a cosine-distance k-nearest neighbor
model over normalized vital sign features. Neighborhood analysis demonstrated that septic patients tended to have a higher proportion of septic neighbors,
supporting the clinical relevance of the learned similarity space.
(See `notebooks/similarity_analysis.ipynb` for full analysis and code.)

### Week 4 Summary: LLM Based Similarity Explanations

Implemented an LLM-based explanation layer that converts numeric patient similarity into human-readable clinical summaries, emphasizing interpretability and safety. (See `notebooks/llm_similarity_explanations.ipynb` for code)

### Week 5: Similarity + LLM Pipeline
Implemented an end-to-end patient similarity analysis pipeline with LLM explanations. Integrated a local LLM via Ollama (Llama 3) to generate clinical reasoning summaries. Added backend/run_pipeline.py to orchestrate similarity retrieval and explanation generation. Prepared backend components for future API deployment.

**The current prototype runs as a backend pipeline script; a REST API and interactive interface will be implemented in the next development phase.

---

## System Architecture
![architecture diagram](FinalArchitecture.jpg)

### Local LLM Setup
This project uses a local large language model via Ollama
to generate clinical explanations for patient similarity.

Setup steps:

1. Install Ollama
2. Start the service:

   ollama serve

3. Pull the model:

   ollama pull llama3:8b

4. Run the backend API


**How it works:**  
1. **Frontend (React):** Enter a query, see similar patients, explanations, and timelines  
2. **Backend (FastAPI):** Orchestrates similarity search and LLM agent queries  
3. **LLM Agent:** Interprets natural language queries and determines how to fetch similarity results  
4. **Embeddings & Vector Store:** Combines structured patient embeddings with text embeddings from notes  

---
<!--
## Handling Clinical Notes
Clinical notes can be really long, so I chunk them:
- By sections (HPI, Assessment, Plan)  
- By token limits (512–768 tokens)  

Only the most relevant chunks are fed to the LLM, so it stays within token limits and focuses on what matters.

---

 ## What I Learned
- Building **multi-modal embeddings** in healthcare is challenging but rewarding  
- Chunking text for LLMs is crucial for handling long notes  
- Vector search is a game-changer for similarity queries  
- Agentic LLM orchestration adds flexibility for querying complex clinical data 

---
--> 

## Tech Stack
- Python, Pandas, NumPy  
- FastAPI  
- React (TypeScript)  
- FAISS / vector database  
- HuggingFace Transformers (BioClinicalBERT for notes)  
- Ollama llama3 model 

---

## Disclaimer
This project is fully independent and **educational**.  
- Only public datasets and synthetic notes were used  
- No proprietary or patient-identifiable data was involved  
- Not intended for clinical use


