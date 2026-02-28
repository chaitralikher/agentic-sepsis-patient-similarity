import ollama

MODEL_NAME = "llama3:8b"

def query_ollama(prompt: str) -> str:
    """
    Send prompt to local Ollama model and return response text.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]