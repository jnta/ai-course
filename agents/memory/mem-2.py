from sympy import limit
import os

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "openapi_base_url": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "collection_name": "memory",
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {
            "model": "all-MiniLM-L6-v2",
        },
    },
}

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

mem = Memory.from_config(config)


def chat_with_memory(user_id, user_message: str):
    matches = mem.search(query=user_message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in matches["results"])

    input_prompt = f"""
    You are a helpful AI assistant. Use the following memories to answer the user's question.
    Memories:
    {memories_str}
    User question: {user_message}
    """
    response = client.responses.create(model=MODEL, input=input_prompt)
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response.output_text},
    ]
    mem.add(messages, user_id=user_id)
    return response.output_text


def main():
    print("Start AI Chat")
    print("Type 'exit' to quit")

    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        response = chat_with_memory("1", user_input)
        print("AI: ", response)


if __name__ == "__main__":
    main()
