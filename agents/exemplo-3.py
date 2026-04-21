import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)


def search_kb(query: str):
    response = requests.post(
        "http://localhost:8000/search", json={"query": query, "limit": 3}
    )
    return response.json()


tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Get financial information related to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in the knowledge base.",
                },
            },
            "required": ["query"],
        },
    },
]

input_list = [
    {
        "role": "user",
        "content": "What are AAPL main financial risks?",
    }
]

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct", tools=tools, input=input_list
)

input_list += response.output

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = search_kb(**args)
        texts = [r["text"] for r in result["results"]]
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"texts": texts}, ensure_ascii=False),
            }
        )

print("Final input: ", input_list)

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    instructions="For the final response, answer the user's question based on the search results.",
    input=input_list,
)

print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)
