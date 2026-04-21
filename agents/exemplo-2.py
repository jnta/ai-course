import json
import os
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)


def get_stock(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    output = {
        "ticker": ticker,
        "company_name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice", 0),
    }
    return json.dumps(output)


tools = [
    {
        "type": "function",
        "name": "get_stock",
        "description": "Get stock information for a given ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, GOOGL, MSFT",
                },
            },
            "required": ["ticker"],
        },
    },
]

input_list = [{"role": "user", "content": "What is the price of Apple stock?"}]

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct", tools=tools, input=input_list
)

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = get_stock(**args)
        input_list.append(
            {"type": "function_call_output", "call_id": item.call_id, "output": result}
        )

print("Final input: ", input_list)

response = client.responses.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    tools=tools,
    instructions="For the final response, tell me what is the current price of Apple stock?",
    input=input_list,
)

print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)
