from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()

client = MemoryClient()

messages = [
    {"role": "user", "content": "What are the main risks of AAPL?"},
    {
        "role": "assistant",
        "content": "The main risks of AAPL include supply chain disruptions, regulatory scrutiny, and competition from other tech companies.",
    },
]

client.add(messages, user_id="1")

print(client.search("What are the main risks of AAPL?", user_id="1"))
