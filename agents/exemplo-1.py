import os
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class FinancialMetric(BaseModel):
    metric_name: str
    value: float
    currency: str = "USD"
    change_percentage: Optional[float] = None


class CompanyAnalysis(BaseModel):
    company_name: str
    ticker: str
    fiscal_quarter: str
    revenue: FinancialMetric
    net_income: FinancialMetric
    key_highlights: List[str] = Field(description="List of strategic takeaways")
    sentiment_score: float = Field(description="Score from 0 (bearish) to 1 (bullish)")


client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

raw_text = """
NVIDIA (NVDA) reported its Q3 2024 results today. They crushed expectations with 
revenue hitting $35.1 billion, a 94% increase year-over-year. Net income soared to 
$19.31 billion. Despite the growth, CFO Colette Kress mentioned supply constraints 
for Blackwell chips. The stock showed high volatility but the outlook remains 
extremely positive for AI infrastructure spend.
"""

try:
    completion = client.beta.chat.completions.parse(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": "Extract detailed financial data into a structured JSON format.",
            },
            {"role": "user", "content": raw_text},
        ],
        response_format=CompanyAnalysis,
    )

    data = completion.choices[0].message.parsed

    print(f"--- Analysis for {data.company_name} ({data.ticker}) ---")
    print(
        f"Revenue: {data.revenue.value} {data.revenue.currency} (+{data.revenue.change_percentage}%)"
    )
    print(f"Sentiment: {'Bullish' if data.sentiment_score > 0.5 else 'Bearish'}")
    print("\nHighlights:")
    for point in data.key_highlights:
        print(f"- {point}")

except Exception as e:
    print(f"Error: {e}")
