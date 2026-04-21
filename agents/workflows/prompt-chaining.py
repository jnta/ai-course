import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
)

model = "meta-llama/llama-4-scout-17b-16e-instruct"


class EventExtraction(BaseModel):
    description: str = Field(description="The description of the event")
    is_calendar_event: bool = Field(
        description="Whether the description is related to a calendar event"
    )
    confidence_score: float = Field(
        description="The confidence score of the description being an event"
    )


class EventDetails(BaseModel):
    name: str = Field(description="The name of the event")
    date: str = Field(
        description="The date of the event following ISO 8601 format, e.g. 2022-01-01"
    )
    duration_in_minutes: Optional[int] = Field(
        description="The duration of the event in minutes"
    )
    participants: list[str] = Field(description="The participants of the event")


class Confirmation(BaseModel):
    message: str = Field(description="The message to the user")
    link: Optional[str] = Field(
        description="The link to the event or meeting, if applicable"
    )


def _extract_event_info(input: str) -> EventExtraction:
    now = datetime.now()
    date_context = f"Today is: {now.strftime('%A, %B %d, %Y')}"
    response = client.responses.parse(
        model=model,
        input=f"{date_context} analyze if the text is related to a calendar event.",
        instructions=f"Extract information about a possible event in the text. {input}",
        text_format=EventExtraction,
    )
    return response.output_parsed


def _extract_event_details(text: str) -> EventDetails:
    today = datetime.now()
    date_context = f"Today is: {today.strftime('%A, %B %d, %Y')}"
    response = client.responses.parse(
        model=model,
        input=f"{date_context} Extract details of the event. When the dates are related to the next tuesday or relative dates, use the current date as reference.",
        instructions=f"Extract structured details for the event text: {text}",
        text_format=EventDetails,
    )
    return response.output_parsed


def _confirm_event(event_details: EventDetails) -> Confirmation:
    response = client.responses.parse(
        model=model,
        input="Generate a confirmation message to the user regarding the event details.",
        instructions=f"Provide a confirmation message based on the event details: {event_details.model_dump()}",
        text_format=Confirmation,
    )
    return response.output_parsed


def process_calendar_event(text: str) -> Optional[Confirmation]:
    event_extraction = _extract_event_info(text)
    if (
        not event_extraction.is_calendar_event
        or event_extraction.confidence_score < 0.7
    ):
        return None

    event_details = _extract_event_details(text)
    confirmation = _confirm_event(event_details)
    return confirmation


input_message = "let's meet next friday at 10am to discuss the project with John and Phil, it should take about 1hour."

result = process_calendar_event(input_message)
if result:
    print("Event confirmed!")
    print(result.model_dump_json(indent=2))
else:
    print("This is not a calendar event.")
