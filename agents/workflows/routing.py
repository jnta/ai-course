from typing import Literal
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


class CalendarEventRequestType(BaseModel):
    request_type: Literal["create_event", "update_event", "other"] = Field(
        description="The type of calendar event request"
    )
    confidence_score: float = Field(
        description="The confidence score of the request type"
    )
    event_request_text: str = Field(
        description="The calendar event request text based on the user input"
    )


class CreateEventDetails(BaseModel):
    name: str = Field(description="The name of the event")
    date: str = Field(
        description="The date of the event following ISO 8601 format, e.g. 2022-01-01"
    )
    duration_in_minutes: Optional[int] = Field(
        description="The duration of the event in minutes"
    )
    participants: list[str] = Field(description="The participants of the event")
    link: Optional[str] = Field(
        description="The link to the event or meeting, if applicable"
    )


class UpdateEvent(BaseModel):
    field_to_update: Literal["name", "date", "duration_in_minutes", "participants"] = (
        Field(description="The field to update")
    )
    new_value: str = Field(description="The new value of the field")


class UpdateEventDetails(BaseModel):
    id: str = Field(description="The ID of the event to update")
    changes: list[UpdateEvent] = Field(
        description="The changes to be made to the event"
    )


class CalendarResponse(BaseModel):
    success: bool = Field(
        description="Whether the calendar event request was processed successfully"
    )
    message: str = Field(description="The response message")
    link: Optional[str] = Field(
        description="The link to the event or meeting, if applicable"
    )


def request_router(input: str) -> CalendarEventRequestType:
    response = client.responses.parse(
        model=model,
        text_format=CalendarEventRequestType,
        input="Analyze the user input and determine if it is create_event (e.g. create a meeting), update_event (e.g. update the date of a meeting), or other (e.g. the user says hi to the bot).",
        instructions=f"Analyze this request: {input}",
    )
    return response.output_parsed


def _handle_create_event(input: str) -> CalendarResponse:
    today = datetime.now()
    date_context = f"Today is: {today.strftime('%A, %B %d, %Y')}"
    response = client.responses.parse(
        model=model,
        text_format=CreateEventDetails,
        input=f"{date_context}, based on that, extract information to create a new calendar event.",
        instructions=f"Generate a confirmation message to the user regarding the event details: {input}",
    )
    details = response.output_parsed

    duration_str = (
        f" with {details.duration_in_minutes} minutes"
        if details.duration_in_minutes is not None
        else ""
    )

    return CalendarResponse(
        success=True,
        message=f"Event {details.name} for {details.date}{duration_str} for {', '.join(details.participants)} created successfully.",
        link=f"calendar://new-event?title={details.name}&date={details.date}&duration={details.duration_in_minutes}&participants={', '.join(details.participants)}",
    )


def _handle_update_event(input: str) -> CalendarResponse:
    response = client.responses.parse(
        model=model,
        text_format=UpdateEventDetails,
        input="Based on the user input, extract information to update an existing calendar event.",
        instructions=f"Extract information to update an existing calendar event based on the following details: {input}",
    )
    details = response.output_parsed
    return CalendarResponse(
        success=True,
        message=f"Event {details.id} updated successfully.",
        link=f"calendar://update-event?id={details.id}&field_to_update={details.field_to_update}&new_value={details.new_value}",
    )


def _handle_other(input: str) -> CalendarResponse:
    return CalendarResponse(
        success=False,
        message="I'm sorry, I can't help you with that.",
        link=None,
    )


def process_calendar_request(input: str) -> CalendarResponse:
    request_router_response = request_router(input)
    if request_router_response.confidence_score < 0.7:
        return CalendarResponse(
            success=False,
            message=f"I'm not sure what you want to do. {request_router_response.description}",
            link=None,
        )

    if request_router_response.request_type == "create_event":
        return _handle_create_event(request_router_response.description)
    elif request_router_response.request_type == "update_event":
        return _handle_update_event(request_router_response.description)
    else:
        return _handle_other(request_router_response.description)


input_message = "create a meeting with John and Phil next Friday at 10am to discuss the project, it should take about 1hour."
result = process_calendar_request(input_message)
print(result.model_dump_json(indent=2))

input_message = (
    "update the date of the meeting with John and Phil to next Saturday at 10am."
)
result = process_calendar_request(input_message)
print(result.model_dump_json(indent=2))

input_message = "What is the capital of France?"
result = process_calendar_request(input_message)
print(result.model_dump_json(indent=2))
