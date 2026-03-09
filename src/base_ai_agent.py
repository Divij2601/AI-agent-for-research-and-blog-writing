from __future__ import annotations

import operator
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
#defining the task node, that is a pydantic object, and will help in planning the blog writing
class Task(BaseModel):
    id: int 
    title: str
    brief: str = Field(..., description="What to cover")
    
#defining the plan node that will actually plan the blog writing with the help of 
class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[list[str], operator.add]
    final: str

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.3, 
)

def Orchestrator(state: State) -> dict:

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog with 5-7 sections on the following topic."
                )
            ),
            HumanMessage(
                content= f("Topic: {state['topic']}"),
            )
        ]
    )
    return {"plan":plan}

 


