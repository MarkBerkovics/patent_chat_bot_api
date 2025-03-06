import os
from dotenv import load_dotenv
import time

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
import langchain_core
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# ----------------
# Creating an app
# ----------------
app = FastAPI()


# ---------------------------------------
# Loading the vector store and the model
# ---------------------------------------
app.state.model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

persistant_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_store')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
app.state.vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=persistant_directory
)


# -----------------
# Building a graph
# -----------------
class State(TypedDict):
    query: str
    knowledge_base: List[Document]
    messages: list[AnyMessage]


def retrieve(state: State):
    retrieved_docs = app.state.vector_store.similarity_search(
        query=state["query"],
        k=20
    )
    return {"knowledge_base": retrieved_docs}


def generate(state: State):

    messages = state['messages'] + [HumanMessage(content=state["query"])]

    docs_content = "\n\n".join(doc.page_content for doc in state["knowledge_base"])

    model_input = messages + [AIMessage(content=docs_content)]

    response = app.state.model.invoke(model_input, config={"streaming": True})
    ai_message = AIMessage(content=response.content)

    messages = messages + [ai_message]

    return {"messages": messages}


graph_builder = StateGraph(State)

graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)

graph_builder.set_entry_point('retrieve')

graph_builder.add_edge('retrieve', 'generate')
graph_builder.add_edge('generate', END)

memory = MemorySaver()
lawyer_graph = graph_builder.compile(checkpointer=memory)
thread = {"configurable": {"thread_id": "12"}}


def generate_chat_response(messages, query):
    for m in lawyer_graph.stream({"messages": messages, "query": query}, config=thread, stream_mode='messages'):
        if isinstance(m[0], langchain_core.messages.ai.AIMessageChunk):
            if "streaming" in m[1].keys():
                yield m[0].content + ""
                time.sleep(0.1)


# --------------------
# Creating end points
# --------------------
@app.get("/")
def root():
    return {"message": "I am an expert patent lawyer"}


@app.post("/response")
async def lawyer_response(request: Request):

    data = await request.json()
    query = data.get("query", "")
    messages = data.get("messages", [])

    return StreamingResponse(generate_chat_response(messages, query), media_type="text-event/stream")
