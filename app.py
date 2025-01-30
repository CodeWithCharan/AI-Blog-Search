from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st
import os

os.environ['USER_AGENT'] = 'myagent'

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize pharma database
db = Chroma(collection_name="rag-chroma",
            embedding_function=embedding_model,
            persist_directory='./chroma_db')

# Create a Retriever Object and apply Similarity Search
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

# Create a Retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]

def add_documents_to_chromadb(url):
    try:
        # fetch the url and load the docs
        docs = WebBaseLoader(url).load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
            )

        doc_chunks = text_splitter.split_documents(docs)

        # Add chunks to database
        db.add_documents(doc_chunks)

        return "Success"
    
    except Exception as e:
        print("Error:", e)
        return None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Edges
## Check Relevance
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
# Nodes
## agent node
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatGoogleGenerativeAI(temperature=0, streaming=True, model="gemini-1.5-pro")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

## rewrite node
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                    Here is the initial question:
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

## generate node
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Initialize a Chat Prompt Template
    prompt_template = hub.pull("rlm/rag-prompt")

    # Initialize a Generator (i.e. Chat Model)
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, streaming=True)

    # Initialize a Output Parser
    output_parser = StrOutputParser()
    
    # RAG Chain
    rag_chain = prompt_template | chat_model | output_parser

    response = rag_chain.invoke({"context": docs, "question": question})
    
    return {"messages": [response]}

# graph function
def get_graph(retriever_tool):
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    return graph

def generate_message(graph, inputs):
    generated_message = ""

    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]
    
    return generated_message

def main():
    st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
    st.header("Agentic RAG with LangGraph: AI Blog Search")

    # Sidebar for Graph Visualization
    with st.sidebar:
        graph_image_path = "Images/graph_output.png"
        st.image(graph_image_path, caption="LangGraph Workflow", use_container_width=True)

        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Done"):
            if gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API key saved!")

            else:
                st.warning("Please enter your Gemini API key to proceed.")

    # Link section
    url = st.text_input(
        ":link: Paste the blog link:",
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/"
    )

    if st.button("Enter"):
        if url:
            with st.spinner("fetching documents..."):
                add_documents_to_chromadb(url)
                st.success("Done!")
        else:
            st.warning("Please paste the url")

    # graph section
    graph = get_graph(retriever_tool)

    # generate messages section
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?"
    )

    inputs = {
        "messages": [
            ("user", query),
        ]
    }

    if st.button("Submit"):
        if not query:
            st.warning("Please ask a question")
        
        else:
            with st.spinner("Thinking..."):
                response = generate_message(graph, inputs)
                st.write(response)

    st.markdown("---")
    st.write("Built with :blue-background[LangGraph] | :blue-background[Gemini] by [Charan](https://www.linkedin.com/in/codewithcharan/)")

if __name__ == "__main__":
    main()