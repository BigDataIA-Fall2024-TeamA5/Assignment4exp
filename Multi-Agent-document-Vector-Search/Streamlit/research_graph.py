from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from document_agent import DocumentAgent
from arxiv_agent import ArxivAgent
from web_search_agent import WebSearchAgent
from rag_agent import main_rag_process

# Define agents
document_agent = DocumentAgent()
arxiv_agent = ArxivAgent()
web_search_agent = WebSearchAgent()
rag_agent = main_rag_process()

# Define the `tool` functions for each agent
@tool
def document_selector(document: str):
    """Select and process a document."""
    return document_agent.run(document)

@tool
def search_arxiv(question: str):
    """Search for relevant research papers on Arxiv."""
    return arxiv_agent.run(question)

@tool
def search_web(question: str):
    """Search for answers on the web."""
    return web_search_agent.run(question)

@tool
def generate_rag_response(context: dict):
    """Generate a response using RAG."""
    return rag_agent.run(context)

# Tools list
tools = [document_selector, search_arxiv, search_web, generate_rag_response]

# Tool node
tool_node = ToolNode(tools)

# Define the function to determine the next step
def next_step(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1].content.lower()

    if "arxiv" in last_message:
        return "arxiv_search"
    elif "web" in last_message:
        return "web_search"
    elif "rag" in last_message:
        return "rag_generator"
    else:
        return END

# Create a function to invoke the agents
def invoke_agent(state: MessagesState):
    messages = state["messages"]
    document = messages[-1].content
    result = document_agent.run(document)
    return {"messages": [result]}

# Create the research graph
def create_research_graph():
    # Initialize the graph
    workflow = StateGraph(MessagesState)

    # Add nodes for each agent
    workflow.add_node("document_selector", invoke_agent)
    workflow.add_node("arxiv_search", tool_node)
    workflow.add_node("web_search", tool_node)
    workflow.add_node("rag_generator", tool_node)

    # Set the entry point
    workflow.add_edge(START, "document_selector")

    # Add conditional edges from `document_selector`
    workflow.add_conditional_edges("document_selector", next_step)

    # Add normal edges
    workflow.add_edge("arxiv_search", "rag_generator")
    workflow.add_edge("web_search", "rag_generator")
    workflow.add_edge("rag_generator", END)

    # Compile the workflow
    app = workflow.compile()
    return app

# Function to run the research graph
def run_research_graph(document, question):
    graph = create_research_graph()
    initial_state = {"messages": [HumanMessage(content=f"Document: {document}, Question: {question}")]}
    final_state = graph.invoke(initial_state)
    return final_state["messages"][-1].content