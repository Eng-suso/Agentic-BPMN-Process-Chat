from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_community.vectorstores import FAISS 


load_dotenv()


llm = ChatOpenAI(
    model_name="gpt-4.1-mini",
    temperature=0.25,
    max_completion_tokens=1000,
    timeout=(10, 60),
    max_retries=3,
    stream_usage=True,
)

loader = PyPDFLoader(r"C:\Users\sohay\Desktop\documenti bpmn\formal-13-12-09.pdf")
docs = loader.load()

len(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
chunks = splitter.split_documents(docs)

len(chunks)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")



#retrival 

retriever= vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})



@tool
def rag_tool(query:str) -> str:
    """
    Retrieve relevant documents from the vector store based on the query and generate a response using the LLM.
    """
    documents = retriever.invoke(query)
    if not documents:
        return "No relevant information was found in the document."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown Source")
        page_number = document.metadata.get("page", "Unknown Page")

        formatted_documents.append(f"Document {index} (Source: {source}, Page: {page_number}):\n{document.page_content}")

    return "\n\n".join(formatted_documents)

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current or external information.

    Use this tool when the user asks about recent events, current facts,
    live data, companies, people, products, documentation, or anything
    that may not be available in the local PDF/vector store.

    Args:
        query: The search query to send to the web search engine.
    """
    results = web_search_tool.invoke({"query": query})

    if isinstance(results, dict):
        items = results.get("results", [])
    else:
        items = results

    if not items:
        return "No web search results were found."

    formatted_results = []

    for index, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "No URL")
        content = item.get("content", "")

        formatted_results.append(
            f"Result {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted_results)

tools =[rag_tool, web_search_tool]

#make the llm tool-aware
llm_tool_aware = llm.bind_tools(tools)