import base64
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS 
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from settings import (
    DOCUMENTS_FOLDER,
    INDEX_PATH,
    MAX_DIVERSE_SOURCES,
    MAX_DOCUMENT_CHARS,
    MAX_RETRIEVAL_DOCS,
    MAX_SOURCE_RETRIEVAL_DOCS,
    MAX_WEB_CONTENT_CHARS,
    MODEL_NAME,
    PDF_VISION_DPI,
    PDF_VISION_ENABLED,
    PDF_VISION_MAX_PAGES,
    PDF_VISION_PAGES,
    REBUILD_INDEX,
    SUPPORTED_IMAGE_EXTENSIONS,
    VISION_CACHE_PATH,
    VISION_MODEL,
)


llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.25,
    max_completion_tokens=1000,
    timeout=(10, 60),
    max_retries=3,
    stream_usage=True,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def load_pdf_text_documents(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def find_pdf_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {folder}")

    return sorted(folder.rglob("*.pdf"))


def find_image_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {folder}")

    return sorted(
        file_path
        for file_path in folder.rglob("*")
        if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def find_source_files(folder_path: str) -> list[Path]:
    files = [*find_pdf_files(folder_path), *find_image_files(folder_path)]

    if not files:
        raise FileNotFoundError(f"No PDF or image files found in documents folder: {folder_path}")

    return sorted(files)


def render_pdf_page_to_data_url(pdf_path: str, page_number: int) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF vision requires PyMuPDF. Install it with: uv add pymupdf"
        ) from exc

    with fitz.open(pdf_path) as pdf:
        page = pdf[page_number - 1]
        pixmap = page.get_pixmap(dpi=PDF_VISION_DPI)
        image_bytes = pixmap.tobytes("png")

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def image_file_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    extension = path.suffix.lower()
    mime_type = "image/png" if extension == ".png" else "image/jpeg"
    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def load_vision_cache() -> dict[str, str]:
    if not VISION_CACHE_PATH.exists():
        return {}

    return json.loads(VISION_CACHE_PATH.read_text(encoding="utf-8"))


def save_vision_cache(cache: dict[str, str]) -> None:
    VISION_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def describe_image_data_url(image_data_url: str, source_label: str) -> str:
    vision_llm = ChatOpenAI(
        model_name=VISION_MODEL,
        temperature=0,
        max_completion_tokens=900,
        timeout=(10, 90),
        max_retries=2,
    )

    response = vision_llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Analyze this image from {source_label} as a BPMN/business process reference. "
                            "If it contains diagrams, extract visible BPMN elements, labels, "
                            "events, tasks, gateways, pools, lanes, sequence flows, message flows, "
                            "data objects, and any modeling constraints. If it contains no useful "
                            "visual process information, say so briefly."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ]
            )
        ]
    )

    return response.content


def describe_pdf_page_image(pdf_path: str, page_number: int) -> str:
    image_data_url = render_pdf_page_to_data_url(pdf_path, page_number)
    return describe_image_data_url(image_data_url, f"{pdf_path}, page {page_number}")


def describe_uploaded_image(image_path: str) -> str:
    image_data_url = image_file_to_data_url(image_path)
    return describe_image_data_url(image_data_url, image_path)


def selected_vision_pages(page_count: int) -> list[int]:
    if not PDF_VISION_PAGES:
        return list(range(1, min(page_count, PDF_VISION_MAX_PAGES) + 1))

    selected_pages = set()

    for part in PDF_VISION_PAGES.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            selected_pages.update(range(int(start), int(end) + 1))
        else:
            selected_pages.add(int(part))

    return sorted(page for page in selected_pages if 1 <= page <= page_count)


def load_pdf_vision_documents(pdf_path: str) -> list[Document]:
    if not PDF_VISION_ENABLED:
        return []

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF vision is enabled but PyMuPDF is missing. Install it with: uv add pymupdf"
        ) from exc

    cache = load_vision_cache()
    vision_documents = []

    with fitz.open(pdf_path) as pdf:
        page_count = len(pdf)

    for page_number in selected_vision_pages(page_count):
        cache_key = f"{Path(pdf_path).resolve()}::page::{page_number}::dpi::{PDF_VISION_DPI}"
        description = cache.get(cache_key)

        if description is None:
            description = describe_pdf_page_image(pdf_path, page_number)
            cache[cache_key] = description
            save_vision_cache(cache)

        vision_documents.append(
            Document(
                page_content=description,
                metadata={
                    "source": pdf_path,
                    "page": page_number - 1,
                    "type": "vision_description",
                },
            )
        )

    return vision_documents


def load_image_vision_documents(image_path: str) -> list[Document]:
    cache = load_vision_cache()
    cache_key = f"{Path(image_path).resolve()}::image"
    description = cache.get(cache_key)

    if description is None:
        description = describe_uploaded_image(image_path)
        cache[cache_key] = description
        save_vision_cache(cache)

    return [
        Document(
            page_content=description,
            metadata={
                "source": image_path,
                "page": "image",
                "type": "image_vision_description",
            },
        )
    ]


def build_vectorstore() -> FAISS:
    all_docs = []

    for pdf_path in find_pdf_files(DOCUMENTS_FOLDER):
        pdf_path_text = str(pdf_path)
        all_docs.extend(load_pdf_text_documents(pdf_path_text))
        all_docs.extend(load_pdf_vision_documents(pdf_path_text))

    for image_path in find_image_files(DOCUMENTS_FOLDER):
        all_docs.extend(load_image_vision_documents(str(image_path)))

    if not all_docs:
        raise FileNotFoundError(f"No indexable PDF or image documents found in: {DOCUMENTS_FOLDER}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_docs)

    return FAISS.from_documents(chunks, embeddings)


if os.path.exists(INDEX_PATH) and not REBUILD_INDEX:
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    vectorstore = build_vectorstore()
    vectorstore.save_local(INDEX_PATH)


def make_retriever(store: FAISS):
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


retriever = make_retriever(vectorstore)


def rebuild_rag_index() -> list[str]:
    global vectorstore, retriever

    vectorstore = build_vectorstore()
    vectorstore.save_local(INDEX_PATH)
    retriever = make_retriever(vectorstore)

    return [file_path.name for file_path in find_source_files(DOCUMENTS_FOLDER)]


def indexed_document_names() -> list[str]:
    return [file_path.name for file_path in find_source_files(DOCUMENTS_FOLDER)]


def indexed_sources() -> list[str]:
    # FAISS exposes indexed documents through the in-memory docstore.
    documents = vectorstore.docstore._dict.values()
    return sorted(
        {
            document.metadata.get("source")
            for document in documents
            if document.metadata.get("source")
        }
    )


@traceable(name="faiss_retrieve", run_type="retriever")
def retrieve_relevant_documents(query: str) -> list[Document]:
    documents_by_key = {}

    def add_document(document: Document) -> None:
        key = (
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.metadata.get("type"),
            document.page_content[:120],
        )
        documents_by_key[key] = document

    for source in indexed_sources()[:MAX_DIVERSE_SOURCES]:
        source_documents = vectorstore.similarity_search(
            query,
            k=MAX_SOURCE_RETRIEVAL_DOCS,
            filter={"source": source},
        )

        for document in source_documents:
            add_document(document)

    for document in retriever.invoke(query):
        add_document(document)

    return list(documents_by_key.values())[:MAX_RETRIEVAL_DOCS]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."



@tool
def rag_tool(query:str) -> str:
    """
    Retrieve relevant BPMN reference documents from the vector store based on the query.
    """
    documents = retrieve_relevant_documents(query)
    if not documents:
        return "No relevant information was found in the document."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown Source")
        page_number = document.metadata.get("page", "Unknown Page")

        content = truncate_text(document.page_content, MAX_DOCUMENT_CHARS)

        formatted_documents.append(f"Document {index} (Source: {source}, Page: {page_number}):\n{content}")

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
    online = TavilySearch(max_results=3)
    results = online.invoke({"query": query})

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
        content = truncate_text(item.get("content", ""), MAX_WEB_CONTENT_CHARS)

        formatted_results.append(
            f"Result {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(formatted_results)

tools = [rag_tool, web_search_tool]
tool_node = ToolNode(tools)
