from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path
from typing import List, Union


def _get_openai_embeddings(model: str) -> OpenAIEmbeddings:
    """Internal helper to create an OpenAIEmbeddings object."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("[ERROR] OPENAI_API_KEY not set in environment.")
    return OpenAIEmbeddings(openai_api_key=api_key, model=model)


def load_documents(folder_path: str,
                   chunk_size: int = 1200,
                   chunk_overlap: int = 250) -> List[Document]:
    """
    Load and split documents into chunks for embedding.

    Args:
        folder_path (str or Path): Path to a folder containing documents.
        chunk_size (int): Max size of each text chunk.
        chunk_overlap (int): Overlap between chunks to preserve context.

    Returns:
        List[Document]: A list of document chunks ready for embedding.
    """

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"[ERROR] Folder not found: {folder_path}")

    all_docs = []

    # Recursively iterate over all files
    for file_path in folder.rglob("*"):  # rglob does recursive search
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path.resolve()))
            elif ext == ".txt":
                loader = TextLoader(str(file_path.resolve()))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path.resolve()))
            elif ext == ".csv":
                loader = CSVLoader(str(file_path.resolve()))
            else:
                continue  # skip unsupported files

            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"[Warning] Failed to load {file_path}: {e}")
            continue

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = splitter.split_documents(all_docs)

    print(f"[INFO] Total raw documents loaded: {len(all_docs)}")
    print(f"[INFO] Total document chunks created: {len(chunked_docs)}")
    return chunked_docs



def create_and_save_vectorstore(
    documents: List[Document],
    save_path: str = "vectorstore",  # path inside the Docker container
    embedding_model: str = "text-embedding-3-small"
 ) -> FAISS:
    """
    Create a FAISS vectorstore from documents and save locally inside Docker.

    Args:
        documents (List): List of split documents (from load_documents).
        save_path (str): Local directory to save the FAISS index inside the container.
        embedding_model (str): OpenAI embedding model to use.

    Returns:
        FAISS: The created vectorstore instance.

    Raises:
        FileNotFoundError: If save_path cannot be created.
    """
    if not documents:
        raise ValueError("[ERROR] No documents provided to create vectorstore.")

    # Ensure save_path exists or raise error
    save_path = Path(save_path)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise FileNotFoundError(f"[ERROR] Cannot create save_path: {save_path}. {str(e)}")

    embeddings = _get_openai_embeddings(embedding_model)

    # Build vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save locally
    vectorstore.save_local(save_path)
    print(f"[INFO] Vectorstore saved locally at: {save_path.resolve()}")

    return vectorstore



def load_vectorstore(save_path: str = "vectorstore",
                     embedding_model: str = "text-embedding-3-small") -> FAISS:
    """
    Load a FAISS vectorstore from local folder.

    Args:
        save_path (str): Path to the saved FAISS vectorstore folder.
        embedding_model (str): OpenAI embedding model to use.

    Returns:
        FAISS: Loaded vectorstore instance.

    Raises:
        FileNotFoundError: If save_path does not exist or is empty.
    """
    save_path = Path(save_path)
    if not save_path.exists() or not any(save_path.iterdir()):
        raise FileNotFoundError(f"[ERROR] Vectorstore folder not found or empty: {save_path.resolve()}")

    embeddings = _get_openai_embeddings(embedding_model)

    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[INFO] Vectorstore loaded from: {save_path.resolve()}")
    return vectorstore


# A more robust prompt template specifically for a medical context
MEDICAL_SYSTEM_PROMPT = """
You are an expert assistant specializing in providing information about breast cancer based on a given set of documents.
Your task is to answer user questions accurately using ONLY the information available in the provided context.

Follow these rules strictly:
1.  Base your entire answer on the context provided. Do not use any external knowledge.
2.  If the context does not contain the information needed to answer the question, you MUST state: "I cannot answer this question based on the provided information."
3.  Directly quote relevant parts of the context to support your answer where possible.
4.  After every answer, you MUST include the following disclaimer:
    "Disclaimer: This information is for informational purposes only and does not constitute medical advice. Please consult with a qualified healthcare professional for any medical concerns."
"""

def qa_chain(
    vectorstore: FAISS,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_k: int = 5):
    """
    Create a RetrievalQA chain using a pre-loaded FAISS vectorstore..

    Args:
        vectorstore (FAISS): Pre-loaded FAISS vectorstore.
        model_name (str): OpenAI chat model name.
        temperature (float): LLM creativity.
        top_k (int): Number of top relevant chunks to retrieve.
        chain_type (str): How to combine document chunks ("stuff", "map_reduce", "refine").

    Returns:
        RetrievalQA: Initialized QA chain ready to answer questions.

    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Use the safe, specialized medical prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", MEDICAL_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {input}\nAnswer:")
    ])

    # Create the combine documents chain
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Create QA chain
    qa = create_retrieval_chain(
    retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
    combine_docs_chain=combine_docs_chain
)

    return qa
