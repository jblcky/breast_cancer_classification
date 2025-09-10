from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.schema import Document
import os
from pathlib import Path
from typing import List, Union


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

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("[ERROR] OPENAI_API_KEY not set in environment.")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model=embedding_model)

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

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("[ERROR] OPENAI_API_KEY not set in environment.")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model=embedding_model)
    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

    return vectorstore



def qa_chain(
    vectorstore: FAISS,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    top_k: int = 5,
    chain_type: str = "refine",  # alternatives: "map_reduce", "refine"
    system_prompt: str = None ) -> RetrievalQA:
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
    if system_prompt:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature,
                         model_kwargs={"system_message": system_prompt})

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        chain_type=chain_type
    )

    return qa
