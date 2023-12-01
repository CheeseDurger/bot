from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentRetriever:
  __vectorstore_path: str = "vectorstore"

  def add_documents(self, documents_path: str) -> None:
    pdf_loader = PyPDFDirectoryLoader(documents_path)
    documents: List[Document] = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 3000,
      chunk_overlap  = 1500,
      length_function = len,
      add_start_index = True,
    )
    documents = text_splitter.split_documents(documents)
    vectorstore: FAISS = FAISS.from_documents(documents, OpenAIEmbeddings())
    vectorstore.save_local(self.__vectorstore_path)

  def retriever(self) -> VectorStoreRetriever:
    return FAISS.load_local(self.__vectorstore_path, OpenAIEmbeddings()).as_retriever()
