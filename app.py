
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

import shutil
import os

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def main():
    generate()

def generate():
    document = load_documents()
    # Split text
    chunks = split_text(document)
    save_to_chroma(chunks)

def load_documents():
    loader = PyPDFLoader("data/SOFI-2023.pdf")
    documents = loader.load_and_split()
    return documents

def split_text(documents: "list[Document]"):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)
    #print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    #print(document)
    #print(document.metadata)

    return chunks

def save_to_chroma(chunks: "list[Document]"):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    
    # Process PDF file and run main function
    main()
    
    # Print or use the chunks as needed
    #for chunk in chunks: