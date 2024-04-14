from flask import Flask, request, render_template
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import shutil

app = Flask(__name__)

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def load_documents():
    loader = PyPDFLoader("data/SOFI-2023.pdf")
    documents = loader.load_and_split()
    return documents

def split_text(documents: "list[Document]"):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: "list[Document]"):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def query_gpt3(prompt):
    api_key = os.getenv('your api key')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': 'text-davinci-002',
        'prompt': prompt,
        'max_tokens': 150
    }
    response = requests.post('https://api.openai.com/v1/completions', json=data, headers=headers)
    try:
        return response.json()['choices'][0]['text'].strip()
    except KeyError:
        return "Failed to get an answer. Please check your inputs and try again."

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        if question:
            answer = query_gpt3(question)
    return render_template('index.html', question=request.form.get('question', ''), answer=answer)

if __name__ == "__main__":
    # Option to run Flask app or process documents
    if os.getenv('FLASK_RUN'):
        app.run(debug=True)
    else:
        # Process PDF file and store it in Chroma
        documents = load_documents()
        chunks = split_text(documents)
        save_to_chroma(chunks)
