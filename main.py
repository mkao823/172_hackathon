from gensim.models import Word2Vec #pip install gensim
import sqlite3
import numpy as np
import re
import os
from pdfminer.high_level import extract_pages, extract_text #pip install pdfminer.six

text = extract_text("SOFI-2023.pdf")
#print(text)

os.makedirs("chunks", exist_ok = True)

chunk_size = 1000
#chunks = re.split(r'\n\s*\n', text)
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#train word2vec model
sentences = [chunk.split() for chunk in chunks]
model = Word2Vec(sentences, vector_size = 100, window=5, min_count=1, workers=4)

model.save("word2vec_model")

conn = sqlite3.connect("embeddings.db")
c = conn.cursor()

c.execute('''CREATE TABLE embeddings
                (id INTEGER PRIMARY KEY, chunk TEXT, embedding TEXT)''')

def get_embedding(chunk):
    words = chunk.split()
    embeddings = [model[word] for word in words if word in model]
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    else:
        return None

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    if(embedding):
        c.execute("INSERT INTO embeddings (id, chunk, embedding) VALUES (?, ?, ?)", (i, chunk, str(embedding)))
    #with open(f"chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
    #    f.write(chunk)
conn.commit()
conn.close()