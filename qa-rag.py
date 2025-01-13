import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

# Load Data
@st.cache
def load_data(file_path):
    return pd.read_csv(r"C:\\Users\\ayush\\Downloads\\qa.csv")

class Retriever:
    def __init__(self, data):
        self.data = data
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  
        self.index = self.create_index(data['Questions'])

    def create_index(self, questions):
        embeddings = self.embed_questions(questions)
        self.nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(embeddings)
        return embeddings

    def embed_questions(self, questions):
        # Use the pre-trained model to generate embeddings
        embeddings = self.model.encode(questions, convert_to_tensor=True)
        return embeddings

    def retrieve(self, query, k=5):
        query_embedding = self.embed_questions([query])
        distances, indices = self.nn.kneighbors(query_embedding, n_neighbors=k)
        return self.data.iloc[indices[0]]

# Generator Class
class Generator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate(self, prompt, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=150, 
            temperature=temperature,
            no_repeat_ngram_size=1,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG Pipeline Class
class RAGPipeline:
    def __init__(self, data):
        self.retriever = Retriever(data)
        self.generator = Generator()

    def ask(self, query):
        context = self.retriever.retrieve(query)
        prompt = f"Context: {context['Answers'].iloc[0]}\nQuestion: {query}\nAnswer:"
        response = self.generator.generate(prompt)
        return response

# Streamlit Interface
st.title("Interview Chatbot")

data_file = r"C:\\Users\\ayush\\Downloads\\qa.csv"
data = load_data(data_file)
pipeline = RAGPipeline(data)

st.subheader("Ask the Interview Bot")
query = st.text_input("Your question:", "")

if query:
    response = pipeline.ask(query)
    st.write(f"NAMMO: {response}")