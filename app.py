import os
import langchain_google_genai
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

# Load API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")

PROMPT = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

# Function to clean text (removes emojis and special characters)
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters

# Initialize session state variables if they don't exist
if "vectors" not in st.session_state:
    st.session_state.vectors = None  
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  

# Vector embedding function
def vector_embedding():
    if st.session_state.vectors is None:  # Prevent re-initialization
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("pdfs")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Clean document text before embedding
        for doc in st.session_state.final_documents:
            doc.page_content = clean_text(doc.page_content)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector store successfully created!")

# Input field
prompt1 = st.text_input("Enter what you want to ask from the documents?")

if st.button("Creating the vector store"):
    vector_embedding()
    st.write("Vector store DB is Ready")

import time

if prompt1:
    if st.session_state.vectors is not None:
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        start = time.process_time()
        response = retriever_chain.invoke({'query': prompt1})
        st.write(response.get('answer', 'No answer found.'))

        # Document Similarity Search
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write("-------------------------")
    else:
        st.warning("Please create the vector store first by clicking the 'Creating the vector store' button.")
