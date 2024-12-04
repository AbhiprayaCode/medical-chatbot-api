import os
import json
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

# Set up Pinecone configuration
api_key = os.environ.get("PINECONE_API_KEY")
index_name = "medical-chatbot"
environment = "us-east-1"

pinecone.init(api_key=api_key, environment=environment)

# Define the Hugging Face embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Extract data from the PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Extract data from the CSV files
def load_csv_file(data):
    loader = DirectoryLoader(data, glob="*.csv", loader_cls=lambda file_path: CSVLoader(file_path, encoding='utf-8'))
    documents = loader.load()
    return documents

# Split the Data in Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Store the index in Pinecone
def embed_store_index(chunks, embeddings, index_name):
    index = Pinecone(
        index_name=index_name,
        embeddings=embeddings
    )
    index.add_documents(chunks)
