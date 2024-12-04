from flask import Flask, request, jsonify, render_template
from groq import Groq
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import uuid
import os

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# MongoDB configuration
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["medical_chatbot"]
collection = db["chat_history"]

# Pinecone and Groq API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
index_name = "medical-chatbot"
from pinecone import Pinecone
pinecone_instance = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pinecone_instance.list_indexes().names():
    pinecone_instance.create_index(
        name=index_name,
        dimension=384,
        metric='cosine'
    )

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="gemma-7b-it",
    temperature=1,
    max_tokens=1024,
    verbose=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{history}\nUser: {input}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=True
)

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Route: Home
@app.route("/")
def home():
    return render_template("index.html")  # Serve a simple HTML template

# Route: Chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("user_input")
    context = data.get("document_content", "")

    # Update memory with user input
    memory.chat_memory.add_user_message(user_input)

    # Generate AI response
    response = conversation_chain({"input": user_input, "context": context})
    bot_response = response["text"]

    # Update memory with bot response
    memory.chat_memory.add_ai_message(bot_response)

    # Save to MongoDB
    session_id = str(uuid.uuid4())
    collection.insert_one({
        "session_id": session_id,
        "user_input": user_input,
        "bot_response": bot_response
    })

    return jsonify({"bot_response": bot_response})

# Route: PDF Upload
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Extract text from the uploaded PDF
        pdf_content = extract_text_from_pdf(file_path)
        return jsonify({"pdf_content": pdf_content})

    return jsonify({"error": "Failed to upload file"}), 500

if __name__ == "__main__":
    app.run(debug=True)
