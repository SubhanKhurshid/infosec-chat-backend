from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from flask_cors import CORS
import os

load_dotenv()





app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embedding_model()

index_name = "information-security-bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=800,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)


@app.route('/')
def index():
    return jsonify({"message": "InfoSec Chat API is running"})
    

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message']
    print("User Query:", user_message)
    
    try:
        response = rag_chain.invoke({"input": user_message})
        print("Response:", response['answer'])
        return jsonify({"response": response['answer']})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)