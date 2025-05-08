from src.helper import load_pdf_doc, text_split, download_hugging_face_embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = "information-security-bot"

extracted_data = load_pdf_doc("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding_model()

if index not in pc.list_indexes().names():
    pc.create_index(
        name=index,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
else:
    print(f"Index '{index}' already exists. Skipping creation.")

docsearch = PineconeVectorStore.from_documents(
    text_chunks,
    embeddings,
    index_name=index,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)





