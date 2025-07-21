from src.helper import load_pdf, text_split, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os 
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

extracted_pdf=load_pdf("data/")
chunks=text_split(extracted_pdf)

embedding = download_huggingface_embeddings()

if 'medicalchatbot' not in pc.list_indexes().names():
        pc.create_index(
            name='medicalchatbot',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
# index_name = "medicalchatbot"

# # Create vector store from documents
# vector_store = PineconeVectorStore.from_documents(
#     documents=chunks,
#     embedding=embedding,
#     index_name=index_name
# )

index_name='medicalchatbot'
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embedding)