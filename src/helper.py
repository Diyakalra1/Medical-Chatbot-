from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents

def text_split(extracted_data):
  splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
  text_chunks=splitter.split_documents(extracted_data)
  return text_chunks

#download the hugging face package
def download_huggingface_embeddings():
  
  model_ = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  
  embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embedding