from langchain.document_loaders import DirectoryLoader, PyPDFLoader

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