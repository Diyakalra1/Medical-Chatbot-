from flask import Flask,render_template,jsonify,request
from src.helper import *
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from ctransformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os
from src.prompt import *
from pinecone import Pinecone, ServerlessSpec


app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name='medicalchatbot'
index = pc.Index(index_name)
embedding=download_huggingface_embeddings()
vectorstore = PineconeVectorStore(index=index, embedding=embedding)

from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained(
#     model_path_or_repo_id="Model/llama-2-7b-chat.Q3_K_S.gguf",  # path to your .gguf file
#     model_type="llama",  # required
     
#     max_new_tokens= 512,
#     temperature= 0.8
    
# )

from langchain_community.llms import CTransformers
llm = CTransformers(
    model="Model/llama-2-7b-chat.Q3_K_S.gguf",
    model_type="llama",  # or "mistral", "gpt2", etc.
    config={"max_new_tokens": 200, "temperature": 0.5}
)

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
def retrieve_query(query,k=2):
  matching_results=vectorstore.similarity_search(query,k=k)
  return matching_results

from langchain.chains.combine_documents import create_stuff_documents_chain
chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

def retrieve_answers(query):
    doc_search = retrieve_query(query)  # retrieves relevant documents
    response = chain.invoke({"context": doc_search, "question": query})

    return response

chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

@app.route('/')
def index():
  return render_template('chat.html')
@app.route('/get',methods=['GET',"POST"])
def chat():
  msg=request.form['msg']
  input=msg
  print('Input:',input)
  result=retrieve_answers(input)
  print('Response',result)
  return result

if __name__=='__main__':
  app.run(debug=True)

