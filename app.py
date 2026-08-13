import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.conversation import handle_conversation
from src.generator import generate_answer
from src.helper import download_huggingface_embeddings
from src.pipeline import retrieve_answers
from src.query.query_router import QueryRouter
from src.retrieval.context_evaluator import ContextEvaluator
from src.retrieval.reranker import MedicalReranker
from src.retrieval.retriever import MedicalRetriever


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

MODEL_NAME = "gemini-3-flash-preview"

medical_retriever = None
medical_reranker = None
context_evaluator = None

models_ready = False
model_loading = False
model_error = None

model_lock = threading.Lock()

client = genai.Client(api_key=GEMINI_API_KEY)
query_router = QueryRouter()


def initialize_medical_pipeline():
    global medical_retriever
    global medical_reranker
    global context_evaluator
    global models_ready
    global model_loading
    global model_error

    with model_lock:
        if models_ready or model_loading:
            return

        model_loading = True

    print("\n--- MODEL INITIALIZATION ---")
    print("Starting MedAssist pipeline warm-up...")

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        index = pc.Index("medicalchatbot")
        embedding = download_huggingface_embeddings()
        vectorstore = PineconeVectorStore(index=index, embedding=embedding)

        medical_retriever = MedicalRetriever(vectorstore)
        medical_reranker = MedicalReranker()
        context_evaluator = ContextEvaluator()

        models_ready = True
        model_error = None

        print("MedAssist pipeline READY")

    except Exception as error:
        model_error = str(error)
        print("MODEL INITIALIZATION ERROR:", error)

    finally:
        model_loading = False


def start_model_warmup():
    threading.Thread(
        target=initialize_medical_pipeline,
        daemon=True,
    ).start()


def build_high_risk_response():
    return """
This may describe a potentially serious medical situation.

Please seek immediate medical attention or contact local emergency
services. Do not rely on MedAssist for urgent medical situations.

If possible, ask someone nearby for assistance and seek professional
medical care immediately.
"""


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/ready")
def ready():
    if models_ready:
        return jsonify({
            "ready": True,
            "status": "READY"
        })

    if model_error:
        return jsonify({
            "ready": False,
            "status": "ERROR",
            "error": model_error
        }), 500

    return jsonify({
        "ready": False,
        "status": "LOADING"
    })


@app.route("/get", methods=["POST"])
def chat():
    message = request.form.get("msg", "").strip()

    if not message:
        return jsonify({
            "answer": "Please enter a health-related question.",
            "trace": None,
        })

    print("Input:", message)

    try:
        result = retrieve_answers(
            query=message,
            query_router=query_router,
            medical_retriever=medical_retriever,
            medical_reranker=medical_reranker,
            context_evaluator=context_evaluator,
            models_ready=models_ready,
            handle_conversation=handle_conversation,
            build_high_risk_response=build_high_risk_response,
            generate_answer=generate_answer,
            client=client,
            model_name=MODEL_NAME,
        )

        print("Response:", result["answer"])
        return jsonify(result)

    except Exception as error:
        print("Error:", error)

        return jsonify({
            "answer": (
                "MedAssist encountered an error while "
                "processing your query."
            ),
            "trace": {
                "decision": "PIPELINE_ERROR"
            }
        }), 500


start_model_warmup()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

# Debug = True is used to get detialed stack traces and debug support during the development
# use_reloader is set false to ensure ML pipeline is initialised only once, the reloader may start background threads for reloading the embeddings re ranker models which could lead to heavy memeory usage and intialisation of the ML pipleline twice 
