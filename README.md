# MedAssist — Evidence-Aware Medical Information Copilot
## 1. Project Overview
**MedAssist** is an **evidence-aware medical information copilot** built to provide **grounded medical responses**. It retrieves information from the medical knowledge base and displays an **evidence trace** showing how each response is supported, improving **transparency and trust**.


## 2. Key Features

* **Evidence Trace:** Displays the supporting medical evidence for every generated response to improve **transparency and trust**.

* **Low-Latency Pipeline:** Reduced **end-to-end pipeline latency by 51.6%** compared to a baseline RAG pipeline, validated on a **100-query benchmark**.

* **Domain-Agnostic Architecture:** Easily adaptable to any knowledge base by **generating embeddings for a new corpus and indexing them in a vector database**, without modifying the core pipeline.


## 3. Tech Stack & Specifications

* **Knowledge Base:** *The Gale Encyclopedia of Medicine (Second Edition)* — **637 pages**
* **Backend Framework:** **Flask** — lightweight and flexible backend for serving the RAG pipeline
* **LLM:** **gemini-3-flash-preview**
* **Vector Database:** **Pinecone**
* **Embedding Model:** **sentence-transformers/all-MiniLM-L6-v2**
* **CrossEncoder Reranker:** **cross-encoder/ms-marco-MiniLM-L6-v2**
* **Similarity Metric:** **Cosine Similarity**
* **Architecture:** **Evidence-aware Retrieval-Augmented Generation (RAG)**

## 4. Architecture

### One time process 
```text
 Knowledge Base --> Load PDF--> Split into Chunks  ---> Generate Embeddings ---> Store in Pinecone Index
 (637 Pages PDF)                (chunk_size = 500,     (dimensions = 384)      (Index: medicalchatbot)
                                chunk_overlap = 20)
                      

```

### Flow of Medical Pipeline
```text
                           User Query
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │    Query Router     |
                      |                     |
                      └──────────┬──────────┘
                                 │
         ┌───────────────────────┼────────────────────────┐
     (if conversation        (if High Risk              ( else )             
       pattern match)         pattern match)               |
          |                       |                       |
          ▼                       ▼                       ▼
  Conversation Route      High-Risk Route          Medical Pipeline
          │______________________|                        │
                      |                                   |
                      ▼                                   ▼ 
       ( Direct Response, LLM call saved )           ┌────────────────────────────┐
                      |                               │ Retrieve Top-10 Chunks     |
                      |                               │      (From Vector DB)      |
                      |                               | (Based on similarity score)|
                      |                               └──────────┬─────────────────┘
                      |                                          │
                      |                                          ▼
                      |                   ┌──────────────────────────────────┐
                      |                   │     CrossEncoder Reranker        │
                      |                   │   Select Top-3 Relevant chunks   |
                      |                   │(Query, Document)-> Relevant Score|
                      |                   └──────────────┬───────────────────┘
                      |                                  │
                      |                                  ▼
                      |                   ┌──────────────────────────┐
                      |                   │  Context  Evaluator      │
                      |                   │ Is the context enough?   │
                      |                   └───────┬─────────┬────────┘
                      |                           │         │
                      |                 No        │         │ Yes
                      |                           │         │______________
                      |                           ▼                        |
                      |                                                    ▼
                      |                Safe Abstention            Model (gemini-3-flash-preview)
                      |                           |                       Generation
                      |                           ▼                          │
                      |                  ( Direct Response                   |
                      |                 , LLM call saved )                   |
                      |                            |_________________________|
                      |                                        │
                      |                                        ▼
                      |----------------------------->  Final Response + Evidence Trace
```



## 5. Project Structure

```text
MedAssist/
│
├── app.py                         # Flask application entry point
├── store_index.py                 # Builds and stores document embeddings in Pinecone
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/
│   └── Medical_book.pdf           # Source medical knowledge base (Gale Encyclopedia)
│
├── src/
│   ├── pipeline.py                # Main RAG orchestration pipeline
│   ├── generator.py               # Gemini grounded response generation
│   ├── conversation.py            # Handles conversational and greeting queries
│   ├── helper.py                  # Embedding model loading and utility functions
│   │
│   ├── query/
│   │   └── query_router.py        # Intent classification and semantic routing
│   │
│   └── retrieval/
│       ├── retriever.py           # Pinecone vector retrieval logic
│       ├── reranker.py            # CrossEncoder reranking implementation
│       └── context_evaluator.py   # Evidence sufficiency and abstention logic
│
├── templates/
│   └── chat.html                  # Frontend chat interface
│
├── static/
│   ├── style.css                  # UI styling
│   └── images/
│       └── medassist-logo.png     # Application logo
│
└── evaluation/
    ├── test_queries.py            # 100-query benchmark dataset
    ├── evaluate_pipeline.py       # Benchmark execution script
    ├── 100_query_results.csv      # Per-query evaluation results
    └── 100_query_benchmark_report.txt
                                     # Final benchmark summary report
```
## 6. Screenshots & Demo

### Main Chat Interface

<p align="center">
  <img width="900" alt="MedAssist Main UI" src="https://github.com/user-attachments/assets/643ca472-b7f3-4d86-98af-da9021e43c7c" />
</p>

---

### Evidence Trace
<p align="center">
  <img width="520" alt="Evidence Trace" src="https://github.com/user-attachments/assets/48a17611-a535-4ef9-bf80-6e246909d319" />
</p>

---

### Demo Video

🎥 **Watch the demo:** https://youtu.be/1YD7Zs4BcCA
--- 


## 7. API Documentation

#### Base URL
For local development:
```text
http://127.0.0.1:5000
```
### API Endpoints
#### 1. Home Page
#### `GET /`

Renders the MedAssist web chat interface (`chat.html`).

#### Response

* **Content-Type:** `text/html`
* **Status:** `200 OK`

#### Example

```bash
curl http://127.0.0.1:5000/
```

---

#### 2. Readiness Check

#### `GET /ready`

Returns the current status of the medical pipeline warm-up process.

This endpoint is useful for:

* frontend startup checks,
* deployment health checks,
* monitoring model initialization.

#### Success Response

```json
{
  "ready": true,
  "status": "READY"
}
```

#### Loading Response

```json
{
  "ready": false,
  "status": "LOADING"
}
```

#### Error Response

```json
{
  "ready": false,
  "status": "ERROR",
  "error": "Pinecone authentication failed"
}
```

#### Example

```bash
curl http://127.0.0.1:5000/ready
```

#### 3. Chat Endpoint

#### `POST /get`

Processes a user query through the complete MedAssist pipeline.

#### Request

#### Headers

```http
Content-Type: application/x-www-form-urlencoded
```

#### Form Parameters

| Parameter | Type   | Required | Description        |
| --------- | ------ | -------- | ------------------ |
| `msg`     | string | Yes      | User medical query |

#### Example Request

```bash
curl -X POST http://127.0.0.1:5000/get \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "msg=What are the symptoms of diabetes?"
```

---

#### Response Format

#### Successful Response

```json
{
  "answer": "Diabetes is a chronic condition in which the body has difficulty regulating blood sugar levels...",
  "trace": {
    "intent": "MEDICAL",
    "routing_method": "evidence_pipeline",
    "routing_ms": 0.21,
    "pipeline_path": "EVIDENCE_PIPELINE",
    "documents_retrieved": 10,
    "retrieval_ms": 334.53,
    "reranking_ms": 173.60,
    "evidence_score": 0.84,
    "decision": "GENERATE",
    "grounding_decision": "SUPPORTED",
    "total_evidence_latency_ms": 508.13
  }
}
```

#### Trace Object Reference

The `trace` object provides transparency into every stage of the RAG pipeline.

| Field                       | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `intent`                    | Query intent classified by the router                 |
| `routing_method`            | Strategy used for routing                             |
| `routing_ms`                | Router execution latency                              |
| `pipeline_path`             | Pipeline branch executed                              |
| `documents_retrieved`       | Number of retrieved candidates                        |
| `retrieval_ms`              | Vector retrieval latency                              |
| `reranking_ms`              | CrossEncoder reranking latency                        |
| `evidence_score`            | Context evaluator confidence score                    |
| `decision`                  | `GENERATE`, `ABSTAIN`, `CONVERSATION`, or `HIGH_RISK` |
| `grounding_decision`        | Whether the retrieved evidence supported the response |
| `total_evidence_latency_ms` | Retrieval + reranking latency                         |

---

#### Pipeline Paths

#### Conversation Path
Triggered for greetings and general conversational queries.

#### Example

#### Request

```json
{ "msg": "Hello" }
```

#### Response

```json
{
  "answer": "Hello! I'm MedAssist, your medical information copilot.",
  "trace": {
    "intent": "CONVERSATION",
    "pipeline_path": "CONVERSATION",
    "decision": "CONVERSATION"
  }
}
```

#### High-Risk Path

Triggered for emergency-style queries.

#### Example

#### Request

```json
{ "msg": "I have severe chest pain and cannot breathe" }
```

#### Response

```json
{
  "answer": "This may describe a potentially serious medical situation. Please seek immediate medical attention...",
  "trace": {
    "intent": "HIGH_RISK",
    "pipeline_path": "HIGH_RISK",
    "decision": "HIGH_RISK"
  }
}
```
#### Evidence Pipeline Path

Triggered for medical information requests.

#### Stages

1. **Intent Classification**
2. **Vector Retrieval**
3. **CrossEncoder Reranking**
4. **Evidence Sufficiency Evaluation**
5. **Grounded LLM Generation**


## 8. Benchmark & Evaluation

MedAssist was evaluated using a **100-query benchmark** covering:

| Category                |           Accuracy |
| ----------------------- | -----------------: |
| **Supported Medical**   |  **84.0% (42/50)** |
| **Unsupported**         | **100.0% (15/15)** |
| **Unsupported Medical** |   **70.0% (7/10)** |
| **Conversation**        |   **70.0% (7/10)** |
| **High-Risk**           |  **46.67% (7/15)** |

---
### Benchmark Results (100 Queries)

| Metric                 |    Result |
| ---------------------- | --------: |
| **Total Queries**      |   **100** |
| **Decision Agreement** | **78.0%** |
| **LLM Calls Saved**    | **52.0%** |
| **Generate Rate**      | **48.0%** |
| **Abstain Rate**       | **38.0%** |
| **Conversation Rate**  |  **7.0%** |
| **High-Risk Rate**     |  **7.0%** |
| **Promotion Rate**     | **56.0%** |
---

### Latency Metrics

| Metric             |       Average |       P50 |       P95 |
| ------------------ | ------------: | --------: | --------: |
| **Retrieval**      | **334.53 ms** | 359.05 ms | 434.87 ms |
| **Reranking**      | **173.60 ms** | 191.74 ms | 260.91 ms |
| **Total Pipeline** | **508.36 ms** | 563.42 ms | 648.95 ms |

---

### Evaluation Artifacts

The benchmark generates the following artifacts:
```text
evaluation/
├── test_queries.py 
├── evaluate_pipeline.py
├── 100_query_results.csv
└── 100_query_benchmark_report.txt
```

* **100_query_results.csv** — per-query routing, retrieval, reranking, and latency traces.
* **100_query_benchmark_report.txt** — aggregated benchmark statistics and latency summaries.

---

### Key Takeaways

* **52% of potential LLM calls were avoided** through query routing and evidence-aware context evaluation.
* The complete **retrieval + reranking + generation pipeline averaged ~508 ms** end-to-end.
* **95% of requests completed in under 649 ms**, enabling responsive real-time interactions.
* The **CrossEncoder reranker improved evidence quality**, promoting better supporting documents in **56% of evaluated queries**.
* The system demonstrated strong reliability for **supported medical queries (84%)** while correctly abstaining on unsupported or insufficiently grounded queries.
---



