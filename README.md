## MedAssist-Medical information copilot
Evidence Aware Medical infomaation co-pilot built to provide grounded Meical information.
<img width="1727" height="872" alt="image" src="https://github.com/user-attachments/assets/643ca472-b7f3-4d86-98af-da9021e43c7c" />
## Evidence Trace 
<img width="588" height="563" alt="image" src="https://github.com/user-attachments/assets/48a17611-a535-4ef9-bf80-6e246909d319" />


##  Architecture

### One time process 
```text
                          Medical PDF
                              │
                              ▼
                           Load PDF
                              │
                              ▼
                        Split into Chunks
                        (chunk_size = 500,
                        chunk_overlap = 20)
                              │
                              ▼
                        Generate Embeddings
                        (HuggingFace)
                              │
                              ▼
                        Store in Pinecone Index
                        (Index: medicalchatbot)

```
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
     (if conversation        (if high Risk              ( else )             
       pattern match)         patter match)               |
          |                       |                       |
          ▼                       ▼                       ▼
  Conversation Route      High-Risk Route          Medical Pipeline
          │______________________|                        │
                      |                                   |
                      ▼                                   ▼ 
       ( Direct Response, LLM call saved )           ┌────────────────────────────┐
                      |                               │ Retrieve Top-10 Chunks     |
                      |                               │      (From Vector DB)      |
                      |                               | (Based on similairty score)|
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
## Technology used & other specifications
### Knowledge Base -  The Gale Encyclopedia of Medicine (Second Edition) -637 pages 
### Gemini Model - gemini-3-flash-preview
### Vector DataBase - PineCone 
### similarity metric cosine similairty
### Embedding model - sentence-transformers/all-MiniLM-L6-v2
### cross encoder used - cross-encoder/ms-marco-MiniLM-L6-v2



### key features
#### Reduced the total pipeline lantency by 51.6% benchmarked against 100 queries.

## Components 
Query Router- specifiess three intenest conversation, high risk, medical   CONVERSATION
MEDICAL
HIGH_RISK
##  YouTube Demo

 **Demo Video:** https://youtu.be/1YD7Zs4BcCA
