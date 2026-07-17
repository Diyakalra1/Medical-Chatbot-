##  Architecture



```text
                           User Question
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │    Query Router     │
                      └──────────┬──────────┘
                                 │
         ┌───────────────────────┼────────────────────────┐
         │                       │                        │
         ▼                       ▼                        ▼
  Conversation Route      High-Risk Route         Medical Pipeline
                                                        │
                                                        ▼
                                          ┌────────────────────────┐
                                          │ Retrieve Top-10 Chunks │
                                          └──────────┬─────────────┘
                                                     │
                                                     ▼
                                         ┌──────────────────────────┐
                                         │ CrossEncoder Reranker    │
                                         │ Select Top-3 Evidence    │
                                         └──────────┬───────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────────┐
                                         │  Evidence Evaluation     │
                                         │ Is the context enough?   │
                                         └───────┬─────────┬────────┘
                                                 │         │
                                       No        │         │ Yes
                                                 │         │
                                                 ▼         ▼
                                      Safe Abstention   Model (gemini-3-flash-preview)
                                                        Generation
                                                              │
                                                              ▼
                                            Final Response + Evidence Trace
                                                        Generation
                                                              │
                                                              ▼
                                            Final Response + Evidence Trace
```


##  YouTube Demo



 **Demo Video:** https://youtu.be/1YD7Zs4BcCA
