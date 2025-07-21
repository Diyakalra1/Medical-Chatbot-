from langchain_core.prompts import PromptTemplate
template="""
use the following peice of information to answer the user's question. If you do not know the answer just simply say that you do not know the answer , do not try to make up an answer.
Context:{context}
Question={question}
Only give the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=template,input_variables=["context","question"])