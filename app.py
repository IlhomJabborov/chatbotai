

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import Docx2txtLoader
import openai

api = "API-KEY"
openai.api_key = api

loader = Docx2txtLoader("./data.docx") 
pages = loader.load_and_split() 

embedding = OpenAIEmbeddings(openai_api_key=api)
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=api)

vectordb = Chroma.from_documents(pages, embedding)

promp_main = """Do not answer questions that are not related to credit and banking,
answer the questions completely, the text of the answer should not exceed 10 sentences, 
In whatever language the question is asked, give the answer in that language,
"

{context}
Question: {question}
Helpful Answer:"""
temp_promp = PromptTemplate(
   input_variables=["context", "question"],
   template=promp_main
)

full_chain = RetrievalQA.from_chain_type(llm, 
                                          retriever=vectordb.as_retriever(), 
                                          return_source_documents=False,
                                       chain_type_kwargs={"prompt": temp_promp})

def my_func(question, history=False):
   return str(full_chain({"query": question})['result'])

# question = "hayotda oq otlar bormi"
# my_func(question)

question = st.chat_input("Savolingizni yozing:")
if question:
    with st.chat_message("user"):
        st.markdown(question)
        
    with st.chat_message("assistant"):
        st.markdown(my_func(question))
