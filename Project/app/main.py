import streamlit as st
import pickle
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together

#Adding api key for LLM call
os.environ["TOGETHER_API_KEY"] = st.secrets["together"]["api_key"]

file_path="faiss_vector_db.pkl"

llm=Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.6,
    max_tokens=512
)

st.title("Articles Research Tool 🤖")

st.sidebar.title("Artcle URLs")

urls=[]

for i in range(3):
    url=st.sidebar.text_input(f"Paste URL {i+1}")
    if url.strip():  # only add if not empty
        urls.append(url)
   
process_url_clicked=st.sidebar.button("Process URLs")

main_placeholder=st.empty()

if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started ...")
    data=loader.load()
    #split data
    
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."],
        chunk_size=600,
        chunk_overlap=50
    )
    main_placeholder.text("Text Splitting Started ...")
    docs=splitter.split_documents(data)
    
    #Create embeddings and save to FAISS index
    
    embedder = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    main_placeholder.text("Embedding and Vector DB Started Building ...")
    vector_db=FAISS.from_documents(docs,embedding=embedder)
    
    with open(file_path,"wb") as f:
        pickle.dump(vector_db,f) 
    
query=main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vector_db=pickle.load(f)
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=retriever)
        result=chain.invoke({'question':query})
        # Result contains answer and source(metadata)
        st.header("Answer for your question: ")
        st.write(result["answer"])
        
        sources=result.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            source_list=sources.split("\n")
            for source in source_list:
                st.write(source)
    
      