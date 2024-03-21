import os
import time
import pickle
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv() # take environment variables from .env

llm = OpenAI(temperature = 0.9, max_tokens=500)

st.title("News Research Tool")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Search")

main_placeholder = st.empty()
vectorstore_openai = None

if process_url_clicked:
    # load data
    main_placeholder.text("Loading data...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # split data
    main_placeholder.text("Splitting data...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )

    docs = text_splitter.split_documents(data)

    # create embeddings and save to FAISS index
    main_placeholder.text("Creating embeddings and saving to FAISS index...")
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    # save FAISS into 


query = main_placeholder.text_input("Question:")
if query:
    if vectorstore_openai:
        # create chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
        result = chain({'question': query}, return_only_outputs=True) # this will be a dictionary with two elements
        # {"answer": "...", "sources": ["...", "..."]}
        st.header("Answer:")
        st.write(result["answer"])

        # display sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)