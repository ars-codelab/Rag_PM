# Add this code to the top of your app.py file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain.retrievers import EnsembleRetriever 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import subprocess # Add this import
import requests

# --- Page Setup ---

st.set_page_config(page_title="CSR Assistant", page_icon="💬")
st.title("Customer Service Representative Assistant")


# --- RAG Pipeline Setup ---

@st.cache_resource
def load_rag_pipeline():
    # --- Fix 1: Handle API Key with Streamlit Secrets ---
    # In your local project, create a file .streamlit/secrets.toml
    # and add your key like this: GOOGLE_API_KEY = "your_api_key_here"
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # --- Fix 2: Initialize the Language Model ---
    generation_llm = ChatGoogleGenerativeAI(model="models/models/gemini-2.0-flash-lite", temperature=0.3)
    
    # --- Suggestion 1: Removed leading spaces from filenames ---
    github_files = {
    "CRM_Pro_Technical_documentation.pdf": "https://raw.githubusercontent.com/ars-codelab/Rag_PM/main/Files/CRM_Pro_Technical_documentation.pdf",
        "CRM_Pro_User_guide.pdf": "https://raw.githubusercontent.com/ars-codelab/Rag_PM/main/Files/CRM_Pro_User_guide.pdf",
        "CRM_Pro_Billing_Information.pdf": "https://raw.githubusercontent.com/ars-codelab/Rag_PM/main/Files/CRM_Pro_Billing_Information.pdf",
        "CRM_Pro_Customer_Support_Guide.pdf": "https://raw.githubusercontent.com/ars-codelab/Rag_PM/main/Files/CRM_Pro_Customer_Support_Guide.pdf",
        "CRM_Pro_Customer_Support_Upsell_guide.pdf": "https://raw.githubusercontent.com/ars-codelab/Rag_PM/main/Files/CRM_Pro_Customer_Support_Upsell_guide.pdf"  
    }

    #-- Start of Corrected Section ---
    # Download each file using the 'requests' library
    for filename, url in github_files.items():
        if not os.path.exists(filename):
            try:
                print(f"Downloading {filename} from GitHub...")
                response = requests.get(url, stream=True)
                # Raise an exception if the download failed
                response.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download {filename}: {e}")
                st.stop()
    # --- End of Corrected Section ---

    # Load the Downloaded PDF Documents
    pdf_files = list(github_files.keys())
    all_docs = []
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata['source'] = os.path.basename(pdf_path)
            all_docs.extend(documents)

    # Chunk the Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(all_docs)

    # Embedding and Vector Storage
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embedding_model)

    # Setup the Hybrid RAG pipeline
    doc_texts = [doc.page_content for doc in chunked_docs]
    
    # --- Fix 3: Correctly configure BM25Retriever ---
    bm25_retriever = BM25Retriever.from_texts(
        texts=doc_texts,
        metadatas=[doc.metadata for doc in chunked_docs]
    )
    bm25_retriever.k = 5 # Set k value here

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # Build the Hybrid Search RAG Pipeline
    hybrid_rag_pipeline = RetrievalQA.from_chain_type(
        llm=generation_llm, # Now uses the defined llm
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True
    )

    return hybrid_rag_pipeline

# We load the pipeline once, and Streamlit caches it
try:
    rag_pipeline = load_rag_pipeline()
except Exception as e:
    st.error(f"Failed to load RAG pipeline. Please ensure all placeholders are filled correctly.")
    st.error(f"Error: {e}")
    st.stop()



# --- Chat History Management ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with customer inquiries today?"}
    ]

# Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message is from the assistant and has sources, display them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Show Sources"):
                for source_doc in message["sources"]:
                    source_name = source_doc.metadata.get('source', 'Unknown Source')
                    st.write(f"**Source:** {source_name}")
                    st.markdown(source_doc.page_content)
                    st.divider()


# --- Main Chat Input and Response Logic ---

if prompt := st.chat_input("What is the customer's question?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get a response from the RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the pipeline
            rag_response = rag_pipeline.invoke({"query": prompt})
            response = rag_response['result']
            sources = rag_response['source_documents']

            # Display the main response
            st.markdown(response)

            # Display the sources in a collapsible expander
            with st.expander("Show Sources"):
                for source_doc in sources:
                    # Extract the source filename from metadata
                    source_name = source_doc.metadata.get('source', 'Unknown Source')
                    st.write(f"**Source:** {source_name}")
                    st.markdown(source_doc.page_content)
                    st.divider()

    # Add the complete assistant response (with sources) to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

