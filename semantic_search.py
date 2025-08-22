from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
import streamlit as st
SUPABASE_URL ="https://zbxxwrnxinzagofdkgdr.supabase.co"
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    docs = splitter.create_documents([text])
    # Lọc chunk chỉ chứa số hoặc quá ngắn
    filtered_docs = [
        doc for doc in docs
        if len(doc.page_content.strip()) > 30 and not doc.page_content.strip().isdigit()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    vectorstore = SupabaseVectorStore.from_documents(
        filtered_docs,
        embedding=embeddings,
        client=supabase,
        table_name="documents"
    )
    return vectorstore
def search_answer(vectorstore, question, top_k=2):
    docs = vectorstore.similarity_search(question, k=top_k)
    return docs
    # return "\n\n".join([doc.page_content for doc in docs])