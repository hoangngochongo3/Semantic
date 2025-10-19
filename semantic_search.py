from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
SUPABASE_URL ="https://zbxxwrnxinzagofdkgdr.supabase.co"
SUPABASE_KEY ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpieHh3cm54aW56YWdvZmRrZ2RyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQyMTUzMjEsImV4cCI6MjA2OTc5MTMyMX0.d3DMvN1CWdm9iVzViNR4vJApjYZZKpWZGS60u1ez0zk"
def build_vector_store(text, sep=None,title=None):
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=50,
    #     separators=["\n\n", "\n", ".", "!", "?"]
    # )
    # docs = splitter.create_documents([text])
    # # Lọc chunk chỉ chứa số hoặc quá ngắn
    # filtered_docs = [
    #     doc for doc in docs
    #     if len(doc.page_content.strip()) > 30 and not doc.page_content.strip().isdigit()
    # ]
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=sep
        # separators=[
        #     "\n\n", "\n", ".", "!", "?",
        #     "Job Overview", "Responsibilities", "Qualifications", 
        #     "Skills", "What to expect", "Salaries", "Benefits", "About us", "How to apply"
        # ]
    )
    docs = splitter.create_documents([text])
    for doc in docs:
        doc.metadata["title"] = title  # Thêm title vào metadata
    # Lọc bỏ chunk quá ngắn hoặc chỉ toàn số/salary
    filtered_docs = [
        doc for doc in docs
        if len(doc.page_content.strip()) > 10
        and not doc.page_content.strip().isdigit()
        # and "salary" not in doc.page_content.lower()
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
def search_answer(vectorstore, question, top_k=3):
    docs = vectorstore.similarity_search(question, k=top_k)
    return docs
    # return "\n\n".join([doc.page_content for doc in docs])

