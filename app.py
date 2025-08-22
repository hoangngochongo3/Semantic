import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/hf_cache/hub"


import sys
import streamlit as st
import openai
from pdf_processor import extract_text_from_pdf
from semantic_search import build_vector_store, search_answer
from supabase import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

# ===========================
# Cấu hình cache cho HuggingFace embeddings (chạy local)
sys.stdout.reconfigure(encoding='utf-8')
# ===========================
# Supabase config
SUPABASE_URL ="https://zbxxwrnxinzagofdkgdr.supabase.co"
SUPABASE_KEY =st.secrets["SUPABASE_KEY"]

# OpenAI API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("Semantic Search PDF với LangChain + Supabase + OpenAI API")

# Hàm generate bằng OpenAI API
def openai_generate_answer(prompt, openai_token, model_id="gpt-3.5-turbo", max_tokens=256):
    import openai
    client = openai.OpenAI(api_key=openai_token)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI, hãy trả lời ngắn gọn, chính xác và bằng tiếng Việt nếu có thể."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi gọi OpenAI API: {e}"

# ===========================
# Upload PDF
# uploaded_file = st.file_uploader("Tải lên file PDF", type="pdf")

# if uploaded_file and "embedding_saved" not in st.session_state:
#     with st.spinner("Đang trích xuất nội dung PDF..."):
#         with open("temp.pdf", "wb") as f:
#             f.write(uploaded_file.read())
#         text = extract_text_from_pdf("temp.pdf")
#         text = text.replace('\x00', '')  # loại bỏ ký tự NULL

#     with st.spinner("Đang lưu embedding vào Supabase..."):
#         build_vector_store(text)

#     st.session_state["embedding_saved"] = True
#     st.success("Đã lưu embedding vào Supabase! Bạn có thể hỏi ngay.")

# ===========================
# Nhập câu hỏi
question = st.text_input("Nhập câu hỏi:")

if st.button("Submit"):
    if question:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

        vectorstore = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )

        with st.spinner("Đang tìm kiếm câu trả lời..."):
            answer_chunks = search_answer(vectorstore, question)
            if isinstance(answer_chunks, list):
                context = "\n".join([doc.page_content for doc in answer_chunks])
            else:
                context = answer_chunks

        prompt = f"""Dựa trên nội dung sau, hãy tổng hợp lại câu trả lời cho câu hỏi bên dưới bằng tiếng Việt.
Nội dung:
{context}

Câu hỏi: {question}
Trả lời:"""

        with st.spinner("Đang sinh câu trả lời với OpenAI..."):
            answer = openai_generate_answer(prompt, OPENAI_API_KEY, model_id="gpt-3.5-turbo", max_tokens=256)
        
        st.write("Context:", context)
        st.write("---")
        st.subheader("Câu trả lời:")

        st.write(answer)
    else:
        st.error("Vui lòng nhập câu hỏi trước khi tìm kiếm.")