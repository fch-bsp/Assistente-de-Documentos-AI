import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai
import re

# Configuração inicial
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="DocGenius AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stFileUploader {
        border: 2px dashed #4a8cff;
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f7ff;
    }
    .stButton>button {
        background-color: #4a8cff;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .model-badge {
        background-color: #4a8cff;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .file-counter {
        color: #4a8cff;
        font-weight: bold;
    }
    .file-list {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

def is_greeting(text):
    """Verifica se o texto é uma saudação simples."""
    greeting_patterns = [
        r'\b(oi|olá|hey|hi|hello)\b',
        r'\b(bom dia|boa tarde|boa noite)\b',
        r'\b(tudo bem|como vai|e aí)\b'
    ]
    
    text_lower = text.lower()
    for pattern in greeting_patterns:
        if re.search(pattern, text_lower):
            return True
    
    if len(text_lower.split()) < 3 and len(text_lower) < 20:
        return True
    
    return False

def handle_greeting(prompt):
    """Retorna uma resposta apropriada para saudações."""
    prompt_lower = prompt.lower()
    
    if re.search(r'\b(bom dia)\b', prompt_lower):
        return "Bom dia! Como posso ajudar você com seus documentos hoje?"
    elif re.search(r'\b(boa tarde)\b', prompt_lower):
        return "Boa tarde! Em que posso ser útil com seus documentos?"
    elif re.search(r'\b(boa noite)\b', prompt_lower):
        return "Boa noite! Como posso ajudar com a análise dos seus documentos?"
    elif re.search(r'\b(tudo bem|como vai)\b', prompt_lower):
        return "Estou bem, obrigado por perguntar! Estou pronto para analisar seus documentos."
    else:
        return "Olá! Estou aqui para ajudar com suas perguntas sobre os documentos. O que gostaria de saber?"

def format_file_size(size):
    """Formata o tamanho do arquivo para KB/MB."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024*1024:
        return f"{size/1024:.1f}KB"
    else:
        return f"{size/(1024*1024):.1f}MB"

def main():
    # Cabeçalho moderno
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://via.placeholder.com/150x50?text=DocGenius", width=150)
    with col2:
        st.title("🤖 Assistente de Documentos AI")
        st.caption("Transforme seus PDFs em conhecimento conversacional")

    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        model_name = st.selectbox(
            "Modelo AI",
            ["gemini-1.5-pro-latest", "gemini-1.0-pro"],
            index=0
        )
        st.markdown(f"<div class='model-badge'>Modelo: {model_name}</div>", unsafe_allow_html=True)
        st.divider()
        st.info("Envie seus documentos e faça perguntas sobre o conteúdo")

    # Área de upload moderna
    st.subheader("Envie seus documentos")
    uploaded_files = st.file_uploader(
        "Arraste e solte arquivos PDF ou TXT aqui",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Limite: 200MB por arquivo • Formatos: PDF, TXT"
    )

    if uploaded_files:
        # Mostra os arquivos carregados
        st.success(f"✅ {len(uploaded_files)} arquivo(s) carregado(s) com sucesso!")
        
        with st.expander("📁 Ver arquivos carregados", expanded=False):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({format_file_size(file.size)})")

        # Processamento dos documentos
        documents = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file.name, "page": i+1}
                    ))
            elif file.type == "text/plain":
                text = file.read().decode("utf-8")
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file.name}
                ))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Área de chat moderna
        st.subheader("Chat com seus documentos")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Faça sua pergunta sobre os documentos..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if is_greeting(prompt):
                    greeting_response = handle_greeting(prompt)
                    st.markdown(greeting_response)
                    st.session_state.messages.append({"role": "assistant", "content": greeting_response})
                else:
                    with st.spinner("Analisando documentos..."):
                        try:
                            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                            llm = genai.GenerativeModel(model_name)
                            
                            docs = vectorstore.similarity_search(prompt, k=2)
                            context = "\n\n".join([doc.page_content for doc in docs])
                            
                            response = llm.generate_content(
                                f"Contexto:\n{context}\n\nPergunta: {prompt}\nResposta:"
                            )
                            
                            st.markdown(response.text)
                            
                            with st.expander("📌 Fontes utilizadas", expanded=False):
                                for doc in docs:
                                    st.write(f"**{doc.metadata['source']}** (página {doc.metadata.get('page', 'N/A')}")
                                    st.text(doc.page_content[:300] + "...")
                            
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        
                        except Exception as e:
                            st.error(f"Erro ao gerar resposta: {str(e)}")

if __name__ == "__main__":
    main()