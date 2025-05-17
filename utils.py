import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar a pasta para armazenar arquivos PDF
folder_files = Path("./pdfs")
folder_files.mkdir(exist_ok=True)

# Configurações do Gemini API
def configure_gemini():
    """Configura a API do Google Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ Chave API do Google não encontrada. Configure a variável GOOGLE_API_KEY no arquivo .env")
        st.stop()
    
    genai.configure(api_key=api_key)
    return api_key

def load_pdfs():
    """Carrega todos os PDFs da pasta folder_files."""
    docs = []
    for pdf in folder_files.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def cria_chain_conversa():
    """Cria uma cadeia de conversação utilizando o Gemini."""
    # Configurar API do Gemini
    api_key = configure_gemini()
    
    with st.status("Processando documentos...", expanded=True) as status:
        # Carregando PDFs
        st.write("Carregando PDFs...")
        docs = load_pdfs()
        
        # Dividindo documentos em pedaços menores
        st.write("Dividindo documentos em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Criando embeddings com Gemini
        st.write("Criando embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Criando base de vetores
        st.write("Criando base de vetores...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Configurando o modelo LLM com Gemini
        st.write("Configurando o modelo de linguagem...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.5,
            top_p=0.85,
            top_k=40,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )
        
        # Configurando memória
        st.write("Configurando memória de conversação...")
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Criando a cadeia de conversação
        st.write("Finalizando configuração...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        # Guardando chain na sessão
        st.session_state["chain"] = chain
        status.update(label="Processamento concluído!", state="complete", expanded=False)