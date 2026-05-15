import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import datetime
import requests
import json

# .env dosyasını yükle
load_dotenv()

# --- ARAYÜZ AYARLARI ---
st.set_page_config(page_title="Terminal v17 - Data Analyst", layout="wide")
st.title("📊 Terminal v17 – Veri Analizi ve Hibrit Zeka")

# 1. API Anahtarlarını al
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Modelleri başlat
yerel_llm = ChatOllama(model="llama3.2:latest", temperature=0)

if OPENROUTER_API_KEY:
    bulut_llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        streaming=True
    )
    st.sidebar.success("✅ Bulut: OpenRouter Aktif")
else:
    bulut_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, streaming=True)

web_tool = TavilySearchResults(max_results=3, include_answer=True)

# --- YARDIMCI FONKSİYONLAR ---

def get_forex_rates():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        data = requests.get(url, timeout=5).json()
        return {"USD": data['rates']['TRY'], "EUR": (1 / data['rates']['EUR']) * data['rates']['TRY']}
    except: return None

# --- DOSYA YÜKLEME VE ANALİZ ---
with st.sidebar:
    st.header("📂 Dosya Yükle")
    uploaded_file = st.file_uploader("Excel veya CSV dosyanı seç", type=['xlsx', 'csv', 'xls'])
    data_context = ""
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("📊 Veri Önizlemesi (İlk 3 Satır):")
            st.dataframe(df.head(3))
            
            # Veriyi modele anlatmak için özetliyoruz
            data_context = f"\n\n[DOSYA ANALİZİ]\nDosya Adı: {uploaded_file.name}\nSütunlar: {df.columns.tolist()}\nSatır Sayısı: {len(df)}\nİstatistiksel Özet:\n{df.describe().to_string()}"
            st.success("Dosya başarıyla işlendi!")
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")

# --- SOHBET GEÇMİŞİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- ANA DÖNGÜ ---
if prompt := st.chat_input("Sorunuzu yazın veya yüklü dosyayı yorumlatın..."):
    # Kullanıcı sorusuna dosya verisini de gizlice ekliyoruz
    combined_prompt = prompt + data_context
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        cevap_alani = st.empty()
        full_response = ""

        # 1. Döviz sorgusu mu?
        if any(k in prompt.lower() for k in ["döviz", "kur", "dolar", "euro"]):
            rates = get_forex_rates()
            if rates:
                cevap = f"**🕒 Güncel Kurlar:**\nUSD: {rates['USD']:.4f} TL | EUR: {rates['EUR']:.4f} TL"
                st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
                st.stop()

        # 2. Karar Mekanizması
        try:
            # Eğer dosya yüklüyse veya güncel bilgi lazımsa Bulut'a git
            if uploaded_file or len(prompt) > 100:
                karar = "EVET" 
            else:
                karar_prompt = f"Soru: {prompt}. Güncel bilgi/karmaşık analiz ister mi? Sadece EVET veya HAYIR."
                karar = yerel_llm.invoke(karar_prompt).content.strip().upper()
        except: karar = "EVET"

        try:
            if "EVET" in karar:
                st.caption("🌐 *Bulut Analist Devreye Girdi...*")
                # Eğer dosya yoksa ama güncel bilgi lazımsa internete bak
                if not uploaded_file:
                    search_result = web_tool.invoke(prompt)
                    context = f"İnternet Verisi: {str(search_result)}"
                else:
                    context = "Dosya verisi yukarıda sağlandı."

                final_prompt = f"Tarih: {datetime.datetime.now()}\n{context}\nSoru: {combined_prompt}"
                
                for chunk in bulut_llm.stream(final_prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += content
                    cevap_alani.markdown(full_response + "▌")
            else:
                st.caption("🏠 *Yerel Yanıt...*")
                for chunk in yerel_llm.stream(prompt):
                    full_response += chunk.content
                    cevap_alani.markdown(full_response + "▌")

            cevap_alani.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Hata: {e}")