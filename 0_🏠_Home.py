import streamlit as st
import os

st.set_page_config(page_title="Home - Financial Insight Engine", page_icon="🏠")

def is_rag_ready():
    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")
    if not os.path.isdir(embeddings_dir):
        return False
    
    faiss_files = [f for f in os.listdir(embeddings_dir) if f.endswith("_faiss.index")]
    return len(faiss_files) > 0

st.subheader("🔍 System Status")
rag_status = is_rag_ready()
col1, col2 = st.columns(2)

with col1:
    st.info("**Index Prediction:** Ready to use")
    
with col2:
    if rag_status:
        st.info("**News RAG:** Ready to use")
    else:
        st.warning("**News RAG:** Setup needed (missing embeddings data)")
        if st.button("Show RAG Setup Instructions"):
            st.code("""
# To set up the RAG feature:
1. Download the embeddings zip file
2. Place it in the project root directory
3. Run the setup script: ./setup_rag.sh
4. Add your Gemini API key to the .env file
            """, language="bash")

# --- Home Page Content ---
st.title("📊 Financial Insight Engine")  # Main title for the application
st.markdown("""
Welcome! This application combines quantitative market prediction with qualitative news analysis, designed as part of the CSEG1021 Python Programming project.

**Navigate using the sidebar on the left to access different features:**
*   **🏠 Home:** You are here! This page provides an overview.
*   **📈 Index Prediction:** Predict the next day's high for the Nifty 50 index using a Linear Regression model trained on historical Open, High, Low, Close (OHLC) data. Includes interactive charts and trend indicators.
*   **📰 News RAG:** Search and get AI-powered explanations based on recent financial news articles.
""")

# System status section
st.subheader("🔍 System Status")

# Check RAG status and display appropriate message
rag_status = is_rag_ready()
col1, col2 = st.columns(2)

with col1:
    st.info("**Index Prediction:** Ready to use")
    
with col2:
    if rag_status:
        st.info("**News RAG:** Ready to use")
    else:
        st.warning("**News RAG:** Setup needed (missing embeddings data)")
        if st.button("Show RAG Setup Instructions"):
            st.code("""
# To set up the RAG feature:
1. Download the embeddings zip file
2. Place it in the project root directory
3. Run the setup script: ./setup_rag.sh
4. Add your Gemini API key to the .env file
            """, language="bash")

st.markdown("---")
st.markdown("""
This project demonstrates concepts including:
*   Data Acquisition (`yfinance`)
*   Data Preprocessing & Cleaning (`pandas`, `numpy`)
*   Exploratory Data Analysis (`matplotlib`, `seaborn`, `mplfinance`)
*   Machine Learning - Linear Regression (`scikit-learn`)
*   Model Evaluation & Persistence (`joblib`)
*   Web Application Development (`streamlit`)
*   Retrieval-Augmented Generation (RAG) with News Data

Select a page from the sidebar to explore the tools.
""")

# --- Footer ---
st.caption("CSEG1021 Project by KUSHAGRA SINGH BISEN")
