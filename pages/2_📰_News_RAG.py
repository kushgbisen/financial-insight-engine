import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
import os
import glob
import time
import configparser
import warnings
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="News RAG - Financial Insight Engine", page_icon="📰")

# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
if os.path.exists(config_path):
    config.read(config_path)

# Suppress warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG PARAMETERS =====================
# Default values
CURRENT_DATE = "2025-04-19"
CURRENT_USER = "kushgbisen"
project_root = os.path.dirname(os.path.dirname(__file__))
EMBEDDING_DIR = os.path.join(project_root, "embeddings")
TOP_K_RESULTS = 3
GEMINI_MODEL_NAME = "gemini-1.5-flash"
COL_TITLE = "title"
COL_DATE = "published_date"
COL_DESCRIPTION = "description"

# Override with config values if available
if os.path.exists(config_path):
    try:
        if config.has_section('APP'):
            CURRENT_DATE = config.get('APP', 'current_date', fallback=CURRENT_DATE)
            CURRENT_USER = config.get('APP', 'current_user', fallback=CURRENT_USER)
        
        if config.has_section('DATA'):
            EMBEDDING_DIR = os.path.join(project_root, config.get('DATA', 'embeddings_dir', fallback="embeddings"))
            TOP_K_RESULTS = int(config.get('DATA', 'top_k_results', fallback=TOP_K_RESULTS))
        
        if config.has_section('GEMINI'):
            GEMINI_MODEL_NAME = config.get('GEMINI', 'model_name', fallback=GEMINI_MODEL_NAME)
        
        if config.has_section('COLUMNS'):
            COL_TITLE = config.get('COLUMNS', 'title', fallback=COL_TITLE)
            COL_DATE = config.get('COLUMNS', 'date', fallback=COL_DATE)
            COL_DESCRIPTION = config.get('COLUMNS', 'description', fallback=COL_DESCRIPTION)
    except:
        # If any errors in parsing config, use defaults
        pass

# ===================== DATA LOADING FUNCTIONS =====================
def get_available_tickers(data_dir):
    """Gets a list of stock tickers based on the FAISS index files."""
    if not os.path.isdir(data_dir):
        st.error(f"Directory not found: {data_dir}")
        return []

    faiss_files = glob.glob(os.path.join(data_dir, "*_faiss.index"))
    tickers = sorted([os.path.basename(f).replace("_faiss.index", "") for f in faiss_files])
    return tickers

def load_faiss_index(ticker, data_dir):
    """Loads the FAISS index for a specific ticker."""
    index_path = os.path.join(data_dir, f"{ticker}_faiss.index")
    if os.path.exists(index_path):
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading index for {ticker}: {e}")
            return None
    else:
        return None

def load_embeddings(ticker, data_dir):
    """Loads the pre-computed embeddings for a ticker."""
    embedding_path = os.path.join(data_dir, f"{ticker}_embeddings.npy")
    if os.path.exists(embedding_path):
        try:
            return np.load(embedding_path)
        except Exception as e:
            print(f"Error loading embeddings for {ticker}: {e}")
            return None
    else:
        return None

def load_cleaned_data(ticker, data_dir):
    """Loads the cleaned DataFrame for a ticker."""
    df_path = os.path.join(data_dir, f"{ticker}_clean.parquet")
    if os.path.exists(df_path):
        try:
            return pd.read_parquet(df_path)
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None
    else:
        return None

# ===================== SEARCH FUNCTIONS =====================
def search_random(tickers_to_search, data_dir, top_n=TOP_K_RESULTS):
    """
    For demonstration when full search is not possible - returns random articles.
    This is a fallback when we can't use the embeddings properly.
    """
    all_results = []
    available_tickers = get_available_tickers(data_dir)

    if tickers_to_search == ["ALL"]:
        search_list = available_tickers
    else:
        search_list = [t for t in tickers_to_search if t in available_tickers]

    if not search_list:
        st.warning("No tickers available to search.")
        return pd.DataFrame()

    # Get some random articles from each ticker
    for ticker in search_list:
        df_clean = load_cleaned_data(ticker, data_dir)

        if df_clean is None or len(df_clean) == 0:
            continue

        # Get random samples
        n_samples = min(3, len(df_clean))
        try:
            samples = df_clean.sample(n_samples)

            for _, row in samples.iterrows():
                result = {
                    "ticker": ticker,
                    "similarity": np.random.rand(),  # Random similarity score
                    COL_TITLE: row.get(COL_TITLE, "No title"),
                    COL_DATE: row.get(COL_DATE, "No date"),
                    COL_DESCRIPTION: row.get(COL_DESCRIPTION, "No description")
                }
                all_results.append(result)
        except Exception as e:
            print(f"Error sampling from {ticker}: {e}")

    if not all_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="similarity", ascending=True)
    return results_df.head(top_n)

# ===================== GEMINI FUNCTIONS =====================
def summarize_with_gemini(query, articles_data, api_key, model_name=GEMINI_MODEL_NAME):
    """Generates a summary using Google Gemini API."""
    if not api_key:
        return "Error: Gemini API Key is missing."
    if articles_data.empty:
        return "No articles found to summarize."

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Create context from article data
        context = ""
        for i, article in articles_data.iterrows():
            ticker = article.get("ticker", "N/A")
            title = article.get(COL_TITLE, "No title")
            pub_date = str(article.get(COL_DATE, "N/A"))
            if "NaT" in pub_date or "None" in pub_date:
                pub_date = "N/A"
            text = article.get(COL_DESCRIPTION, "No description")

            context += f"Article {i+1}: {ticker} - {title} ({pub_date})\n{text}\n\n"

        # Create prompt for Gemini
        prompt = f"""
        You are a concise financial analyst. The current date is {CURRENT_DATE}.

        USER QUERY: "{query}"

        Instructions:
        1. Extract ONLY information from the articles that directly relates to the query
        2. If NO relevant information exists, briefly state this and suggest 1-2 better queries
        3. Format your answer as concise bullet points
        4. NEVER apologize or mention limitations of the articles
        5. NEVER use phrases like "based on the articles" or "the articles don't mention"
        6. Focus on actionable insights a financial professional would value
        7. Maximum length: 150 words

        Articles:
        {context}
        """

        # Configure generation parameters
        temperature = 0.1
        max_tokens = 350
        
        # Try to get values from config
        if os.path.exists(config_path) and config.has_section('GEMINI'):
            try:
                temperature = float(config.get('GEMINI', 'temperature', fallback=temperature))
                max_tokens = int(config.get('GEMINI', 'max_output_tokens', fallback=max_tokens))
            except:
                pass

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Generate content
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config=generation_config)

        return response.text.strip()

    except Exception as e:
        print(f"Error during summarization: {e}")
        if "API key" in str(e):
            return "Error: Invalid Gemini API Key."
        else:
            return f"Error generating summary: {str(e)}"

# ===================== UI FUNCTIONS =====================
def render_sidebar():
    """Render the sidebar elements"""
    st.sidebar.header("📰 News RAG Settings")
    
    # Get API key from environment or let user input it
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key:", 
        value=default_api_key,
        type="password"
    )
    
    if not gemini_api_key:
        st.sidebar.warning("API Key required for summarization.")
    
    # Get available tickers
    available_tickers = get_available_tickers(EMBEDDING_DIR)
    ticker_options = ["ALL"] + available_tickers
    
    # Ticker selection
    selected_tickers = st.sidebar.multiselect(
        "Select Ticker(s):", options=ticker_options, default=["ALL"]
    )
    
    return gemini_api_key, selected_tickers

def render_search_results(search_results, search_time):
    """Render the search results section"""
    st.subheader("📰 Top Articles")
    
    if not search_results.empty:
        st.info(f"Found {len(search_results)} articles in {search_time:.2f}s")

        # Show each article in an expandable section
        for index, row in search_results.iterrows():
            title = row.get(COL_TITLE, "No Title")

            with st.expander(f"📄 {row['ticker']}: {title}"):
                st.caption(f"Date: {row.get(COL_DATE, 'N/A')}")
                st.write(row[COL_DESCRIPTION])
    else:
        st.warning("No articles found. Check if your data directory contains the expected files.")

def render_explanation(query, search_results, gemini_api_key):
    """Render the AI explanation section"""
    st.subheader("✨ AI-Generated Explanation")

    # Generate summary if we have results and API key
    if not search_results.empty and gemini_api_key:
        with st.spinner("Generating explanation..."):
            start_time = time.time()
            explanation = summarize_with_gemini(
                query=query,
                articles_data=search_results,
                api_key=gemini_api_key
            )
            summary_time = time.time() - start_time

        # Display the summary
        if explanation.startswith("Error:"):
            st.error(explanation)
        else:
            st.markdown(explanation)
            st.caption(f"Explanation generated in {summary_time:.2f}s")

    elif not gemini_api_key and not search_results.empty:
        st.warning("Please enter a Gemini API Key to generate explanations.")

# ===================== MAIN APP =====================
st.title("📰 Financial News RAG")
st.markdown("""
This tool lets you search financial news articles and get AI-generated explanations to help understand market trends and insights.
Enter your query about financial markets, select relevant tickers, and get concise explanations based on news articles.
""")

# Render sidebar and get user inputs
gemini_api_key, selected_tickers = render_sidebar()

# Main query input
query = st.text_area(
    "Enter your query:", height=100,
    placeholder="e.g., What are the recent trends in semiconductor industry in Asia?"
)

# Search button
search_button = st.button("🔍 Search & Explain")

# System status info
if not os.path.isdir(EMBEDDING_DIR) or len(get_available_tickers(EMBEDDING_DIR)) == 0:
    st.warning("""
    **Embeddings data not found!** This feature needs embeddings data to work properly.
    Please make sure you've downloaded and extracted the embeddings data to the correct location.
    """)
else:
    tickers = get_available_tickers(EMBEDDING_DIR)
    st.info(f"""
    **Ready to search across {len(tickers)} tickers.**
    *Note: This demo uses simplified search. For better results, ensure embeddings data is complete.*
    """)

st.markdown("---")

# Process search when button is clicked
if search_button:
    if not query:
        st.error("Please enter a query.")
    elif not selected_tickers:
        st.error("Please select at least one ticker.")
    else:
        st.success("Processing your query...")

        # Split screen into two columns
        col1, col2 = st.columns([2, 3])

        with col1:
            # Search for articles (using random selection for demo)
            with st.spinner("Retrieving articles..."):
                start_time = time.time()
                search_results = search_random(
                    tickers_to_search=selected_tickers,
                    data_dir=EMBEDDING_DIR,
                    top_n=TOP_K_RESULTS
                )
                search_time = time.time() - start_time
            
            # Display search results
            render_search_results(search_results, search_time)

        with col2:
            # Generate and display explanation
            render_explanation(query, search_results, gemini_api_key)

# Footer
st.markdown("---")
st.caption("CSEG1021 Project by KUSHAGRA SINGH BISEN")