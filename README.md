# Financial Insight Engine

**Repository:** [https://github.com/kushgbisen/financial-insight-engine](https://github.com/kushgbisen/financial-insight-engine)

**Live Demo (via Colab & ngrok):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14HjAdWJ11iJXsZmcuCx2XRO8fV2lDUPT?usp=sharing)
*(Note: Requires Google account. Clones the repo and runs the app within Colab using ngrok for access.)*

## Project Goal

Develop a web application combining quantitative stock index prediction and qualitative financial news analysis using Retrieval-Augmented Generation (RAG). Built for the CSEG1021 Python Programming course (BCA-B2).

## Core Features

*   **Nifty 50 Prediction (` Index Prediction` page):**
    *   **Task:** Predicts next trading day's high price for Nifty 50.
    *   **Model:** Linear Regression (Scikit-learn) using OHLC features.
    *   **Input:** Fetches latest Nifty 50 data automatically (`yfinance`).
    *   **Output:** Predicted high value.
    *   **Visualization:** Interactive Plotly candlestick chart (1 Month - Max timeframe).
    *   **Indicator:** Latest Close vs. 5-Day SMA.
    *   **Performance:** R  0.998 on test set (see notebook).

*   **Financial News Analysis (` News RAG` page):**
    *   **Task:** Search financial news and generate AI explanations for user queries.
    *   **Search:** FAISS vector search over pre-computed news embeddings for ~180+ Indian stocks.
    *   **Generation:** Google Gemini API for concise, point-wise summaries based on retrieved articles.
    *   **Requires:** Google Gemini API Key (`.env` file) and `embeddings/` data.

*   **Home Page (` Home` page):**
    *   Application overview and navigation.
    *   Status check for prediction model and RAG data.

## Technology Stack

*   **Language:** Python 3
*   **Web Framework:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **ML/Prediction:** Scikit-learn, Joblib
*   **Data Acquisition:** yfinance
*   **RAG Search:** FAISS (CPU)
*   **RAG Embeddings:** (Offline: Sentence Transformers)
*   **RAG LLM:** Google Gemini (`google-generativeai`)
*   **Plotting:** Plotly, mplfinance, Matplotlib, Seaborn
*   **Configuration:** ConfigParser, python-dotenv

## Project Structure

```
financial-insight-engine/

 app.py                     # Main Streamlit script (minimal)
 pages/                     # Streamlit page modules
    0__Home.py           # Home page UI and logic
    1__Index_Prediction.py # Prediction page UI and logic
    2__News_RAG.py       # RAG page UI and logic

 data/                      # Raw data
    NIFTY50_raw_max.csv    # Nifty 50 OHLC data

 models/                    # Saved ML models & scalers
    nifty_high_lr_model.joblib # Prediction model
    ohlc_scaler.joblib     # Feature scaler

 notebooks/                 # Development notebooks
    NIFTY50LR.ipynb        # Nifty prediction model development

 embeddings/                # Pre-processed RAG data
    TICKER_clean.parquet   # Cleaned news data per ticker
    TICKER_faiss.index     # FAISS index per ticker
    ... (files for ~180+ tickers)

 .env                       # API Keys (GITIGNORED)
 .gitattributes             # Git LFS config (if used)
 .gitignore                 # Files ignored by Git
 config.ini                 # App configuration
 launch.sh                  # App launch script
 requirements.txt           # Python dependencies
 README.md                  # This file
```

## Setup Instructions

**Prerequisites:**
*   Python 3.8+ & pip
*   Git & Git LFS (if managing large `embeddings/` via LFS)

**Steps:**

1.  **Clone:**
    ```bash
    git clone https://github.com/kushgbisen/financial-insight-engine.git
    cd financial-insight-engine
    # If using Git LFS: git lfs pull
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install TA-Lib C Library:** (Dependency for `requirements.txt`)
    *   *Linux (Debian/Ubuntu):* `sudo apt-get update && sudo apt-get install -y libta-lib-dev`
    *   *macOS:* `brew install ta-lib`
    *   *Windows:* Install appropriate wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib).
4.  **Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure API Key:**
    *   Create `.env` file in the project root.
    *   Add line: `GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE`
6.  **Ensure RAG Data:** Verify `embeddings/` directory contains `*_clean.parquet` and `*_faiss.index` files. (Assumed present).

## How to Run Locally

1.  Activate virtual environment: `source venv/bin/activate`
2.  Navigate to project root directory.
3.  Make launch script executable: `chmod +x launch.sh`
4.  Run: `bash launch.sh`
5.  Open the local URL shown (e.g., `http://localhost:8501`).

## Development Details

*   **Prediction Model:** Linear Regression predicting Nifty 50 next day high from current OHLC. See `notebooks/NIFTY50LR.ipynb`.
*   **RAG System:** Uses FAISS indices and Gemini LLM. Configured via `config.ini`.

## Author

*   **Name:** KUSHAGRA SINGH BISEN
*   **Batch:** BCA-B2
*   **SAP ID:** 590014177
