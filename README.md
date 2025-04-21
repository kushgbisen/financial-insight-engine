# Financial Insight Engine

**Repository:** [https://github.com/kushgbisen/financial-insight-engine](https://github.com/kushgbisen/financial-insight-engine)

**Live Demo (via Colab & ngrok):**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14HjAdWJ11iJXsZmcuCx2XRO8fV2lDUPT?usp=sharing)  
*(Note: Requires Google account. Clones the repo and runs the app within Colab using ngrok for access.)*

**📽️ Demo Video:**  

[![Watch the demo](https://img.youtube.com/vi/SqWsuBnbr6w/0.jpg)](https://www.youtube.com/watch?v=SqWsuBnbr6w)

## Project Goal

Develop a web application combining quantitative stock index prediction and qualitative financial news analysis using Retrieval-Augmented Generation (RAG). Built for the CSEG1021 Python Programming course (BCA-B2).

## Core Features

*   **Nifty 50 Prediction (`Index Prediction` page):**
    *   **Task:** Predicts next trading day's high price for Nifty 50.
    *   **Model:** Linear Regression (Scikit-learn) using OHLC features.
    *   **Input:** Fetches latest Nifty 50 data automatically (`yfinance`).
    *   **Output:** Predicted high value.
    *   **Visualization:** Interactive Plotly candlestick chart (1 Month - Max timeframe).
    *   **Indicator:** Latest Close vs. 5-Day SMA.
    *   **Performance:** R² ≈ 0.998 on test set (see notebook).

*   **Financial News Analysis (`News RAG` page):**
    *   **Task:** Search financial news and generate AI explanations for user queries.
    *   **Search:** FAISS vector search over pre-computed news embeddings for ~180+ Indian stocks.
    *   **Generation:** Google Gemini API for concise, point-wise summaries based on retrieved articles.
    *   **Requires:** Google Gemini API Key (`.env` file) and `embeddings/` data.

*   **Home Page (`Home` page):**
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
├── app.py                    # Main Streamlit script (minimal)
├── pages/                    # Streamlit page modules
│   ├── 0__Home.py            # Home page UI and logic
│   ├── 1__Index_Prediction.py # Prediction page UI and logic
│   └── 2__News_RAG.py        # RAG page UI and logic
├── data/                     # Raw data
│   └── NIFTY50_raw_max.csv
├── models/
│   ├── nifty_high_lr_model.joblib
│   └── ohlc_scaler.joblib
├── notebooks/
│   └── NIFTY50LR.ipynb
├── embeddings/
│   ├── TICKER_clean.parquet
│   └── TICKER_faiss.index
├── .env                      # API keys (ignored by Git)
├── .gitattributes
├── .gitignore
├── config.ini                # App configuration
├── launch.sh                 # App launch script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
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
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install TA-Lib C Library:**
    *   *Linux:* `sudo apt-get install -y libta-lib-dev`
    *   *macOS:* `brew install ta-lib`
    *   *Windows:* Download wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set API Key:**
    ```env
    # In .env file
    GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
    ```
6.  **Check `embeddings/` Folder:** Ensure all `.parquet` and `.index` files are present.

## How to Run Locally

1.  Activate environment: `source venv/bin/activate`
2.  Navigate to root directory.
3.  Make script executable: `chmod +x launch.sh`
4.  Run: `bash launch.sh`
5.  Visit: `http://localhost:8501`

## Development Details

*   **Prediction Model:** See `notebooks/NIFTY50LR.ipynb`
*   **RAG System:** FAISS + Google Gemini (see `config.ini`)

## Author

**Name:** KUSHAGRA SINGH BISEN  
**Batch:** BCA-B2  
**SAP ID:** 590014177
