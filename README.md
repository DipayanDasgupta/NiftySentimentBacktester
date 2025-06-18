# Nifty 50 News Sentiment Backtester

This project is a Streamlit web application designed to backtest a trading strategy for the Nifty 50 index. The strategy generates buy/sell signals based on the sentiment derived from daily news articles related to the Indian stock market.

## Features

*   Interactive dashboard to configure and run backtests.
*   Fetches Nifty 50 historical price data using `yfinance`.
*   Fetches relevant news articles using `NewsAPI.org`.
*   Performs sentiment analysis on news content using NLTK's VADER.
*   Implements a rolling sentiment average strategy (momentum-based).
*   Allows users to adjust:
    *   Backtest period (end date and number of trading days).
    *   Initial capital.
    *   Sentiment lookback period (for rolling average).
    *   Positive and negative sentiment thresholds for trade signals.
    *   Transaction costs.
*   Displays backtest results including:
    *   Performance metrics (Total Return, Buy & Hold Return, Sharpe Ratio, Number of Trades).
    *   Visualizations:
        *   Portfolio value vs. Nifty 50 Buy & Hold.
        *   Nifty 50 price chart with trade markers (Buy/Sell signals).
        *   Sentiment analysis chart (daily raw sentiment and rolling average sentiment).
    *   Sample table of backtest data for the last few days.
*   Uses Streamlit's caching to optimize performance and reduce API calls on re-runs.

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   `pip` (Python package installer)

2.  **Clone the Repository (Optional, if hosted on GitHub):**
    ```bash
    git clone https://github.com/DipayanDasgupta/NiftySentimentBacktester.git
    cd NiftySentimentBacktester
    ```
    If you just have the `sentiment_dashboard.py` file, create a project directory and place it inside.

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # .\venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file (as provided separately) in your project root and run:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK VADER Lexicon:**
    The script will attempt to download this on first run if not found. If it fails, you can do it manually:
    ```bash
    python
    >>> import nltk
    >>> nltk.download('vader_lexicon')
    >>> exit()
    ```

## Running the Application

1.  **API Key:**
    *   You will need a free API key from [NewsAPI.org](https://newsapi.org/).
    *   Enter this key into the "NewsAPI Key" field in the application's sidebar. **Do not commit your API key directly into public code if you plan to share it.** Use the sidebar input or environment variables for more secure practices in shared projects.

2.  **Run the Streamlit App:**
    Navigate to your project directory in the terminal (where `sentiment_dashboard.py` is located) and run:
    ```bash
    streamlit run sentiment_dashboard.py
    ```
    The application will open in your default web browser.

## How to Use

1.  Enter your **NewsAPI Key** in the sidebar.
2.  Configure the **Backtest Period** (End Date, Number of Trading Days).
3.  Adjust **Strategy Parameters** (Initial Capital, Sentiment Lookback, Sentiment Thresholds, Transaction Cost).
4.  Click the "🚀 Run Backtest" button.
5.  Analyze the displayed performance metrics, visualizations, and sample data.
6.  Experiment with different parameters to observe their impact on the strategy's performance.

## Important Notes

*   **System Date:** The application uses the system's current date as the default end date for backtests. If your system clock is set to a future date, `yfinance` might provide unusual (projected/test) market data, and `NewsAPI.org` (free tier) has limitations on how far back/forward it can fetch news relative to its *actual* current date. For realistic backtesting on historical data, ensure your system clock is set to the correct current date.
*   **NewsAPI Rate Limits:** The free tier of NewsAPI.org has request limits (e.g., ~100 requests per day). The application uses caching to minimize calls, but frequent re-runs with different date parameters can still hit these limits. If rate-limited, you may need to wait for the limit to reset.
*   **Backtest Limitations:** This is a simplified backtester. It does not account for slippage, more complex order types, or live market microstructure. Results are indicative and based on historical data and the specific strategy logic. Past performance is not indicative of future results.

## Future Enhancements (Ideas)

*   Implement local file caching for news data to overcome API rate limits for repeated analyses.
*   Add more sophisticated sentiment analysis models (e.g., FinBERT).
*   Incorporate more strategy variations (e.g., sentiment delta, price confirmation).
*   Add risk management features like stop-loss or take-profit.
*   Allow selection of different stock tickers or assets.