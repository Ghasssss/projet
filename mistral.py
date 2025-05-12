import os
import gradio as gr
import yfinance as yf
import pandas as pd
import re
import logging
from mistralai import Mistral

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mistral client setup
api_key = "7FbaVyZq544wID4c6mAbzhVgUCfHpyKF"
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Ticker extraction
class TickerExtractor:
    def extract_from_text(self, text):
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})(?=\s+(?:stock|shares|equity))',
            r'\b([A-Z]{1,5})\b'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                for potential_ticker in matches:
                    try:
                        ticker = yf.Ticker(potential_ticker)
                        info = ticker.info
                        if info and "shortName" in info:
                            return potential_ticker
                    except:
                        continue
        return None

    def get_ticker_from_ai(self, text):
        prompt = f"""Extract the stock ticker symbol from this text. If multiple tickers are mentioned, 
        return the most relevant one. If no clear ticker is found, suggest the most likely company's ticker 
        being discussed. Return ONLY the ticker symbol, nothing else.

        Text: {text}"""

        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial expert. Extract or suggest stock tickers from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            potential_ticker = response.choices[0].message.content.strip().upper()
            ticker = yf.Ticker(potential_ticker)
            info = ticker.info
            if info and "shortName" in info:
                return potential_ticker
        except Exception as e:
            logger.error(f"AI ticker extraction failed: {str(e)}")
        return None

# Financial agent
class FinancialAgent:
    def __init__(self):
        self.extractor = TickerExtractor()
        self.context = """You are a sophisticated financial expert assistant. Analyze the provided metrics and generate insights about:
        1. Current market position based on technical indicators
        2. Volume analysis and its implications
        3. Trend strength and potential reversal points
        4. Risk assessment based on volatility metrics
        5. Specific trading signals from indicators

        Format your response with clear sections and bullet points when appropriate."""
        self.history = []

    def analyze_text(self, text):
        ticker = self.extractor.extract_from_text(text)
        if not ticker:
            ticker = self.extractor.get_ticker_from_ai(text)
        if not ticker:
            return "No valid ticker found.", None

        try:
            df = yf.Ticker(ticker).history(period="6mo")
            if df.empty:
                return "Data not found for ticker.", None
        except Exception as e:
            return f"Failed to fetch data: {str(e)}", None

        df_description = df.describe().to_string()

        self.history.append({"role": "user", "content": text})
        prompt = f"Here is the historical data description for {ticker}:\n{df_description}"

        messages = [{"role": "system", "content": self.context}] + self.history + [{"role": "user", "content": prompt}]

        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            analysis = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": analysis})
            return f"**Ticker**: {ticker}\n\n{analysis}", df
        except Exception as e:
            return f"Error generating analysis: {str(e)}", None

# Create agent instance
agent = FinancialAgent()

# Gradio UI
def interface(text):
    result, df = agent.analyze_text(text)
    chart = None
    if df is not None:
        df = df.reset_index()
        fig = df.plot(x="Date", y="Close", title="Closing Price", grid=True).get_figure()
        chart = fig
    return result, chart

# Gradio app
demo = gr.Interface(
    fn=interface,
    inputs=gr.Textbox(label="Enter financial news or question"),
    outputs=[
        gr.Markdown(label="Analysis"),
        gr.Plot(label="Stock Chart")
    ],
    title="Mistral Financial Analyst",
    description="This app uses Mistral AI to extract tickers and generate financial analysis from text input."
)

if __name__ == "__main__":
    demo.launch()
