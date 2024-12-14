import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import googleapiclient.discovery
import numpy as np
from datetime import datetime
from IPython.display import display, HTML

from PortfolioAnalyser import PortfolioAnalyser, Engine, optimized_report, return_values
import base64
from PIL import Image
# Import the interface for the news database
from news_database_interface import interface

##########################################################################################

# from typing import Any, Dict
# import numpy as np
# from langchain_groq import ChatGroq
# import warnings
# import yfinance as yf
# # Initialize the main LLM
# # local_llm = "llama3.2:3b-instruct-q3_K_S"
# # llm = ChatOllama(model=local_llm, temperature=0)
# # Suppress all UserWarnings
# warnings.filterwarnings('ignore', category=UserWarning)

# # Or suppress specific warning messages
# warnings.filterwarnings('ignore', message="Detected filter using positional arguments. Prefer using the 'filter' keyword argument instead.")
# ### 2. Document Loaders and Vector Stores
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_chroma import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema import Document
# from langchain.embeddings.base import Embeddings
# from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.schema import SystemMessage

# ### 1. LLM Setup
# import firebase_admin
# from firebase_admin import firestore,credentials
# import os
# from langchain.schema import HumanMessage
# from tavily import TavilyClient
# import pandas as pd
# from datetime import datetime

from final_as_of_now import impoter
##############################################################################################


# Load news database
obj = interface.NewsDatabase()
database_1, database_2 = obj.to_dataframe()

# Page configuration
st.set_page_config(page_title="Equity Research Agent", layout="wide")

# Styling
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .custom-button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        flex-grow: 1;
        margin: 0 5px;
        text-align: center;
    }
    .news-item {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .news-item h4 {
        margin: 0;
        color: #1f77b4;
    }
    .news-item p {
        margin: 5px 0;
        color: #666;
        font-size: 0.9em;
    }
    .news-item a {
        color: #1f77b4;
        text-decoration: none;
    }
    .news-item a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NTPC.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "SBIN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "TITAN.NS", "ADANIPORTS.NS", "TECHM.NS", "SUNPHARMA.NS",
    "BAJFINANCE.NS", "BRITANNIA.NS", "HINDALCO.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "M&M.NS", "CIPLA.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "APOLLOHOSP.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "UPL.NS", "ADANIENT.NS", "TATASTEEL.NS", "HDFCLIFE.NS", "JSWSTEEL.NS",
    "ONGC.NS", "BPCL.NS", "TATACONSUM.NS", "LTIM.NS", "VEDL.NS"
]

# Initialize session states
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

async def async_wrapper(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return await asyncio.ensure_future(func(*args, **kwargs))
    else:
        return await loop.run_in_executor(None, func, *args, **kwargs)
        
def get_moneycontrol_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_list = []
        news_items = soup.find_all('li', class_='clearfix')
        
        for item in news_items[:5]:
            title_element = item.find('h2')
            if title_element:
                title = title_element.text.strip()
                link = title_element.find('a')['href'] if title_element.find('a') else ""
                time_element = item.find('span', class_='ago')
                time = time_element.text.strip() if time_element else ""
                
                news_list.append({
                    'title': title,
                    'link': link,
                    'time': time
                })
        
        return news_list
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return get_fake_news()

def get_youtube_videos():
    try:
        youtube = googleapiclient.discovery.build(
            'youtube', 'v3',
            developerKey="AIzaSyDiMF5BGMfLZ42xbwGjh4kdYT7GMWE32qw"
        )

        search_response = youtube.search().list(
            q="indian economy market analysis",
            part="snippet",
            maxResults=4,
            type="video",
            relevanceLanguage="en",
            order="date"
        ).execute()

        videos = []
        for item in search_response['items']:
            video = {
                'title': item['snippet']['title'],
                'video_id': item['id']['videoId'],
                'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                'channel': item['snippet']['channelTitle']
            }
            videos.append(video)
        
        return videos
    except Exception as e:
        st.error(f"Error fetching YouTube videos: {str(e)}")
        return []

def get_fake_news():
    return [
        {
            'title': "Markets Rally on Strong Earnings Reports",
            'summary': "Major indices surge as tech giants exceed expectations",
            'link': "https://example.com/news1",
            'time': "1 hour ago"
        },
        {
            'title': "Central Bank Maintains Interest Rates",
            'summary': "Policy remains unchanged amid economic stability",
            'link': "https://example.com/news2",
            'time': "2 hours ago"
        },
        {
            'title': "New Tech IPO Sees Strong Market Debut",
            'summary': "Shares jump 50% on first day of trading",
            'link': "https://example.com/news3",
            'time': "3 hours ago"
        }
    ]

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = ((current_price - prev_close) / prev_close) * 100
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'change_percent': round(change, 2)
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_balance_sheet(symbol):
    try:
        stock = yf.Ticker(symbol)
        balance_sheet = stock.quarterly_balance_sheet
        
        if balance_sheet.empty:
            return None
        
        balance_sheet = balance_sheet.fillna(0)
        balance_sheet = balance_sheet.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        
        return balance_sheet
    except Exception as e:
        st.error(f"Error fetching balance sheet for {symbol}: {str(e)}")
        return None

def init_chat_model(model_name):
    # Uncomment and adjust the following lines if you have these models and API keys
    # if model_name == "OpenAI":
    #     return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"])
    # elif model_name == "Anthropic":
    #     return ChatAnthropic(anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"])
    # elif model_name == "Groq":
    #     return ChatGroq(api_key=st.secrets["GROQ_API_KEY"])
    return None

def show_dashboard():
    st.markdown('<p class="big-font">Equity Research Agent</p>', unsafe_allow_html=True)
    
    # Removed the button container section
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY Chart")
        try:
            nifty_data = yf.download("^NSEI", start=datetime.now() - timedelta(days=365), end=datetime.now())
            
            if nifty_data.empty:
                print("No data retrieved for the selected ticker. Please try again later.")
            else:
                print("Data retrieved successfully!")
                fig = go.Figure(data=[go.Candlestick(x=nifty_data.index,
                                                    open=nifty_data['Open'],
                                                    high=nifty_data['High'],
                                                    low=nifty_data['Low'],
                                                    close=nifty_data['Close'])])
                print("hello")
                fig.update_layout(height=400)
                print("hello1")
                st.plotly_chart(fig, use_container_width=True)
                print("hello2")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        # YouTube Videos Section
        st.subheader("Latest Market Analysis Videos")
        videos = get_youtube_videos()
        video_cols = st.columns(2)
        for idx, video in enumerate(videos):
            with video_cols[idx % 2]:
                st.markdown(f"""
                <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                    <img src="{video['thumbnail']}" style='width:100%;border-radius:5px;'>
                    <h4>{video['title']}</h4>
                    <p>{video['channel']}</p>
                    <a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">Watch Video</a>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Latest Market News")
        news_items = get_moneycontrol_news()
        for item in news_items:
            st.markdown(f"""
            <div class="news-item">
                <h4>{item['title']}</h4>
                <p>{item['time']}</p>
                <a href="{item['link']}" target="_blank">Read more</a>
            </div>
            """, unsafe_allow_html=True)

# def show_stocks():
#     st.title("Stocks Analysis")
    
#     search_term = st.text_input("Search for a stock", "").upper()
    
#     st.subheader("NIFTY 50 Stocks")
    
#     cols = st.columns(5)
#     for i, symbol in enumerate(NIFTY50_SYMBOLS):
#         info = get_stock_info(symbol)
#         if info:
#             with cols[i % 5]:
#                 color = "green" if info['change_percent'] >= 0 else "red"
#                 st.markdown(f"""
#                 <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
#                     <h4>{info['symbol'].replace('.NS', '')}</h4>
#                     <p>₹{info['current_price']}</p>
#                     <p style='color:{color};'>{info['change_percent']}%</p>
#                 </div>
#                 """, unsafe_allow_html=True)

def show_equity_report():
    st.title("Equity Research Report")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.selectbox("Select Stock", NIFTY50_SYMBOLS)
        if symbol:
            try:
                stock = yf.Ticker(symbol)
                
                # Company Info Section
                st.subheader("Company Overview")
                info = stock.info
                with st.expander("Company Information", expanded=True):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Sector", info.get('sector', 'N/A'))
                        st.metric("Market Cap", f"₹{info.get('marketCap', 0):,.0f}")
                        st.metric("52 Week High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}")
                    with col_info2:
                        st.metric("Industry", info.get('industry', 'N/A'))
                        st.metric("P/E Ratio", f"{info.get('trailingPE', 0):,.2f}")
                        st.metric("52 Week Low", f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}")
                
                # Stock Metrics Dashboard
                st.subheader("Stock Metrics Dashboard")
                
                # Custom CSS for enhanced styling
                st.markdown("""
                    <style>
                    .metric-card {
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                        margin-bottom: 10px;
                    }
                    .metric-title {
                        color: #333;
                        font-size: 20px;
                        font-weight: 600;
                        margin-bottom: 10px;
                    }
                    .metric-value {
                        font-size: 30px;
                        font-weight: 700;
                    }
                    .positive { color: #4caf50; } /* Green for positive values */
                    .negative { color: #e53935; } /* Red for negative values */
                    </style>
                    """, unsafe_allow_html=True)

                try:
                    # Normalize symbols
                    database_2['stock_symbol'] = database_2['stock_symbol'].str.strip().str.upper()
                    symbol_normalized = symbol.strip().upper()
                    
                    # Filter data
                    df = database_2[database_2['stock_symbol'] == symbol_normalized]
                    
                    # Check if df is empty
                    if df.empty:
                        st.warning(f"No data available for {symbol}")
                    else:
                        # Convert columns to numeric
                        numeric_cols = ['Earnings', 'Revenue', 'Margins', 'Dividend', 'EBITDA', 'Debt', 'Sentiment']
                        for col in numeric_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Remove rows with all zeros or NaNs
                        df = df.dropna(subset=numeric_cols, how='all')
                        
                        # Compute metrics
                        metrics = {}
                        for col in numeric_cols:
                            valid_data = df.loc[df[col].notnull() & (df[col] != 0), col]
                            metrics[col] = valid_data.mean() if not valid_data.empty else 0
                        
                        # Display Metric Cards with new styling
                        metric_names = ["Earnings Growth", "Revenue Growth", "Profit Margins", "Debt Ratio"]
                        metric_keys = ["Earnings", "Revenue", "Margins", "Debt"]

                        cols = st.columns(4)
                        for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
                            value = metrics[key] * 100  # Assuming metrics are in decimal form
                            color_class = "positive" if value >= 0 else "negative"
                            with cols[i]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-title">{name}</div>
                                    <div class="metric-value {color_class}">{value:.2f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Market Sentiment Gauge (Centered and Semicircular)
                        st.subheader("Market Sentiment")
                        fig_sentiment = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=metrics['Sentiment'] * 100,
                        domain={'x': [0, 1], 'y': [0, 0.5]},  # Display only the top half
                        gauge={
                            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 15}},
                            'bar': {'color': "green" if metrics['Sentiment'] >= 0 else "red"},
                            'bgcolor': "white",
                            'steps': [
                                        {'range': [-100, 0], 'color': 'rgba(255, 0, 0, 0.2)'},
                                        {'range': [0, 100], 'color': 'rgba(0, 255, 0, 0.2)'}
                                    ],
                            'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': metrics['Sentiment'] * 100
                                    }
                                }
                            ))

                        fig_sentiment.update_layout(
                            height=300,
                            margin={'t': 5, 'b': 5, 'l': 0, 'r': 0},
                            showlegend=False,
                        )

                        st.plotly_chart(fig_sentiment, use_column_width=True)



                        # Growth Metrics Bar Chart
                        st.subheader("Growth Metrics")
                        growth_metrics = ['Earnings', 'Revenue', 'EBITDA']
                        growth_values = [metrics[col] * 100 for col in growth_metrics]
                        colors = ['green' if val >= 0 else 'red' for val in growth_values]
                        fig_growth = go.Figure([go.Bar(
                            x=growth_metrics,
                            y=growth_values,
                            marker_color=colors,
                            text=[f"{val:.2f}%" for val in growth_values],
                            textposition='auto'
                        )])
                        fig_growth.update_layout(
                            yaxis_title="Percentage",
                            xaxis_title="Metrics",
                            height=400
                        )
                        st.plotly_chart(fig_growth, use_column_width=True)
                        
                        # Performance Indicators Bar Chart
                        st.subheader("Performance Indicators")
                        performance_metrics = ['Margins', 'Dividend', 'Debt']
                        performance_values = [metrics[col] * 100 for col in performance_metrics]
                        # For Debt, reverse the color coding
                        colors = []
                        for metric, val in zip(performance_metrics, performance_values):
                            if metric == 'Debt':
                                color = 'red' if val >= 0 else 'green'
                            else:
                                color = 'green' if val >= 0 else 'red'
                            colors.append(color)
                        fig_performance = go.Figure([go.Bar(
                            x=performance_metrics,
                            y=performance_values,
                            marker_color=colors,
                            text=[f"{val:.2f}%" for val in performance_values],
                            textposition='auto'
                        )])
                        fig_performance.update_layout(
                            yaxis_title="Percentage",
                            xaxis_title="Indicators",
                            height=400
                        )
                        st.plotly_chart(fig_performance, use_column_width=True)
                        
                except Exception as e:
                    st.error(f"Error loading stock metrics: {str(e)}")
                

                ### Section for AI Generated Report
                valid_list = ["INFY", "RELIANCE", "HINDUNILVR", "HDFCBANK", "ICICIBANK"]
                symbol2 = symbol.replace(".NS", "")
                if symbol2 in valid_list:
                    
                    # Remove .NS from symbol
                    # Put a button to generate report
                    if st.button("Generate AI Report"):

                        input_state = {"ticker": symbol2}
                        result = impoter.graph.invoke(input_state)
                        st.markdown(result.get("equity_research_report", "Report generation failed."))
                        st.markdown("## Sentiment Analysis Section")
                        st.markdown(result.get("sentiment_analysis", "Sentiment analysis failed."))
                        #st.markdown("## Financial Analysis Section")


                else:
                    st.warning("AI Generated Report is only available for selected stocks: INFY.NS, RELIANCE.NS, HINDUNILVR.NS, HDFCBANK.NS, ICICIBANK.NS")
                    print(symbol2)
                    return

                ### Section Ends Here



                # Technical Analysis Section
                st.subheader("Technical Analysis")
                tab1, tab2 = st.tabs(["Price Chart", "Volume Analysis"])
                
                with tab1:
                    # Candlestick chart with additional indicators
                    hist = stock.history(period="1y")
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist['Open'],
                        high=hist['High'],
                        low=hist['Low'],
                        close=hist['Close'],
                        name="OHLC"
                    ))
                    
                    # Add moving averages
                    ma20 = hist['Close'].rolling(window=20).mean()
                    ma50 = hist['Close'].rolling(window=50).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=ma20,
                        name="20-day MA",
                        line=dict(color='orange')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=ma50,
                        name="50-day MA",
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Movement",
                        yaxis_title="Price (₹)",
                        xaxis_title="Date",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Volume analysis
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=hist.index,
                        y=hist['Volume'],
                        name="Volume"
                    ))
                    fig_volume.update_layout(
                        title="Trading Volume",
                        yaxis_title="Volume",
                        xaxis_title="Date",
                        height=300
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                # Fundamental Analysis Section
                st.subheader("Fundamental Analysis")
                tabs_fundamental = st.tabs(["Balance Sheet", "Income Statement", "Key Ratios"])
                
                with tabs_fundamental[0]:
                    balance_sheet = get_balance_sheet(symbol)
                    if balance_sheet is not None and not balance_sheet.empty:
                        st.dataframe(balance_sheet, use_container_width=True)
                    else:
                        st.warning("Balance sheet data not available.")
                
                with tabs_fundamental[1]:
                    try:
                        income_stmt = stock.quarterly_income_stmt
                        if not income_stmt.empty:
                            income_stmt = income_stmt.fillna(0)
                            income_stmt = income_stmt.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                            st.dataframe(income_stmt, use_container_width=True)
                        else:
                            st.warning("Income statement data not available.")
                    except:
                        st.warning("Error fetching income statement data.")
                
                with tabs_fundamental[2]:
                    key_ratios = {
                        "P/E Ratio": info.get('trailingPE', 'N/A'),
                        "P/B Ratio": info.get('priceToBook', 'N/A'),
                        "Debt to Equity": info.get('debtToEquity', 'N/A'),
                        "ROE": info.get('returnOnEquity', 'N/A'),
                        "ROA": info.get('returnOnAssets', 'N/A'),
                        "Current Ratio": info.get('currentRatio', 'N/A'),
                    }
                    
                    ratio_cols = st.columns(3)
                    for i, (ratio_name, ratio_value) in enumerate(key_ratios.items()):
                        with ratio_cols[i % 3]:
                            st.metric(ratio_name, f"{ratio_value:,.2f}" if isinstance(ratio_value, (int, float)) else ratio_value)
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # with col2:
    #     st.subheader("AI Research Assistant")
        
    #     model_options = {
    #         "OpenAI": "GPT-4 - Fast, reliable, good for financial analysis",
    #         "Anthropic": "Claude - Excellent for detailed research and analysis",
    #         "Groq": "LLaMA 2 - Fast inference, good for quick insights"
    #     }
        
    #     selected_model = st.selectbox(
    #         "Choose AI Model",
    #         list(model_options.keys()),
    #         format_func=lambda x: f"{x}: {model_options[x]}"
    #     )
        
    #     if st.button("Initialize Chat"):
    #         st.session_state.chat_model = init_chat_model(selected_model)
    #         st.success(f"Initialized {selected_model} model!")
        
    #     if st.session_state.chat_model:
    #         if symbol:
    #             st.info(f"Ask anything about {symbol} and its performance!")
            
    #         user_input = st.text_input("Ask about the selected stock:")
    #         if user_input and st.button("Send"):
    #             with st.spinner("Generating response..."):
    #                 enhanced_prompt = f"Analysis for {symbol}: {user_input}"
    #                 response = st.session_state.chat_model.invoke(enhanced_prompt)
    #                 st.session_state.chat_history.append(("You", user_input))
    #                 st.session_state.chat_history.append(("Assistant", response))
            
    #         for role, message in st.session_state.chat_history:
    #             if role == "You":
    #                 st.markdown(f"🗣️ **You:** {message}")
    #             else:
    #                 st.markdown(f"🤖 **Assistant:** {message}")
                  
def show_portfolio_analysis():
    st.title("Portfolio Analysis")
    
    # Portfolio input
    st.subheader("Enter Your Portfolio")
    
    portfolio_data = []
    for i in range(5):
        cols = st.columns([2, 1, 1])
        with cols[0]:
            symbol = st.selectbox(f"Stock {i+1}", [""] + NIFTY50_SYMBOLS, key=f"symbol_{i}")
        with cols[1]:
            quantity = st.number_input(f"Quantity", key=f"quantity_{i}", min_value=0)
        with cols[2]:
            buy_price = st.number_input(f"Buy Price", key=f"price_{i}", min_value=0.0)
        
        if symbol and quantity and buy_price:
            portfolio_data.append({
                'symbol': symbol,
                'quantity': quantity,
                'buy_price': buy_price
            })
    
    if st.button("Analyze Portfolio"):
        if portfolio_data:
            st.subheader("Portfolio Analysis")
            portfolio = {}
            total_value = 0
            for item in portfolio_data:
                info = get_stock_info(item['symbol'])
                if info:
                    current_value = info['current_price'] * item['quantity']
                    portfolio[str(item['symbol'])] = float(current_value)
                    investment_value = item['buy_price'] * item['quantity']
                    profit_loss = current_value - investment_value
                    total_value += current_value
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{item['symbol']}</h4>
                        <p>Quantity: {item['quantity']}</p>
                        <p>Current Value: ₹{current_value:,.2f}</p>
                        <p>Profit/Loss: ₹{profit_loss:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"### Total Portfolio Value: ₹{total_value:,.2f}")
        
            stocks = list(portfolio.keys())
            weights = list(portfolio.values())
            
            portfolio = Engine(
                start_date="2024-04-01",
                portfolio=stocks,
                weights=weights
            )
            try:
                st.session_state.assistant.portfolio = portfolio
            except Exception as e:
                print(e)
                pass
            PortfolioAnalyser(portfolio, report=True)
 
            # View report PDF
            # Opening file from file path
            with open("report.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Embedding PDF in HTML
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.warning("Please enter at least one stock in your portfolio.")

    if st.button("Optimize Portfolio over CAGR"):
        if portfolio_data:
            st.subheader("Original Portfolio")
            portfolio = {}
            total_value = 0
            for item in portfolio_data:
                info = get_stock_info(item['symbol'])
                if info:
                    current_value = info['current_price'] * item['quantity']
                    portfolio[str(item['symbol'])] = float(current_value)
                    investment_value = item['buy_price'] * item['quantity']
                    profit_loss = current_value - investment_value
                    total_value += current_value
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{item['symbol']}</h4>
                        <p>Quantity: {item['quantity']}</p>
                        <p>Current Value: ₹{current_value:,.2f}</p>
                        <p>Profit/Loss: ₹{profit_loss:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"### Total Portfolio Value: ₹{total_value:,.2f}")
        
            stocks = list(portfolio.keys())
            weights = list(portfolio.values())
            
            portfolio = Engine(
                start_date="2024-04-01",
                portfolio=stocks,
                weights=weights
            )

            stocks, weights = optimized_report(portfolio, optimize="CAGR")

            st.subheader("Optimized Portfolio")
            portfolio = {}
            total2 = 0
            for i, stock in enumerate(stocks):
                info = get_stock_info(stock)
                if info:
                    current_value = info['current_price']
                    portfolio[str(stock)] = float(current_value)
                    quantity = int(weights[i] * total_value / current_value) + 1
                    total2 += current_value * quantity
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{stock}</h4>
                        <p>Quantity: {quantity}</p>
                        <p>Current Value: ₹{current_value * quantity:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"### Total Optimized Portfolio Value: ₹{total2:,.2f}")

            # View report PDF
            st.subheader("Optimized Portfolio Report")
            # Opening file from file path
            with open("optim_report.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Embedding PDF in HTML
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)


            st.subheader("Comparison of Portfolio Returns")
            image = Image.open('optimized_returns.png')
            st.image(image, caption='Optimized Portfolio Returns', use_column_width=True)
        else:
            st.warning("Please enter at least one stock in your portfolio.")
        
    if st.button("Optimize Portfolio over Sharpe Ratio"):
        if portfolio_data:
            st.subheader("Original Portfolio")
            portfolio = {}
            total_value = 0
            for item in portfolio_data:
                info = get_stock_info(item['symbol'])
                if info:
                    current_value = info['current_price'] * item['quantity']
                    portfolio[str(item['symbol'])] = float(current_value)
                    investment_value = item['buy_price'] * item['quantity']
                    profit_loss = current_value - investment_value
                    total_value += current_value
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{item['symbol']}</h4>
                        <p>Quantity: {item['quantity']}</p>
                        <p>Current Value: ₹{current_value:,.2f}</p>
                        <p>Profit/Loss: ₹{profit_loss:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown(f"### Total Portfolio Value: ₹{total_value:,.2f}")
        
            stocks = list(portfolio.keys())
            weights = list(portfolio.values())
            
            portfolio = Engine(
                start_date="2024-04-01",
                portfolio=stocks,
                weights=weights
            )
            
            stocks, weights = optimized_report(portfolio, optimize="Sharpe")

            st.subheader("Optimized Portfolio")
            portfolio = {}
            total2 = 0
            for i, stock in enumerate(stocks):
                info = get_stock_info(stock)
                if info:
                    current_value = info['current_price']
                    portfolio[str(stock)] = float(current_value)
                    quantity = int(weights[i] * total_value / current_value) + 1
                    total2 += current_value * quantity
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'>
                        <h4>{stock}</h4>
                        <p>Quantity: {quantity}</p>
                        <p>Current Value: ₹{current_value * quantity:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"### Total Optimized Portfolio Value: ₹{total2:,.2f}")

            # View report PDF
            st.subheader("Optimized Portfolio Report")
            # Opening file from file path
            with open("optim_report.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Embedding PDF in HTML
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            
            # Displaying File
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.subheader("Comparison of Portfolio Returns")
            image = Image.open('optimized_returns.png')
            st.image(image, caption='Optimized Portfolio Returns', use_column_width=True)
        else:
            st.warning("Please enter at least one stock in your portfolio.")

import test as tt
######################CHATBOT SECTION################################################
import streamlit as st
import asyncio
from test import FinancialAssistant, ChatGroq, GROQ_API_KEY
import os
import test as tt

def initialize_assistant():
    """Initialize the FinancialAssistant if it doesn't exist in session state"""
    if 'assistant' not in st.session_state:
        llm = ChatGroq(
            temperature=0.1,
            model_name="llama-3.2-90b-text-preview",
            groq_api_key=GROQ_API_KEY,
            max_tokens=3000
        )
        st.session_state.assistant = FinancialAssistant(llm=llm)

def initialize_chat_history():
    """Initialize chat history if it doesn't exist in session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display all messages in the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

async def process_user_input(user_input: str):
    """Process the user input and get AI response"""
    response = await st.session_state.assistant.process_query(user_input)
    return response

# def ai_chatbot():
#     st.title("Financial Assistant Chatbot")
    
#     # Initialize the assistant and chat history
#     initialize_assistant()
#     initialize_chat_history()
    
#     # Display chat history
#     display_chat_history()
    
#     # Chat input
#     if user_input := st.chat_input("What would you like to know about?"):
#         # Display user message
#         st.chat_message("user").markdown(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})
        
#         # Get AI response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 # Create event loop and run async function
#                 # loop = asyncio.new_event_loop()
#                 # asyncio.set_event_loop(loop)
#                 # response = loop.run_until_complete(process_user_input(user_input))
#                 # loop.close()
                
#                 # # # Display AI response
#                 # st.markdown(response)
#                 # st.session_state.messages.append({"role": "assistant", "content": response})
#                 #if user_input and st.button("Send"):
#                     #with st.spinner("Generating response..."):
#                         response = asyncio.run(async_wrapper(
#                             st.session_state.assistant.process_query, user_input
#                         ))
#                         #response = process_user_input(user_input)
#                         # if 'chat_history' not in st.session_state:
#                         #     st.session_state.chat_history = []
#                         # st.session_state.chat_history.append(("You", user_input))
#                         # st.session_state.chat_history.append(("Assistant", response))
#                         st.markdown(response)
#                         st.session_state.messages.append({"role": "assistant", "content": response})


async def async_wrapper(func, *args, **kwargs):
    """Wrapper to run async functions in Streamlit"""
    return await func(*args, **kwargs)

def ai_chatbot():
    st.title("Financial Assistant Chatbot")
   
    # Initialize the assistant and chat history
    initialize_assistant()
    initialize_chat_history()
   
    # Display chat history
    display_chat_history()
   
    # Chat input
    if user_input := st.chat_input("What would you like to know about?"):
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
       
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use asyncio.run() in a more controlled manner
                    response = asyncio.run(
                        async_wrapper(
                            st.session_state.assistant.process_query, 
                            user_input
                        )
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
#######################################################################
# Sidebar Navigation
with st.sidebar:
    st.markdown('<p class="big-font">Navigation</p>', unsafe_allow_html=True)
    
    if st.button("Dashboard", key="dashboard_btn"):
        st.session_state.page = "Dashboard"
    if st.button("Equity Report", key="report_btn"):
        st.session_state.page = "Equity Report"
    if st.button("Portfolio Analysis", key="portfolio_btn"):
        st.session_state.page = "Portfolio Analysis"
    if st.button("AI Chatbot", key="ai_chatbot_btn"):
        st.session_state.page = "AI Chatbot"

# Main content based on selected page
if st.session_state.page == "Dashboard":
    show_dashboard()
# elif st.session_state.page == "Stocks":
#     show_stocks()
elif st.session_state.page == "Equity Report":
    show_equity_report()
elif st.session_state.page == "Portfolio Analysis":
    show_portfolio_analysis()
elif st.session_state.page == "AI Chatbot":
    ai_chatbot()
