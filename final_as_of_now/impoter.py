### 1. LLM Setup
import firebase_admin
from firebase_admin import firestore,credentials
import os
from langchain.schema import HumanMessage
from tavily import TavilyClient
import pandas as pd
from datetime import datetime
pwd = os.getcwd()
# Get the directory in which this file is located
module_import_path = os.path.join(pwd, "final_as_of_now")
# interface is in the final_as_of_now folder

from io import BytesIO
import requests
import fitz  # PyMuPDF
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from tqdm import tqdm
def setup_logger():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_firefox_headers():
    """Return headers that mimic Firefox browser"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update(get_firefox_headers())
    return session
def extract_text_from_pdf_url(url):
    """
    Extract text from a PDF URL without downloading to disk
    
    Args:
        url (str): URL of the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        requests.exceptions.RequestException: If download fails
        fitz.FileDataError: If PDF processing fails
    """
    logger = setup_logger()
    session = create_session_with_retries()
    
    try:
        # Get PDF content with progress bar
        response = session.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Download PDF into memory
        pdf_data = BytesIO()
        with tqdm(total=total_size, unit='iB', unit_scale=True,
                 desc=f"Processing PDF from {url.split('/')[-1]}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_data.write(chunk)
                    pbar.update(len(chunk))
        
        # Reset BytesIO position
        pdf_data.seek(0)
        
        # Extract text from PDF in memory
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        
        logger.info(f"Successfully extracted text from PDF at {url}")
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

import sys
sys.path.append(module_import_path)
import interface
#
from typing import Any, Dict
import numpy as np
from langchain_groq import ChatGroq
import warnings
import yfinance as yf
# Initialize the main LLM
# local_llm = "llama3.2:3b-instruct-q3_K_S"
# llm = ChatOllama(model=local_llm, temperature=0)
# Suppress all UserWarnings
warnings.filterwarnings('ignore', category=UserWarning) 

# Or suppress specific warning messages
warnings.filterwarnings('ignore', message="Detected filter using positional arguments. Prefer using the 'filter' keyword argument instead.")
### 2. Document Loaders and Vector Stores
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
import tempfile
### 3. Agent Prompts

GROQ_API_KEY = "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"  # Replace with your actual API key
TAVLY_API_KEY = "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"

# Initialize Groq LLM
llm = ChatGroq(
    api_key = GROQ_API_KEY,
    temperature=0,
    model_name="llama-3.2-90b-text-preview"  
)
# Report Generator

report_generator_prompt = """You are a senior equity research analyst tasked with creating a comprehensive investment report.

You have been provided with two key analyses:

1. Annual Report Analysis:
{annual_report_summary}

2. Market Sentiment and News Analysis:
{sentiment_analysis}

3. Company Overview:
{company_overview_report}

4. Peer Comparison Report:
{peer_comparison_report}

Based on these inputs, generate a detailed equity research report for {ticker}. Your report should include:

1. Executive Summary:
- Clear BUY, SELL, or HOLD recommendation with target price range
- Key investment highlights
- Primary risks

2. Financial Analysis (from Annual Report):
- Revenue and profit trends
- Balance sheet strength
- Cash flow analysis
- Key ratios and metrics
- Management effectiveness

3. Market Position & Sentiment:
- Recent developments and their impact
- Market perception
- Competitive position
- News flow analysis
- Sentiment trends

4. Risk Assessment:
- Business risks
- Financial risks
- Market risks
- Regulatory risks

5. Investment Thesis:
- Growth catalysts
- Competitive advantages
- Valuation perspective
- Timeline for expected developments

6. Forward Looking Assessment:
- Short-term outlook (6-12 months)
- Long-term prospects (2-3 years)
- Key metrics to monitor

Ensure your analysis is data-driven, balanced, and provides clear rationale for all conclusions."""

### 4. Agent Functions

# [NewsAnalyzer class implementation remains the same as before]
class NewsAnalyzer:
    def __init__(self, model_name: str = "llama-3.2-90b-text-preview", groq_api_key:str = GROQ_API_KEY, tavily_api_key: str = TAVLY_API_KEY):
        """Initialize the news analyzer with LLM and Tavily API."""
        self.llm = ChatGroq(
            api_key=groq_api_key,   
            temperature=0,
            model_name="llama-3.2-90b-text-preview"
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key or os.getenv("TAVILY_API_KEY"))
        self.news_db = interface.NewsDatabase()

    def get_headlines(self, ticker: str, limit: int = 20) -> pd.DataFrame:
        """Get and sort headlines for a specific ticker from the database."""
        try:
            # Append '.NS' to the ticker
            ticker_with_suffix = ticker + '.NS'
            
            # Get dataframe from news_db and ensure it's a DataFrame
            df_data = self.news_db.to_dataframe()
            if isinstance(df_data, tuple):
                # If it's a tuple, take the first element assuming it's the DataFrame
                df = df_data[0] if df_data else pd.DataFrame()
            else:
                df = df_data

            if df.empty:
                return pd.DataFrame()
                
            # Filter for ticker
            ticker_df = df[df['stock_symbol'] == ticker_with_suffix].copy()
            
            # Convert published_date to datetime
            if 'published_date' in ticker_df.columns:
                ticker_df['published_date'] = pd.to_datetime(ticker_df['published_date'])
            
            # Check required columns
            if 'title' not in ticker_df.columns or 'url' not in ticker_df.columns:
                print("Required columns are missing from the DataFrame.")
                print(f"Available columns: {ticker_df.columns}")
                return pd.DataFrame()
            
            # Sort by date and get the most recent articles
            sorted_df = ticker_df.sort_values('published_date', ascending=False).head(limit)
            return sorted_df
            
        except Exception as e:
            print(f"Error in get_headlines: {e}")
            return pd.DataFrame()

    def get_article_context(self, headline: str, url: str) -> str:
        """Get additional context about the headline using Tavily API."""
        try:
            search_result = self.tavily_client.search(
                query=f"Context and implications of: {headline}",
                search_depth="advanced",
                max_results=2
            )
            
            # Check the type of search_result
            if isinstance(search_result, dict):
                # If search_result is a dictionary
                context = search_result.get('content', 'No additional context found.')
            elif isinstance(search_result, list):
                # If search_result is a list
                contexts = []
                for result in search_result:
                    if isinstance(result, dict):
                        content = result.get('content')
                        if content:
                            contexts.append(content)
                context = " ".join(contexts) if contexts else "No additional context found."
            elif isinstance(search_result, str):
                # If search_result is a string
                context = search_result
            else:
                context = "No additional context found."
            
            return context
        except Exception as e:
            print(f"Error fetching context: {e}")
            return "Unable to fetch additional context."

    def format_date(self, date: pd.Timestamp) -> str:
        """Convert datetime to a readable format."""
        return date.strftime('%B %d, %Y')
    
    def get_financial_metrics(self, row: pd.Series) -> str:
        """Format non-zero financial metrics into a readable string."""
        metrics = ['Earnings', 'Revenue', 'Margins', 'Dividend', 'EBITDA', 'Debt']
        non_zero_metrics = {metric: row[metric] for metric in metrics if row[metric] != 0}
        if not non_zero_metrics:
            return "No financial metrics reported"
        
        return '\n'.join([f"- {metric}: {value}" for metric, value in non_zero_metrics.items()])

    def format_headlines_for_prompt(self, headlines_df: pd.DataFrame) -> str:
        """Format headlines and their data into a readable string for the prompt."""
        formatted_headlines = []
        
        for idx, row in headlines_df.iterrows():
            context = self.get_article_context(row['title'], row['url'])
            financial_metrics = self.get_financial_metrics(row)
            
            headline_text = f"""Headline {len(formatted_headlines) + 1}:
    Date: {self.format_date(row['published_date'])}
    Title: {row['title']}
    Sentiment Score: {row['Sentiment']}
    Financial Metrics:
    {financial_metrics}
    Context:
    {context}
    """
            formatted_headlines.append(headline_text)
        
        return "\n\n".join(formatted_headlines)

    def generate_analysis(self, headlines_df: pd.DataFrame, query = '') -> str:
        """Generate a comprehensive analysis using the LLM."""
        if headlines_df.empty:
            return "No recent headlines found for this ticker."
            
        formatted_headlines = self.format_headlines_for_prompt(headlines_df)
        sentiment_scores = headlines_df['Sentiment'].tolist()

        prompt = f"""As a stock market analyst, analyze these recent headlines and associated financial metrics for a company. For each headline, evaluate its implications for the company's future performance and market position.

Latest Headlines and Data (from most recent to oldest):

{formatted_headlines}

Please provide a comprehensive analysis that includes:

1. Latest Development Analysis:
- Analyze the most recent headline in detail
- Evaluate any associated financial metrics and their implications
- Explain whether this development is BULLISH, BEARISH, or NEUTRAL for the company
- Support your assessment with context and available financial data

2. Financial Metrics Analysis:
- Analyze any changes in key financial metrics (Earnings, Revenue, Margins, etc.)
- Identify trends or patterns in the financial data
- Explain how these metrics support or contradict the narrative from headlines

3. Trend Analysis:
- Compare the nature of news across all headlines
- Identify if there's an improvement or deterioration in company developments
- Note any pattern in the types of news (operational, strategic, market-related)

4. Market Sentiment Evolution:
- Each of these values is a real number between -1 and 1, where -1 is extremely negative, 0 is neutral, and 1 is extremely positive.
- Whenever refering to a news headline, do not refer to it by its number, but by its content.
- Also, try to not print the sentiment scores, but rather analyze them in the context of the news.
- Analyze the sentiment scores: {sentiment_scores}
- Explain if the news trajectory suggests strengthening or weakening market position
- Note any divergence between sentiment scores and actual news/financial impact

5. Forward Looking Assessment:
- Based on these developments and metrics, provide a brief outlook
- Highlight key areas to monitor going forward
- Identify potential risks and opportunities based on the available data

Keep the analysis evidence-based and focused on both qualitative news impact and quantitative financial metrics."""
        if query != '':  
           prompt += "\n Additionally answer the following question: \n"+query
        response = self.llm.invoke([HumanMessage(content=prompt)])
        #print("SENTIMENT ANALYSIS",response.content)
        return response.content
    
    def analyze(self, ticker: str, query = '') -> Dict[str, Any]:
        """Main analysis function that processes headlines and returns results."""
        # Get and process headlines
        recent_headlines = self.get_headlines(ticker)
        
        if recent_headlines.empty:
            return {
                "content": f"No recent headlines found for {ticker}",
                "sentiment_scores": []
            }
        
        # Generate comprehensive analysis
        analysis = self.generate_analysis(recent_headlines,query)
        #print("SENTIMENT ANALYSIS",analysis)
        return {
            "content": analysis,
            "sentiment_scores": recent_headlines['Sentiment'].tolist()
        }
def initialize_firestore(setup_file):
    try:
        # Check if Firebase apps have already been initialized
        #if not firebase_admin._apps:
            cred = credentials.Certificate(setup_file)
            firebase_admin.initialize_app(cred)
        #else:
        #    print("Firebase app already initialized.")
    except Exception as e:
        print(e)
    db = firestore.client()
    return db
class FakeEmbeddings(Embeddings):
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts):
        return [self.embedding[0] for _ in texts]

    def embed_query(self, text):
        return self.embedding[0]
# [AnnualReportAnalyzer class implementation remains the same as before]
def fetch_financials(ticker):
    newpath = os.path.join(module_import_path, "secrets_Balaji.json")
    db = initialize_firestore(newpath)
    docs = db.collection("Stock Financial Data").limit(1).stream()
    doc = next(docs, None).to_dict()  # Get the first document from the iterator
    return doc[ticker]
class AnnualReportAnalyzer:
    def __init__(self, ticker, model_name: str = "llama-3.2-90b-text-preview", groq_api_key:str = GROQ_API_KEY,curtail_length:int = 10000):
        self.llm = ChatGroq(
            api_key=groq_api_key,   
            temperature=0,
            model_name="llama-3.2-90b-text-preview"
        )
        self.report_text = extract_text_from_pdf_url(fetch_financials(ticker)["report_url"])[:curtail_length]
        self.ticker = ticker
        self.obtain_ragger()
        
    def obtain_ragger(self,model_name: str = "all-mpnet-base-v2",chunk_size: int = 4096, chunk_overlap: int = 200,k: int = 2) -> Chroma:
        """
        Chunks texts using CharacterTextSplitter and creates embeddings
        
        Args:
            texts (List[str]): List of texts to process
            model_name (str): Name of the model to use for embeddings
            
        Returns:
            tuple: (List of document chunks, List of embeddings)
        """
        # Convert texts to Document objects
        doc = Document(page_content=self.report_text) 
        
        # Create text splitter
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
            # Split document
        splits = text_splitter.split_documents([doc])
        
        # Create embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        
        # Create temporary directory for Chroma
        persist_directory = tempfile.mkdtemp()
        
        try:
            # Create and persist Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory=persist_directory
            )
            
            # Configure and return retriever
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        except Exception as e:
            # Clean up temporary directory if there's an error
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
            raise e
        return self.retriever
    def analyze_annual_report(self, query: str) -> str:
        try:
            system_prompt = """You are a financial expert and market analyst. Analyze the annual financial report 
            of the company and provide a precise summary of the company's performance and investment potential.
            
            Retrieved Context:
            {context}
            
            If the context is limited or missing, focus on providing general analysis based on the available information.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                prompt,
                document_variable_name="context"
            )
            
            rag_chain = create_retrieval_chain(
                self.retriever,
                question_answer_chain
            )
            
            results = rag_chain.invoke({"input": query})
            #print("Annual Report Anlysis",results["answer"])
            return results["answer"]
        except Exception as e:
            print(f"Error in analyze_annual_report: {str(e)}")
            return f"Unable to analyze annual report for {self.ticker}. Please ensure the required data is available in the database."
class SectorResearchAgent:
    def __init__(self, tavily_api_key: str = TAVLY_API_KEY):
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            temperature=0,
            model_name="llama-3.2-90b-text-preview"
        )
    
    def determine_company_sector(self, ticker: str) -> str:
        """Research and determine company's sector through web search."""
        try:
            # Search for company sector information
            results = self.tavily_client.search(
                query=f"What sector and industry does {ticker} company operate in?",
                search_depth="advanced",
                max_results=2
            )
            
            # Extract relevant information from search results
            sector_info = ""
            if isinstance(results, list):
                sector_info = " ".join([r.get('content', '') for r in results if 'content' in r])
            elif isinstance(results, dict):
                sector_info = results.get('content', '')
            
            # Use LLM to extract sector from the search results
            prompt = f"""Based on this information about {ticker}, what is the company's main sector? 
            Information: {sector_info}
            
            Return ONLY the sector name (e.g., 'Technology', 'Healthcare', 'Finance', etc.) without any additional text or explanation."""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            sector = response.content.strip()
            
            return sector if sector else "Unknown"
            
        except Exception as e:
            print(f"Error determining sector for {ticker}: {e}")
            return "Unknown"
    # Add this method to the SectorResearchAgent class
    def research_competitor(self, competitor_ticker: str) -> str:
        """Research a specific competitor company."""
        try:
            # Search for competitor information using Tavily
            queries = [
                f"{competitor_ticker} company performance and market position",
                f"{competitor_ticker} competitive advantages and strategy",
                f"{competitor_ticker} recent developments and challenges"
            ]
            
            findings = []
            for query in queries:
                results = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=2
                )
                
                if isinstance(results, list):
                    findings.extend(r.get('content', '') for r in results if 'content' in r)
                elif isinstance(results, dict):
                    findings.append(results.get('content', ''))
            
            # Use LLM to synthesize findings
            if findings:
                prompt = f"""Analyze the following information about {competitor_ticker} and provide a concise summary 
                of their competitive position, strengths, and challenges:

                {' '.join(findings)}

                Focus on:
                1. Market position
                2. Key strengths and weaknesses
                3. Recent developments
                4. Competitive advantages
                """
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            
            return f"No detailed information found for competitor {competitor_ticker}"
            
        except Exception as e:
            print(f"Error researching competitor {competitor_ticker}: {e}")
            return f"Unable to analyze competitor {competitor_ticker}"

    def research_sector(self, sector_name: str, ticker: str, timeframe: str = "recent") -> str:
        """Conduct research on sector performance and trends."""
        if sector_name == "Unknown":
            # If sector is unknown, perform company-specific industry research
            queries = [
                f"{ticker} industry analysis and market trends {timeframe}",
                f"{ticker} competitive landscape and market position {timeframe}",
                f"{ticker} industry outlook and market dynamics {timeframe}"
            ]
        else:
            queries = [
                f"Latest {sector_name} sector performance analysis {timeframe}",
                f"{sector_name} sector outlook and challenges {timeframe}",
                f"{sector_name} industry trends and market dynamics {timeframe}"
            ]
        
        findings = []
        for query in queries:
            try:
                results = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=3
                )
                
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and 'content' in result:
                            findings.append(result['content'])
                elif isinstance(results, dict) and 'content' in results:
                    findings.append(results['content'])
                    
            except Exception as e:
                print(f"Error in sector research: {e}")
                continue
                
        return "\n\n".join(findings) if findings else "No sector research data available."

class EnhancedComparison:
    def __init__(self, ticker, model_name: str = "llama-3.2-90b-text-preview", groq_api_key: str = GROQ_API_KEY):
        self.llm = ChatGroq(
            api_key=groq_api_key,   
            temperature=0,
            model_name=model_name
        )
        self.ticker = ticker
        self.financial_data = fetch_financials(ticker)
        self.sector_researcher = SectorResearchAgent()
        
    def enhanced_comparison(self, query: str) -> str:
        # Get peer comparison data
        peer_data = self.financial_data["Peer Comparison"]
        rivals = [ticker_ for ticker_ in peer_data if ticker_ != self.ticker]
        
        # Determine sector through research
        sector = self.sector_researcher.determine_company_sector(self.ticker)
        print(f"Detected sector for {self.ticker}: {sector}")
        
        # Conduct sector research
        sector_research = self.sector_researcher.research_sector(sector, self.ticker)
        
        # Research each competitor
        competitor_research = {}
        for rival in rivals[:2]:  # Limit to top 2 competitors to avoid rate limits
            competitor_research[rival] = self.sector_researcher.research_competitor(rival)
        
        # Create enhanced system prompt
        system_prompt = (
            f"You are a senior equity research analyst conducting a comprehensive competitive analysis "
            f"for {self.ticker} in the {sector} sector. "
            f"Analyze the following data sources to provide a detailed comparison:\n\n"
        )
        
        # Add financial metrics
        system_prompt += f"1. FINANCIAL METRICS\n{self.ticker}:\n"
        for metric in peer_data[self.ticker]:
            system_prompt += f"- {metric}: {peer_data[self.ticker][metric]}\n"
        
        system_prompt += "\nCompetitor Financials:\n"
        for rival in rivals:
            system_prompt += f"{rival}:\n"
            for metric in peer_data[rival]:
                system_prompt += f"- {metric}: {peer_data[rival][metric]}\n"
            system_prompt += "\n"
            
        # Add sector research
        system_prompt += f"\n2. SECTOR ANALYSIS\n{sector_research}\n\n"
        
        # Add competitor research
        system_prompt += "3. COMPETITOR INSIGHTS\n"
        for rival, research in competitor_research.items():
            system_prompt += f"\n{rival} Analysis:\n{research}\n"
        
        analysis_prompt = f"""
        Based on the provided data, create a comprehensive competitive analysis for {self.ticker} in the {sector} sector that includes:
        
        1. Sector Overview:
        - Current sector dynamics and trends
        - Key challenges and opportunities
        - Regulatory environment
        
        2. Competitive Position Analysis:
        - Market share and positioning
        - Relative financial performance
        - Competitive advantages/disadvantages
        
        3. Peer Comparison:
        - Detailed financial metrics comparison
        - Operational efficiency comparison
        - Growth metrics analysis
        
        4. Forward-Looking Assessment:
        - Expected sector developments
        - Competitive strategy implications
        - Growth opportunities and threats
        
        Ensure the analysis is data-driven and provides actionable insights for investment decisions.
        """
        
        # Create the ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", analysis_prompt)
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({"input": query})
        return result.content

def get_stock_price_data(ticker: str) -> dict:
    """Get current stock price and weekly performance"""
    # Add .NS suffix for Indian stocks
    ticker_symbol = f"{ticker}.NS"
    try:
        stock = yf.Ticker(ticker_symbol)
        # Get current data
        current_price = stock.info.get('currentPrice', 0)
        
        # Get historical data for weekly performance
        hist = stock.history(period='5d')
        week_open = hist['Open'].iloc[0] if not hist.empty else 0
        week_change = ((current_price - week_open) / week_open * 100) if week_open != 0 else 0
        
        return {
            "current_price": current_price,
            "week_open": week_open,
            "week_change_percent": round(week_change, 2)
        }
    except Exception as e:
        print(f"Error fetching stock price data: {e}")
        return {
            "current_price": 0,
            "week_open": 0,
            "week_change_percent": 0
        }

# Modify the Overview class
class Overview:
    def __init__(self, ticker, model_name: str = "llama-3.2-90b-text-preview", groq_api_key: str = GROQ_API_KEY):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            temperature=0,
            model_name="llama-3.2-90b-text-preview"
        )
        self.ticker = ticker
        self.financial_data = fetch_financials(ticker)
        self.stock_price_data = get_stock_price_data(ticker)
        
    def company_report(self, query):
        metrics = [metric for metric in self.financial_data if metric not in ["Peer Comparison", "report_url"]]
        
        # Add stock price data to system prompt
        system_prompt = (
            f"You are an expert market analyst and have to provide a detailed overview of the company based on its "
            f"financials and current market data.\n\n"
            f"Current Market Data for {self.ticker}:\n"
            f"- Current Stock Price: ₹{self.stock_price_data['current_price']}\n"
            f"- Weekly Opening Price: ₹{self.stock_price_data['week_open']}\n"
            f"- Weekly Performance: {self.stock_price_data['week_change_percent']}%\n\n"
            f"Company Financials:\n"
        )

        for metric in metrics:
            system_prompt += f"{metric}:\n"
            for sub_metric, value in self.financial_data[metric].items():
                system_prompt += f"--- {sub_metric}: {value}\n"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"input": query})
        return result.content


# Main analysis functions
def analyze_sentiment(state,query = ''):
    ticker = state["ticker"]
    analyzer = NewsAnalyzer()
    analysis_result = analyzer.analyze(ticker,query)
    print("SENTIMENT ANALYSIS===================",analysis_result["content"])
    return {"sentiment_analysis": analysis_result["content"]}

def rag_annual_report(state,query = ''):
    ticker = state["ticker"]
    analyzer = AnnualReportAnalyzer(ticker)
    if query == '':
        query = f"Give me a comprehensive analysis of {ticker}'s financial performance and future outlook."
    analysis_result = analyzer.analyze_annual_report(
        query
    )
    print("ANNUAL REPORT=================",analysis_result)
    return {"annual_report_summary": analysis_result}

def peer_comparison_report(state,query = ''):
    ticker = state["ticker"]
    comparison = EnhancedComparison(ticker)
    if query == '':
        query = f"Provide a comprehensive competitive analysis of {ticker} including sector dynamics and peer comparison."
    peer_report = comparison.enhanced_comparison(
        query
    )
    print("PEER COMPARISION ANALYSIS=================",peer_report)
    return {"peer_comparison_report": peer_report}

def company_overview_report(state,query = ''):
    ticker = state["ticker"]
    overview = Overview(ticker)
    if query == '':
        query = f"Give me a comprehensive overview of {ticker}'s financial performance based on its metrics , stick to the facts and also report them,in your summary"
    company_report = overview.company_report(query)
    print("COMPANY OVERVIEW===============",company_report)
    return {"company_overview_report":company_report}

# Optionally, add validation within the generate_equity_report function
def generate_equity_report(state,query = ''):
    if not all(k in state for k in ("sentiment_analysis", "annual_report_summary", "company_overview_report", "peer_comparison_report")):
        return {"equity_research_report": "Missing required data for report generation."}
    
    ticker = state["ticker"]
    prompt = report_generator_prompt.format(
        ticker=ticker,
        sentiment_analysis=state.get("sentiment_analysis", ""),
        annual_report_summary=state.get("annual_report_summary", ""),
        company_overview_report=state.get("company_overview_report", ""),
        peer_comparison_report=state.get("peer_comparison_report", "")
    )
    if query != '':
        prompt += "\n Additionally answer the following question: \n"+query
    report = llm.invoke([HumanMessage(content=prompt)])
    return {"equity_research_report": report.content}

### 5. Workflow Graph
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Dict

# Define the state structure
class GraphState(TypedDict, total=False):
    ticker: str
    sentiment_analysis: str
    annual_report_summary: str
    equity_research_report: str
    peer_comparison_report: str
    company_overview_report: str
# Initialize the workflow
workflow = StateGraph(GraphState)

# Add nodes for each agent
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("rag_annual_report", rag_annual_report)
workflow.add_node("generate_equity_report", generate_equity_report)
workflow.add_node("peer_comparison", peer_comparison_report)
workflow.add_node("company_overview",company_overview_report)
# Set the entry point to run the sentiment analysis first
workflow.set_entry_point("analyze_sentiment")

# Add edges between nodes without conditions
workflow.add_edge("analyze_sentiment", "company_overview")
workflow.add_edge("company_overview", "peer_comparison")
workflow.add_edge("peer_comparison", "rag_annual_report")
workflow.add_edge("rag_annual_report", "generate_equity_report")
# Add the final report generation
workflow.add_edge("generate_equity_report", END)

# Compile the graph
graph = workflow.compile()

# #Run the analysis
if __name__ == "__main__":
    # Initialize with a ticker
    input_state = {"ticker": "HINDUNILVR"}
    
    # Run the analysis
    print("Starting analysis...")
    result = graph.invoke(input_state)
    
    print("\nFinal Equity Research Report:")
    print(result.get("equity_research_report", "Report generation failed."))