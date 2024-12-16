import pandas as pd
import sqlite3
import os

aspects = ['Earnings', 'Revenue', 'Margins', 'Dividend', 'EBITDA', 'Debt', 'Sentiment', 'Trendlyne', 'Stock Symbol', 'Rating', 'Verdict', 'Quarterly Metrics' ,'Stakeholder Perception' ,'Strategic Innovation' ,'Equity Dynamics' ,'Technological Proliferation' ,'Executive Foresight' ,'Valuation Analysis' ,'Operational Synergy' ,'Technological Advancements' ,'Economic Impact' ,'Transformational Potential' ,'Strategic Governance' ,'Compliance Framework' ,'Operational Trade-offs' ,'Margin Compression' ,'Sustainable Advantage' ,'Risk Mitigation' ,'Adaptive Capacity' ,'Capital Deployment' ,'Expenditure Management' ,'Growth Trajectory' ,'Economic Prudence' ,'Stakeholder Assurance' ,'Operational Prioritization' ,'Organizational Redesign' ,'Technological Disruption' ,'Structural Evolution' ,'Workforce Recomposition' ,'Strategic Pivot' ,'Value Realization' ,'Market Fluctuations' ,'Technological Legitimacy' ,'Institutional Sentiment' ,'Innovative Potential' ,'Volatility Management' ,'Strategic Investment' ,'Global Market Instability' ,'Portfolio Resilience' ,'Risk Containment' ,'Strategic Optimization' ,'Economic Ambiguity' ,'Investment Acumen' ,'Leadership Vacuum' ,'Cultural Transformation' ,'Strategic Renewal' ,'Market Reassurance' ,'Potential Rejuvenation' ,'Investor Dynamics' ,'Innovation Funding' ,'Profitability Trade-offs' ,'Future Vision' ,'Market Dominance' ,'Resource Efficiency' ,'Strategic Alignment' ,'Debt Restructuring' ,'Cost Optimization' ,'Fiscal Transparency' ,'Structural Vulnerability' ,'Sustainability Strategy' ,'Strategic Transparency', 'Sentiment']
ticker_aliases = {
    'ADANIPORTS.NS': ['Adani Ports'],
    'APOLLOHOSP.NS': ['Apollo Hospitals'],
    'ASIANPAINT.NS': ['Asian Paints'],
    'AXISBANK.NS': ['Axis Bank'],
    'BAJAJ-AUTO.NS': ['Bajaj Auto'],
    'BAJFINANCE.NS': ['Bajaj Finance'],
    'BAJAJFINSV.NS': ['Bajaj Finserv'],
    'BHARTIARTL.NS': ['Bharti Airtel'],
    'BPCL.NS': ['Bharat Petroleum', 'BPCL'],
    'BRITANNIA.NS': ['Britannia'],
    'CIPLA.NS': ['Cipla'],
    'COALINDIA.NS': ['Coal India'],
    'DIVISLAB.NS': ["Divi's Labs"],
    'DRREDDY.NS': ["Dr. Reddy's", "Dr. Reddy's Labs"],
    'EICHERMOT.NS': ['Eicher Motors'],
    'GRASIM.NS': ['Grasim Industries', 'Grasim'],
    'HCLTECH.NS': ['HCL Technologies', 'HCL Tech'],
    'HDFC.NS': ['HDFC'],
    'HDFCBANK.NS': ['HDFC Bank'],
    'HDFCLIFE.NS': ['HDFC Life'],
    'HEROMOTOCO.NS': ['Hero MotoCorp'],
    'HINDALCO.NS': ['Hindalco'],
    'HINDUNILVR.NS': ['Hindustan Unilever'],
    'ICICIBANK.NS': ['ICICI Bank'],
    'INDUSINDBK.NS': ['IndusInd Bank'],
    'INFY.NS': ['Infosys'],
    'ITC.NS': ['ITC'],
    'JSWSTEEL.NS': ['JSW Steel'],
    'KOTAKBANK.NS': ['Kotak Mahindra Bank', 'Kotak Bank'],
    'LT.NS': ['Larsen & Toubro', 'L&T'],
    'MARUTI.NS': ['Maruti Suzuki'],
    'M&M.NS': ['Mahindra & Mahindra'],
    'NESTLEIND.NS': ['Nestle India'],
    'NTPC.NS': ['NTPC'],
    'ONGC.NS': ['Oil & Natural Gas Corporation', 'ONGC'],
    'POWERGRID.NS': ['Power Grid'],
    'RELIANCE.NS': ['Reliance Industries', 'Reliance'],
    'SBILIFE.NS': ['SBI Life'],
    'SBIN.NS': ['State Bank of India', 'SBI'],
    'SHREECEM.NS': ['Shree Cement'],
    'SUNPHARMA.NS': ['Sun Pharmaceutical'],
    'TATACONSUM.NS': ['Tata Consumer Products', 'Tata Consumer'],
    'TATAMOTORS.NS': ['Tata Motors'],
    'TATASTEEL.NS': ['Tata Steel'],
    'TCS.NS': ['Tata Consultancy Services', 'TCS'],
    'TECHM.NS': ['Tech Mahindra'],
    'TITAN.NS': ['Titan Company'],
    'ULTRACEMCO.NS': ['UltraTech Cement'],
    'UPL.NS': ['UPL'],
    'WIPRO.NS': ['Wipro']
}

aspects_average = [asp+"_average" for asp in aspects] 

class NewsDatabase:
    def __init__(self, db_name1='stock_news.db', db_name2='result_stock_news.db'):
        """Initialize the NewsDatabase with a specified SQLite database name and aspects list."""
        
        db_name1 = os.path.join(os.path.dirname(__file__), db_name1)
        db_name2 = os.path.join(os.path.dirname(__file__), db_name2)

        
        # Connect to SQLite database
        self.connection = sqlite3.connect(db_name1)
        self.cursor = self.connection.cursor()
        self.connection2 = sqlite3.connect(db_name2)
        self.cursor2 = self.connection2.cursor()
        
        # Define the list of aspects
        self.aspects = aspects 
        
        # Initialize the SentimentAnalyser with the specified aspects
        
        # Create the table with columns for each aspect
        #self.create_table()
        self.to_dataframe()
        
    def create_table(self):
        """Create a table for news articles with columns for each aspect's score."""
        # Basic schema with fixed columns
        columns = '''
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT,
            headline TEXT,
            published_date TEXT,
            url TEXT,
            embedding BLOB,
        '''
        
        # Add a column for each aspect using the aspect's name directly
        for aspect in range(len(self.aspects)-1):
            columns += f'{self.aspects[aspect]} REAL,\n'
        
        columns += f'{self.aspects[-1]} REAL\n'
        # Create the table if it doesn't already exist
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS news_articles ({columns})
        ''')
        self.connection.commit()
        print(f"Table created with columns for each aspect with this query: {columns}")

    def insert_data(self, ticker, headline, published_date, url):
        """Insert data into the database, including scores for each aspect."""

        # Generate aspect-based sentiment scores for each aspect
        aspect_scores = self.sentiment_analyser.analyze_sentiment(headline)

        # Prepare SQL for dynamic insertion based on aspects
        aspect_columns = ', '.join(self.aspects)  # Create columns list for SQL
        placeholders = ', '.join(['?'] * (4 + len(self.aspects)))  # Placeholder for SQL
        values = [ticker, headline, published_date, url] + list(aspect_scores.values())
        
        print(values, type(values))
        print(aspect_columns, type(aspect_columns))
        # Insert into the database
        self.cursor.execute(f'''
            INSERT INTO news_articles (stock_symbol, headline, published_date, url, {aspect_columns})
            VALUES ({placeholders})
        ''', values)
        self.connection.commit()
        print(f"Data inserted for {ticker} with aspect scores.")

    def to_dataframe(self):
        """Retrieve all data from the database and return it as a Pandas DataFrame with average columns for each aspect, ignoring 0 values."""
        query = "SELECT * FROM news_articles"
        self.df = pd.read_sql_query(query, self.connection)
        query = "SELECT * FROM stock_news_results"

        self.df2 = pd.read_sql_query(query, self.connection2)

        if 'stock_symbol' not in self.df2.columns:
            self.df2['stock_symbol'] = self.df2['ticker']
        return self.df,self.df2

    def get_values_by_ticker(self, ticker):
        """
        Retrieve the values for each aspect for the given stock symbol and return them as a flat dictionary.
        """
        # Filter the dataframe for the specified ticker
        ticker_df = self.df2[self.df2['ticker'] == ticker]
        
        # Convert to dictionary and get the first row without indices
        if not ticker_df.empty:
            return ticker_df.iloc[0].to_dict()
        return {}

    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()
        print("Database connection closed.")
