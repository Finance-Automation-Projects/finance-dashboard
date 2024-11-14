# Automated Market Research Analyst

This project aims to streamline and automate the role of a market research analyst. By leveraging machine learning models, sentiment analysis, data scraping, and a chatbot interface, this system provides real-time market analysis and insights, enhancing efficiency and decision-making capabilities for financial analysts.

## Project Objectives
- **Efficient Data Collection**: Automate and reduce time spent on collecting and analyzing market data.
- **Real-Time Insights**: Provide real-time updates and forecast trends.
- **Sentiment Analysis Integration**: Integrate news sentiment analysis into generated reports.
- **User-Friendly Interface**: Interact with the system through a chatbot interface.

## Key Components
1. **News Database with Sentiment Analysis**:
   - Aspect-based sentiment analysis using **DeBERTa-v3** and **FinBERT**.
   - Daily news updates with sentiment annotations using GitHub Actions.
   - Data stored in SQLite for lightweight, efficient access.

2. **Portfolio Analyzer and Optimizer**:
   - Backtesting and performance metrics generation using **empyrical** and **quantstats**.
   - Risk assessment and optimization using **PyPortfolioOpt**.
   - PDF report generation for historical comparison.

3. **Equity Research Report Generator**:
   - Multi-agent LLM-based system to generate comprehensive reports.
   - Each report covers company overview, sector comparison, sentiment insights, etc.

4. **Chatbot Interface**:
   - Built using Langchain to classify user intents.
   - Routes queries to appropriate agents for financial overviews, peer comparison, sentiment analysis, and portfolio insights.

## Technology Stack
- **Data Scraping**: BeautifulSoup, Selenium
- **Sentiment Analysis Models**: DeBERTa-v3, FinBERT
- **Databases**: Firebase for financials, SQLite for news data
- **Portfolio Optimization**: PyPortfolioOpt, empyrical, quantstats
- **Chatbot Framework**: Langchain
- **CI/CD**: GitHub Actions for automated updates

## Usage
1. **News and Sentiment Analysis**:
   - Automatically fetches and processes daily financial news.
2. **Portfolio Analysis**:
   - Users enter their portfolio through a UI, which then backtests and provides metrics.
3. **Chatbot Interface**:
   - Chat with the system for insights on specific companies, peer comparisons, sentiment analysis, and portfolio performance.

## Challenges and Improvements
- **Data Quality**: Limited by accessible financial data; future work could involve direct data purchases for accuracy.
- **Database Constraints**: Currently using a split database approach; future plans include consolidating into a more robust database.
- **Future Scope**: Extend functionality to cover commodities, improve forecast explainability, integrate portfolio uploading, incorporate macroeconomic factors, and enhance agent interaction.


## Learning Outcomes
- **Data Pipeline Management**: Building automated data collection and processing workflows.
- **Advanced Sentiment Analysis**: Implementing fine-tuned BERT models for aspect-based sentiment analysis.
- **CI/CD**: Automating updates through GitHub Actions for daily news data.
- **UI Design**: Building a UI framework using streamlit and integrating backend components with frontend interface.

## License
This project is licensed under the MIT License.
