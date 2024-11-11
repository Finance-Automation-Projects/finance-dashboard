from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.tools import TavilySearchResults
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
import re
import json
import os
import asyncio

# Import your functions
from final_as_of_now.impoter import *

# Initialize global variables
os.environ["TAVILY_API_KEY"] = TAVLY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
COMPANIES = ["ICICIBANK", "HDFCBANK","INFY","RELIANCE","HINDUNILVR"]
PORTFOLIO = None

# Function to retrieve portfolio values (placeholder function)
def retrieve_values(portfolio):
    return {"example_metric": "value"}

class FinancialAssistant:
    def __init__(self, llm):
        self.llm = llm
        
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.query_classifier_chain = self._create_query_classifier()
        self.tavily_search = TavilySearchResults()

    def _create_query_classifier(self):
        template = """You are a financial query classifier. Based on the user's query, classify it into one of the following categories:
        a) Getting the company's financial overview
        b) Peer Comparison Of A Company
        c) News And Sentiment Analysis For A Company
        d) Detailed Analysis Of A Company
        e) Analyzing the company's annual report
        f) Detailed Analysis Of The User's Portfolio
        g) None of the above

        Also extract any company name mentioned in the query.

        Previous conversation:
        {chat_history}

        User query: {query}

        Provide your response in the following JSON format:
        {{
            "category": "letter_of_category",
            "company": "extracted_company_name_or_null"
        }}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain

    async def _process_company_query(self, category: str, company: str, state: Dict[str, Any]):
        if company not in COMPANIES:
            search_results = await self.tavily_search.ainvoke(
                {"query": f"{company} company financial information"}
            )
            state["search_context"] = search_results
        
        state["ticker"] = company
        
        if category == "a":
            result = (await asyncio.to_thread(company_overview_report, state))["company_overview_report"]
        elif category == "b":
            result = (await asyncio.to_thread(peer_comparison_report, state))["peer_comparison_report"]
        elif category == "c":
            result = (await asyncio.to_thread(analyze_sentiment, state))["sentiment_analysis"]
        elif category == "d":
            result = (await asyncio.to_thread(generate_equity_report, state))["equity_research_report"]
        elif category == "e":
            result = (await asyncio.to_thread(rag_annual_report, state))["annual_report_summary"]
            
        return result

    async def _process_portfolio_query(self) -> str:
        if PORTFOLIO is None:
            return "I don't have access to your portfolio yet. Please add your portfolio first."
            
        portfolio_metrics = await asyncio.to_thread(retrieve_values, PORTFOLIO)
        
        summary_template = """Based on the portfolio metrics provided, here's an analysis of your portfolio:
        {metrics}
        
        Would you like to know more about any specific aspect of your portfolio?"""
        
        return summary_template.format(metrics=portfolio_metrics)

    async def _handle_general_query(self, query: str) -> str:
        template = """You are a knowledgeable financial expert having a conversation with a user.
        Use your expertise to provide helpful, accurate information while maintaining a conversational tone.
        
        Previous conversation:
        {chat_history}
        
        User query: {query}
        
        Provide your response:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({
            "query": query,
            "chat_history": self.memory.chat_memory.messages
        })
        return response

    async def process_query(self, query: str) -> str:
        # Get query classification
        classification = await self.query_classifier_chain.ainvoke({
            "query": query,
            "chat_history": self.memory.chat_memory.messages
        })
        
        try:
            classification_dict = json.loads(classification)
        except json.JSONDecodeError:
            # Fallback to eval if JSON parsing fails
            classification_dict = eval(classification)
            
        category = classification_dict["category"]
        company = classification_dict["company"]
        
        state = {"query": query}
        
        if category in ["a", "b", "c", "d", "e"]:
            if not company:
                return "I couldn't identify which company you're asking about. Could you please specify the company name?"
            response = await self._process_company_query(category, company, state)
        elif category == "f":
            response = await self._process_portfolio_query()
        else:
            response = await self._handle_general_query(query)
        
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)
        
        return response

async def main():
    assistant = FinancialAssistant(llm)
    
    queries = [
        "Can you give me an overview of ICICIBANK's financials?",
        "How's INFY innovating compared to its competitors?",
        "What's the latest news about Tesla?",
        "Can you analyze my portfolio performance?",
        "What do you think about cryptocurrency investments?",
    ]
    
    for query in queries:
        response = await assistant.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())