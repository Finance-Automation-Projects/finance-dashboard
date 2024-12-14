# # from typing import Dict, Any
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from langchain_core.runnables import RunnablePassthrough
# # from langchain_groq import ChatGroq
# # from langchain.memory import ConversationSummaryBufferMemory
# # from langchain_community.tools import TavilySearchResults
# # from langchain.schema import SystemMessage, HumanMessage, AIMessage
# # from langchain_core.messages import BaseMessage
# # import re
# # import json
# # import os
# # import asyncio

# # # Import your functions
# # from final_as_of_now.impoter import *

# # # Initialize global variables
# # os.environ["TAVILY_API_KEY"] = TAVLY_API_KEY
# # os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# # COMPANIES = ["ICICIBANK", "HDFCBANK","INFY","RELIANCE","HINDUNILVR"]
# # PORTFOLIO = None

# # # Function to retrieve portfolio values (placeholder function)
# # def retrieve_values(portfolio):
# #     return {"example_metric": "value"}

# # class FinancialAssistant:
# #     def __init__(self, llm):
# #         self.llm = llm
        
# #         self.memory = ConversationSummaryBufferMemory(
# #             llm=self.llm,
# #             max_token_limit=2000,
# #             memory_key="chat_history",
# #             return_messages=True
# #         )
        
# #         self.query_classifier_chain = self._create_query_classifier()
# #         self.tavily_search = TavilySearchResults()

# #     def _create_query_classifier(self):
# #         template = """You are a financial query classifier. Based on the user's query, classify it into one of the following categories:
# #         a) Getting the company's financial overview
# #         b) Peer Comparison Of A Company
# #         c) News And Sentiment Analysis For A Company
# #         d) Detailed Analysis Of A Company
# #         e) Analyzing the company's annual report
# #         f) Detailed Analysis Of The User's Portfolio
# #         g) None of the above

# #         Also extract any company name mentioned in the query.

# #         Previous conversation:
# #         {chat_history}

# #         User query: {query}

# #         Provide your response ONLY in the following JSON format. Do not include any other information in the response:
# #         {{
# #             "category": "letter_of_category",
# #             "company": "extracted_company_name_or_null"
# #         }}
# #         """

# #         prompt = ChatPromptTemplate.from_messages([
# #             ("system", template),
# #             MessagesPlaceholder(variable_name="chat_history"),
# #             ("human", "{query}")
# #         ])

# #         chain = prompt | self.llm | StrOutputParser()
# #         return chain

# #     async def _process_company_query(self, category: str, company: str, state: Dict[str, Any]):
# #         if company not in COMPANIES:
# #             search_results = await self.tavily_search.ainvoke(
# #                 {"query": f"{company} company financial information"}
# #             )
# #             state["search_context"] = search_results
        
# #         state["ticker"] = company
        
# #         if category == "a":
# #             result = (await asyncio.to_thread(company_overview_report, state))["company_overview_report"]
# #         elif category == "b":
# #             result = (await asyncio.to_thread(peer_comparison_report, state))["peer_comparison_report"]
# #         elif category == "c":
# #             result = (await asyncio.to_thread(analyze_sentiment, state))["sentiment_analysis"]
# #         elif category == "d":
# #             result = (await asyncio.to_thread(generate_equity_report, state))["equity_research_report"]
# #         elif category == "e":
# #             result = (await asyncio.to_thread(rag_annual_report, state))["annual_report_summary"]
            
# #         return result

# #     async def _process_portfolio_query(self) -> str:
# #         if PORTFOLIO is None:
# #             return "I don't have access to your portfolio yet. Please add your portfolio first."
            
# #         portfolio_metrics = await asyncio.to_thread(retrieve_values, PORTFOLIO)
        
# #         summary_template = """Based on the portfolio metrics provided, here's an analysis of your portfolio:
# #         {metrics}
        
# #         Would you like to know more about any specific aspect of your portfolio?"""
        
# #         return summary_template.format(metrics=portfolio_metrics)

# #     async def _handle_general_query(self, query: str) -> str:
# #         template = """You are a knowledgeable financial expert having a conversation with a user.
# #         Use your expertise to provide helpful, accurate information while maintaining a conversational tone.
        
# #         Previous conversation:
# #         {chat_history}
        
# #         User query: {query}
        
# #         Provide your response:"""
        
# #         prompt = ChatPromptTemplate.from_messages([
# #             ("system", template),
# #             MessagesPlaceholder(variable_name="chat_history"),
# #             ("human", "{query}")
# #         ])
        
# #         chain = prompt | self.llm | StrOutputParser()
# #         response = await chain.ainvoke({
# #             "query": query,
# #             "chat_history": self.memory.chat_memory.messages
# #         })
# #         return response

# #     async def process_query(self, query: str) -> str:
# #         # Get query classification
# #         #print(query)
# #         print(self.memory.chat_memory.messages)
# #         #print(self.query_classifier_chain)
# #         classification = await self.query_classifier_chain.ainvoke({
# #             "query": query,
# #             "chat_history": self.memory.chat_memory.messages
# #         })
# #         print(classification)
# #         try:
# #             classification_dict = json.loads(classification)
# #         except json.JSONDecodeError:
# #             # Fallback to eval if JSON parsing fails
# #             classification_dict = eval(classification)
            
# #         category = classification_dict["category"]
# #         company = classification_dict["company"]
# #         print(category)
# #         state = {"query": query}
        
# #         if category in ["a", "b", "c", "d", "e"]:
# #             if not company:
# #                 return "I couldn't identify which company you're asking about. Could you please specify the company name?"
# #             response = await self._process_company_query(category, company, state)
# #         elif category == "f":
# #             response = await self._process_portfolio_query()
# #         else:
# #             response = await self._handle_general_query(query)
        
# #         self.memory.chat_memory.add_user_message(query)
# #         self.memory.chat_memory.add_ai_message(response)
        
# #         return response

# # async def main():
# #     assistant = FinancialAssistant(llm)
    
# #     queries = [
# #         "Can you give me an overview of ICICIBANK's financials?",
# #         "How's INFY innovating compared to its competitors?",
# #         "What's the latest news about Tesla?",
# #         "Can you analyze my portfolio performance?",
# #         "What do you think about cryptocurrency investments?",
# #     ]
    
# #     for query in queries:
# #         response = await assistant.process_query(query)
# #         print(f"\nQuery: {query}")
# #         print(f"Response: {response}")
# #         print("-" * 50)

# # if __name__ == "__main__":
# #     import asyncio
# #     asyncio.run(main())
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

# # Import your functions
# from final_as_of_now.impoter import *

# # Initialize global variables
# os.environ["TAVILY_API_KEY"] = TAVLY_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY
# COMPANIES = ["ICICIBANK", "HDFCBANK","INFY","RELIANCE","HINDUNILVR"]
# # PORTFOLIO = None

# # # Function to retrieve portfolio values (placeholder function)
# # def retrieve_values(portfolio):
# #     return {"example_metric": "value"}

# class FinancialAssistant:
#     def _init_(self, llm, portfolio=None, retrieve_values=lambda portfolio: {"example_metric": "value"}):
#         self.llm = llm
#         self.portfolio = portfolio
#         self.retrieve_values = retrieve_values
#         self.memory = ConversationSummaryBufferMemory(
#             llm=self.llm,
#             max_token_limit=2000,
#             memory_key="chat_history",
#             return_messages=True
#         )
        
#         self.query_classifier_chain = self._create_query_classifier()
#         self.tavily_search = TavilySearchResults()

#     def _create_query_classifier(self):
#         template = """You are a financial query classifier. Based on the user's query, classify it into one of the following categories:
#         a) Getting the company's financial overview
#         b) Peer Comparison Of A Company
#         c) News And Sentiment Analysis For A Company
#         d) Detailed Analysis Of A Company
#         e) Analyzing the company's annual report
#         f) Detailed Analysis Of The User's Portfolio
#         g) None of the above

#         Also extract any company name mentioned in the query.

#         Previous conversation:
#         {chat_history}

#         User query: {query}

#         Provide your response in the following JSON format:
#         {{
#             "category": "letter_of_category",
#             "company": "extracted_company_name_or_null"
#         }}
#         """

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", template),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{query}")
#         ])

#         chain = prompt | self.llm | StrOutputParser()
#         return chain

#     async def _process_company_query(self, category: str, company: str, state: Dict[str, Any]):
#         if company not in COMPANIES:
#             search_results = await self.tavily_search.ainvoke(
#                 {"query": f"{company} company financial information"}
#             )
#             state["search_context"] = search_results
        
#         state["ticker"] = company
        
#         if category == "a":
#             result = (await asyncio.to_thread(company_overview_report, state))["company_overview_report"]
#         elif category == "b":
#             result = (await asyncio.to_thread(peer_comparison_report, state))["peer_comparison_report"]
#         elif category == "c":
#             result = (await asyncio.to_thread(analyze_sentiment, state))["sentiment_analysis"]
#         elif category == "d":
#             result = (await asyncio.to_thread(generate_equity_report, state))["equity_research_report"]
#         elif category == "e":
#             result = (await asyncio.to_thread(rag_annual_report, state))["annual_report_summary"]
            
#         return result

#     async def _process_portfolio_query(self) -> str:
#         if self.portfolio is None:
#             return "I don't have access to your portfolio yet. Please add your portfolio first."
            
#         portfolio_metrics = await asyncio.to_thread(self.retrieve_values, self.portfolio)
        
#         summary_template = """Based on the portfolio metrics provided, here's an analysis of your portfolio:
#         {metrics}
        
#         Would you like to know more about any specific aspect of your portfolio?"""
        
#         return summary_template.format(metrics=portfolio_metrics)

#     async def _handle_general_query(self, query: str) -> str:
#         template = """You are a knowledgeable financial expert having a conversation with a user.
#         Use your expertise to provide helpful, accurate information while maintaining a conversational tone.
        
#         Previous conversation:
#         {chat_history}
        
#         User query: {query}
        
#         Provide your response:"""
        
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", template),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{query}")
#         ])
        
#         chain = prompt | self.llm | StrOutputParser()
#         response = await chain.ainvoke({
#             "query": query,
#             "chat_history": self.memory.chat_memory.messages
#         })
#         return response

#     async def process_query(self, query: str) -> str:
#         try:
#             # Get query classification with a timeout
#             classification = await asyncio.wait_for(
#                 self.query_classifier_chain.ainvoke({
#                     "query": query,
#                     "chat_history": self.memory.chat_memory.messages
#                 }),
#                 timeout=10.0  # 10 seconds timeout
#             )
            
#             try:
#                 classification_dict = json.loads(classification)
#             except (json.JSONDecodeError, TypeError):
#                 # More robust error handling
#                 classification_dict = {"category": "g", "company": None}
                
#             category = classification_dict.get("category", "g")
#             company = classification_dict.get("company")
            
#             state = {"query": query}
            
#             # Add timeout to prevent hanging
#             try:
#                 if category in ["a", "b", "c", "d", "e"]:
#                     if not company:
#                         return "I couldn't identify which company you're asking about. Could you please specify the company name?"
#                     response = await asyncio.wait_for(
#                         self._process_company_query(category, company, state),
#                         timeout=15.0
#                     )
#                 elif category == "f":
#                     response = await asyncio.wait_for(
#                         self._process_portfolio_query(),
#                         timeout=10.0
#                     )
#                 else:
#                     response = await asyncio.wait_for(
#                         self._handle_general_query(query),
#                         timeout=10.0
#                     )
#             except asyncio.TimeoutError:
#                 return "I'm sorry, but the query took too long to process. Could you please try again?"
            
#             self.memory.chat_memory.add_user_message(query)
#             self.memory.chat_memory.add_ai_message(response)
            
#             return response
        
#         except Exception as e:
#             # Catch-all error handling
#             return f"An unexpected error occurred: {str(e)}"

# async def main():
#     assistant = FinancialAssistant(llm)
    
#     queries = [
#         "Can you give me an overview of ICICIBANK's financials?",
#         "How's INFY innovating compared to its competitors?",
#         "What's the latest news about Tesla?",
#         "Can you analyze my portfolio performance?",
#         "What do you think about cryptocurrency investments?",
#     ]
    
#     for query in queries:
#         response = await assistant.process_query(query)
#         print(f"\nQuery: {query}")
#         print(f"Response: {response}")
#         print("-" * 50)

# if __name__ == "_main_":
#     import asyncio
#     asyncio.run(main())
import asyncio
import json
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.output_parsers import JSONOutputParser,StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Import from your impoter file (assumed to contain these)
from final_as_of_now.impoter import (
    company_overview_report, 
    peer_comparison_report, 
    analyze_sentiment, 
    generate_equity_report, 
    rag_annual_report,
    retrieve_values,
    llm  # Assuming the LLM is imported here
)

class FinancialAssistant:
    def __init__(self, 
                 llm, 
                 portfolio=None, 
                 macroeconomic_data: Dict[str, Dict[str, float]] = None):
        self.llm = llm
        self.portfolio = portfolio
        self.macroeconomic_data = macroeconomic_data or {}
        self.tavily_search = TavilySearchResults(max_results=3)
        
        # Create tools for each specialized agent
        self.tools = self._create_tools()
        
        # Create a tool-aware LLM
        self.tool_calling_llm = self._create_tool_calling_llm()
        
        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _create_tools(self) -> List[BaseTool]:
        def create_tool(name, func, description):
            return BaseTool(
                name=name,
                func=lambda state: func(state, query=state.get('tool_query', '')),
                description=description
            )
        
        return [
            create_tool(
                "company_overview", 
                company_overview_report, 
                "Get a comprehensive financial overview of a specific company"
            ),
            create_tool(
                "peer_comparison", 
                peer_comparison_report, 
                "Compare a company's performance with its peers in the industry"
            ),
            create_tool(
                "sentiment_analysis", 
                analyze_sentiment, 
                "Analyze market sentiment for a specific company"
            ),
            create_tool(
                "equity_report", 
                generate_equity_report, 
                "Generate a detailed equity research report for a company"
            ),
            create_tool(
                "annual_report", 
                rag_annual_report, 
                "Retrieve and summarize a company's annual report"
            ),
            create_tool(
                "portfolio_analysis", 
                lambda state, query='': {"portfolio_metrics": retrieve_values(self.portfolio)}, 
                "Analyze the performance metrics of the user's portfolio"
            ),
            BaseTool(
                name="web_search",
                func=lambda state: {
                    "web_search_results": self.tavily_search.invoke({
                        "query": state.get('tool_query', state.get('query', ''))
                    })
                },
                description="Perform a web search to gather additional context"
            )
        ]

    def _create_tool_calling_llm(self):
        # Modify the LLM to support tool calling
        return self.llm.bind_tools(
            self.tools,
            tool_choice="auto"  # Let the LLM decide which tools to use
        )

    def _create_workflow(self):
        # Define the state schema
        class AgentState(TypedDict):
            query: str
            chat_history: List[Dict[str, str]]
            results: List[Dict[str, Any]]
            tool_query: Optional[str]

        # Create the graph
        workflow = StateGraph(AgentState)

        # Node to generate tool queries and decide on tools
        def generate_tool_queries(state: AgentState):
            # Create a prompt to generate tool-specific queries
            tool_query_template = """You are an intelligent financial assistant tasked with breaking down the user's query 
            into specific sub-queries for different financial analysis tools.

            User Query: {query}
            Chat History: {chat_history}

            For each available tool, generate a specific, targeted sub-query that will help gather 
            comprehensive information. If a tool is not relevant, return an empty query.

            Output a JSON with tool names as keys and their specific queries as values:
            {
                "company_overview": "Specific query for company overview",
                "peer_comparison": "Specific query for peer comparison",
                ...
            }
            """

            # Create a prompt template
            prompt = ChatPromptTemplate.from_template(tool_query_template)

            # Create a chain to generate tool queries
            chain = (
                prompt 
                | self.llm 
                | JSONOutputParser()
            )

            # Generate tool-specific queries
            tool_queries = chain.invoke({
                "query": state['query'],
                "chat_history": state.get('chat_history', [])
            })

            return {
                **state, 
                'tool_queries': tool_queries
            }

        # Node to run selected tools
        def run_tools(state: AgentState):
            # Prepare to store results
            results = []

            # Run each tool with its specific query
            for tool in self.tools:
                # Get the specific query for this tool
                tool_query = state.get('tool_queries', {}).get(tool.name, '')
                
                # Only run the tool if a query is provided
                if tool_query:
                    try:
                        # Run the tool with the specific query
                        tool_result = tool.run({
                            'query': state['query'],
                            'tool_query': tool_query
                        })
                        
                        # Add the tool name to the result for tracking
                        if isinstance(tool_result, dict):
                            tool_result['tool_name'] = tool.name
                        
                        results.append(tool_result)
                    except Exception as e:
                        results.append({
                            'tool_name': tool.name,
                            'error': str(e)
                        })

            return {
                **state,
                'results': results
            }

        # Final response generation node
        def generate_final_response(state: AgentState):
            # Combine results from different tools
            context = "\n\n".join([
                f"{result.get('tool_name', 'Unknown Tool')}: {json.dumps(result)}" 
                for result in state.get('results', [])
            ])

            # Create a comprehensive prompt
            template = """You are a financial expert synthesizing information from multiple sources.

            User Query: {query}
            Chat History: {chat_history}

            Available Context:
            {context}

            Provide a comprehensive, well-reasoned response that addresses the user's query 
            using insights from the gathered information."""

            prompt = ChatPromptTemplate.from_template(template)
            
            # Create a chain to generate the final response
            chain = (
                prompt 
                | self.tool_calling_llm 
                | StrOutputParser()
            )

            # Generate the response
            response = chain.invoke({
                "query": state['query'],
                "chat_history": state.get('chat_history', []),
                "context": context
            })

            return {"response": response}

        # Define the workflow nodes and edges
        workflow.add_node("generate_tool_queries", generate_tool_queries)
        workflow.add_node("run_tools", run_tools)
        workflow.add_node("generate_response", generate_final_response)

        # Define the edges
        workflow.add_edge("generate_tool_queries", "run_tools")
        workflow.add_edge("run_tools", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set the entry point
        workflow.set_entry_point("generate_tool_queries")

        # Compile the workflow
        return workflow.compile()

    async def process_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Process the user query using the multi-agent workflow
        """
        try:
            # Invoke the workflow
            result = await self.workflow.ainvoke({
                "query": query,
                "chat_history": chat_history or []
            })
            return result.get('response', 'Unable to generate a response.')
        except Exception as e:
            return f"An error occurred: {str(e)}"
# Example usage
async def main():
    # Initialize the assistant
    assistant = FinancialAssistant(
        llm=llm,  # From impoter file
        portfolio=None,  # You can pass a portfolio if available
        macroeconomic_data={}  # Optional macroeconomic data
    )

    # Test queries
    test_queries = [
        "What's the financial overview of ICICIBANK?",
        "Compare INFY with its competitors",
        "What's the sentiment around RELIANCE?",
        "Can you generate an equity research report for HDFCBANK?",
        "Summarize the annual report for HINDUNILVR"
    ]

    for query in test_queries:
        response = await assistant.process_query(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())