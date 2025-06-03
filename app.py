import streamlit as st
import os
import requests
import feedparser
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "dummy"  
os.environ["EMBEDCHAIN_VECTORDB"] = "faiss"


# Set page config for better UI
st.set_page_config(page_title="Research Trends Tracker", layout="wide")
st.title("📊 Research Trends Tracker with CrewAI")

# 1. Input API key securely
api_key = st.text_input("Enter your OpenRouter API Key", type="password")
topic = st.text_input("Enter Research Topic", value="Artificial Intelligence")

def fetch_arxiv(topic):
    """Fetch papers from arXiv with error handling."""
    try:
        url = f'http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=10'
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTP errors
        feed = feedparser.parse(response.text)
        return [
            {
                "title": entry.title,
                "summary": entry.summary,
                "authors": [author.name for author in entry.authors]
            }
            for entry in feed.entries
        ]
    except Exception as e:
        st.error(f"arXiv fetch failed: {e}")
        return []

def arxiv_fetch_wrapper(input):
    """Handle both string and dict inputs for arXiv tool."""
    topic = input.get("topic") if isinstance(input, dict) else input
    return fetch_arxiv(topic)

if api_key and topic:
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Initialize LLM (Mistral via OpenRouter)
    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.7
    )

    # Define Tools
    arxiv_tool = Tool(
        name="ArxivResearchFetcher",
        func=arxiv_fetch_wrapper,
        description="Fetches recent research papers from arXiv.",
    )

    # Define Agents
    fetcher = Agent(
        role="Research Fetcher",
        goal=f"Find the latest papers on {topic}",
        tools=[arxiv_tool],
        backstory="Expert at gathering academic papers using arXiv.",
        verbose=True,
        llm=llm
    )

    analyzer = Agent(
        role="Trend Analyzer",
        goal="Extract trending keywords and themes",
        backstory="Skilled in text mining and NLP.",
        verbose=True,
        llm=llm
    )

    reporter = Agent(
        role="Author Reporter",
        goal="Identify top authors and institutions",
        backstory="Specializes in academic contributor analysis.",
        verbose=True,
        llm=llm
    )

    # Define Tasks
    fetch_task = Task(
        description=f"Fetch papers on '{topic}' from arXiv.",
        expected_output="List of paper titles and abstracts.",
        agent=fetcher
    )

    trend_task = Task(
        description=f"Analyze papers to identify top 5 trends in '{topic}'.",
        expected_output="Bullet list of 5 trends with 1-sentence descriptions.",
        agent=analyzer,
        context=[fetch_task]
    )

    author_task = Task(
        description=f"List top 3-5 authors in '{topic}' with affiliations.",
        expected_output="Format: Author Name – Institution (N papers)",
        agent=reporter,
        context=[fetch_task]
    )

    # Run Crew
    if st.button("🚀 Run Analysis"):
        with st.spinner("Running CrewAI tasks..."):
            crew = Crew(
                agents=[fetcher, analyzer, reporter],
                tasks=[fetch_task, trend_task, author_task],
                verbose=True
            )
            result = crew.kickoff(inputs={"topic": topic})

            # Display Results
            st.subheader("🔍 Research Trends")
            if trend_task.output:
                st.write(trend_task.output)
            else:
                st.warning("No trend analysis output generated.")

            st.subheader("👥 Top Authors")
            if author_task.output:
                st.write(author_task.output)
            else:
                st.warning("No author data found.")

            st.subheader("📄 Fetched Papers (Raw)")
            st.json(fetch_task.output or "No papers fetched.")

else:
    st.info("Please enter your OpenRouter API Key and research topic.")
