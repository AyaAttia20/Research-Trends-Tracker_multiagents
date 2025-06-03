import streamlit as st
import os
import requests
import feedparser
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import warnings

warnings.filterwarnings('ignore')

def fetch_arxiv(topic):
    url = f'http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=10'
    response = requests.get(url)
    feed = feedparser.parse(response.text)
    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "authors": [author.name for author in entry.authors]
        })
    return results

def arxiv_fetch_wrapper(input):
    if isinstance(input, dict):
        topic = input.get("topic") or input.get("subject_area") or next(iter(input.values()))
    else:
        topic = input
    return fetch_arxiv(topic)

# Streamlit UI
st.set_page_config(page_title="Research Trends Tracker", layout="wide")
st.title("🔬 Research Trends Tracker")

# Secure API Key input
api_key = st.text_input("🔑 Enter your OpenRouter API Key", type="password")
topic = st.text_input("📚 Enter a research topic", "Artificial Intelligence")
run_button = st.button("🚀 Run Analysis")

if run_button:
    if not api_key:
        st.error("Please enter your OpenRouter API Key before running.")
    else:
        with st.spinner("Running multi-agent crew..."):

            # Initialize LLM with user-supplied API key
            llm = ChatOpenAI(
                model_name="mistralai/mistral-7b-instruct",
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=api_key,
                temperature=0.7
            )

            # Tool for arXiv fetch
            arxiv_tool = Tool(
                name="ArxivResearchFetcher",
                func=arxiv_fetch_wrapper,
                description="Fetches recent research papers from arXiv given a topic string or a dict with 'topic'",
                return_direct=True
            )

            # Agents
            fetcher = Agent(
                role="Research Fetcher",
                goal="Find the latest research papers on a given topic",
                tools=[arxiv_tool],
                backstory="Expert at gathering academic papers using arXiv.",
                verbose=True,
                llm=llm
            )

            analyzer = Agent(
                role="Trend Analyzer",
                goal="Analyze recent papers and extract trending keywords and hot topics",
                verbose=True,
                backstory="Skilled in text mining and NLP to extract useful trends.",
                llm=llm
            )

            reporter = Agent(
                role="Author & Institution Reporter",
                goal="Find top authors and institutions publishing in the research field",
                verbose=True,
                backstory="Specializes in identifying the key contributors in academic fields.",
                llm=llm
            )

            # Tasks
            fetch_task = Task(
                description="Fetch recent research papers from arXiv for the topic {topic}.",
                expected_output="A list of paper titles and abstracts.",
                agent=fetcher,
                async_execution=False
            )

            trend_task = Task(
                description=(
                    "Analyze the following list of research papers (title, summary, authors) on the topic '{topic}' "
                    "and identify the top 5 trending keywords or research themes in bullet points."
                ),
                expected_output="A list of 5 trending research topics or keywords with descriptions.",
                agent=analyzer,
                context=[fetch_task],
                async_execution=False
            )

            author_task = Task(
                description=(
                    "Based on the list of paper titles, authors, and summaries for the topic {topic}, "
                    "identify the most frequently mentioned authors. Include affiliations if available."
                ),
                expected_output=(
                    "- A list of top 3–5 authors, their affiliations, and publication counts.\n"
                    "- Format: Author Name – Institution (N papers)"
                ),
                agent=reporter,
                context=[fetch_task],
                async_execution=False
            )

            # Crew
            crew = Crew(
                agents=[fetcher, analyzer, reporter],
                tasks=[fetch_task, trend_task, author_task],
                verbose=True
            )

            inputs = {"topic": topic}
            result = crew.kickoff(inputs=inputs)

            # Display results
            st.success("✅ Crew run complete!")

            st.subheader("📄 Final Report")
            st.markdown(result)

            st.subheader("📚 Papers Fetched")
            st.markdown(fetch_task.output)

            st.subheader("📈 Trend Analysis")
            st.markdown(trend_task.output)

            st.subheader("👩‍🔬 Author & Institution Report")
            st.markdown(author_task.output)
