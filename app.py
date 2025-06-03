import streamlit as st
import os
import requests
import feedparser

from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

st.title("Research Trends Tracker with Crew AI")

# 1. Input API key securely
api_key = st.text_input("Enter your OpenRouter API Key", type="password")

# 2. Input research topic
topic = st.text_input("Enter Research Topic", value="Artificial Intelligence")

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
    # Handle both string and dict input
    if isinstance(input, dict):
        topic = input.get("topic") or input.get("subject_area") or next(iter(input.values()))
    else:
        topic = input
    return fetch_arxiv(topic)

if api_key and topic:
    # Set the environment variable for OpenRouter API key
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Initialize the LLM with user API key
    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        temperature=0.7
    )

    # Setup tools and agents
    arxiv_tool = Tool(
        name="ArxivResearchFetcher",
        func=arxiv_fetch_wrapper,
        description="Fetches recent research papers from arXiv given a topic string or a dict with 'topic'",
        return_direct=True
    )

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

    fetch_task = Task(
        description=f"Fetch recent research papers from arXiv for the topic {topic}.",
        expected_output="A list of paper titles and abstracts.",
        agent=fetcher,
        async_execution=False
    )

    trend_task = Task(
        description=(
            f"Analyze the following list of research papers (title, summary, authors) on the topic '{topic}' "
            "and identify the top 5 trending keywords or research themes in bullet points.\n"
            "Data:\n"
        ),
        expected_output=(
            "- A list of 5 trending research topics or keywords with 1-sentence descriptions each.\n"
            "- Format: Bullet points."
        ),
        agent=analyzer,
        context=[fetch_task],
        async_execution=False
    )

    author_task = Task(
        description=(
            f"Based on the list of paper titles, authors, and summaries for the topic {topic}, "
            "identify the most frequently mentioned authors. "
            "If affiliations are mentioned or can be inferred, include them. "
            "Return a list of the top 3–5 authors with their affiliation (if available) and how many times they appeared."
        ),
        expected_output=(
            "- A list of top 3–5 authors, their affiliations, and publication counts.\n"
            "- Format: Author Name – Institution (N papers)"
        ),
        agent=reporter,
        context=[fetch_task],
        async_execution=False
    )

    crew = Crew(
        agents=[fetcher, analyzer, reporter],
        tasks=[fetch_task, trend_task, author_task],
        verbose=True
    )

    if st.button("Run Research Trends Analysis"):
        with st.spinner("Running Crew AI tasks..."):
            result = crew.kickoff(inputs={"topic": topic})

            st.subheader("Overall Result")
            st.write(result)

            st.subheader("Fetch Task Output")
            st.json(fetch_task.output)

            st.subheader("Trend Analyzer Output")
            st.text(trend_task.output)

            st.subheader("Author Reporter Output")
            st.text(author_task.output)

else:
    st.info("Please enter your OpenRouter API Key and research topic above.")
