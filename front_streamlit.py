import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient
from langchain.schema.messages import HumanMessage
from langchain.prompts import PromptTemplate
from typing import List

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True  # ✅ Required to avoid SystemMessage error
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pinecone vector store
pinecone = PineconeClient(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# Session memory initialization
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, input_key="question", memory_key="chat_history"
    )
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=st.session_state.memory,
)

def needs_web_search(prompt: str) -> bool:
    prompt = prompt.lower()
    keywords = [
        "latest", "news", "current", "today", "weather", "recent", "happening", "real-time",
        "get articles", "give me articles", "fetch articles", "find articles", 
        "latest updates", "recent news", "recent updates", "any update", "recent info",
        "what's happening", "new info", "current trends"
    ]
    return any(key in prompt for key in keywords)

def rewrite_search_prompt(user_question: str, context_summary: str) -> str:
    """Use Gemini to rewrite the search query more precisely based on context."""
    rewrite_prompt = PromptTemplate.from_template("""
You are helping a user find city-specific or relocation-related news, lifestyle trends, or updates.

User's preferences and context:
{summary}

Original user question:
{question}

Rewrite the user's question to a very specific and clear web search query using the context above. Keep it concise and suitable for web search.
""")
    prompt = rewrite_prompt.format(summary=context_summary, question=user_question)
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content.strip()

def search_web(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        results = response.json()
        snippets = []
        for item in results.get("organic", [])[:5]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            snippets.append(f"{title}\n{snippet}\nURL: {link}")
        return "\n\n".join(snippets)
    else:
        return "Sorry, I couldn't fetch data from the web."

def update_summary(user_input, bot_response, max_tokens=500):
    st.session_state.history.append(f"User: {user_input}\nBot: {bot_response}")
    combined = "\n\n".join(st.session_state.history)

    summarization_prompt = PromptTemplate.from_template("""
You are an AI assistant helping a user evaluate cities for relocation.
Extract only reusable insights from the chat:
- career field and job preferences
- lifestyle needs (climate, walkability, etc.)
- family requirements (schools, safety)
- housing constraints and budget
- discussed cities or neighborhoods
- transportation concerns
- remote/hybrid work setup
- cost of living or QoL comparisons

Discard irrelevant chit-chat or generic Q&A.

Conversation:
{history}

Summarized user preferences and key context:
""")

    messages = [HumanMessage(content=summarization_prompt.format(history=combined))]
    summary_response = llm.invoke(messages)
    summary = summary_response.content.strip()

    if len(summary.split()) > max_tokens:
        summary = " ".join(summary.split()[:max_tokens])

    st.session_state.summary = summary

# Streamlit UI
st.set_page_config(page_title="City Relocation Chatbot", page_icon="🌆")
st.title("🌆 City Relocation Chatbot")
st.caption("Get help finding your ideal city — job market, housing, lifestyle, and more!")

user_input = st.chat_input("Ask me about cities, lifestyle, job prospects, or trends...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    if needs_web_search(user_input):
        with st.spinner("🔍 Searching the web with your preferences..."):
            # Step 1: Rewrite search query with context
            enriched_query = rewrite_search_prompt(user_input, st.session_state.summary)

            # Step 2: Search the web
            web_data = search_web(enriched_query)

            # Step 3: Use LLM to answer based on web and context
            if not web_data or "couldn't fetch" in web_data.lower():
                answer = "Sorry, I couldn't retrieve the latest articles. Try rephrasing or check your internet connection."
            else:
                web_prompt = (
                    "You are helping someone evaluate cities to relocate to.\n\n"
                    f"Real-time web information:\n{web_data}\n\n"
                    f"User Question: {user_input}\n\n"
                    f"Use the web data and context to answer clearly. If the question is vague, infer their intent."
                )
                response = llm.invoke([HumanMessage(content=web_prompt)])
                answer = response.content.strip()
    else:
        result = qa_chain.invoke({"question": user_input})
        answer = result["answer"]

        fallback_phrases = [
            "i don't have memory", "each interaction starts fresh",
            "i don't retain information", "i don't remember"
        ]
        if any(phrase in answer.lower() for phrase in fallback_phrases):
            repair_prompt = (
                "The user thinks you forgot earlier context.\n\n"
                f"Here's what you've learned so far:\n{st.session_state.summary}\n\n"
                f"Now answer this question using the above summary:\n\n{user_input}"
            )
            response = llm.invoke([HumanMessage(content=repair_prompt)])
            answer = response.content.strip()

    update_summary(user_input, answer)
    st.session_state.chat_history.append({"role": "bot", "text": answer})

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["text"])
