# chatbot.py

# ─────────────────────────────
# Standard Library Imports
# ─────────────────────────────
import asyncio
import re
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Tuple
from country_list import countries_for_language
import re
import json
# ─────────────────────────────
# Third-Party Libraries
# ─────────────────────────────
import nltk
import torch
from deep_translator import GoogleTranslator
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langdetect import detect
from nltk.tokenize import RegexpTokenizer, word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─────────────────────────────
# LangChain & Related Libraries
# ─────────────────────────────
#from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
#from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
#from langchain.memory import ConversationBufferMemory
#from langchain.retrievers import BM25Retriever
#from langchain_ollama import OllamaLLM
#from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_core.documents import Document
#import chromadb

from langchain_core.prompts import PromptTemplate
#from langchain import PromptTemplate
#from langchain.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.memory import ConversationBufferMemory
#from langchain.memory import ConversationBufferMemory
from langchain_classic.retrievers import BM25Retriever
#from langchain.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb
# ─────────────────────────────
# Project-Specific Imports
# ─────────────────────────────
from models import Interaction
from pred_res import PREDEFINED_RESPONSES


import logging
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from datetime import datetime

import logging
import os


nltk.download("punkt")

# ========== Config ==========
#CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT"))
CHROMA_SSL = os.getenv("CHROMA_SSL")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
INFO_COLLECTION_NAME = "information_embeddings"
VIS_COLLECTION_NAME = "visualization_embeddings"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# ========== Globals ==========
memory_sessions: Dict[str, Tuple[datetime, ConversationBufferMemory]] = {}

embedding_model = None
info_vectordb = None
info_retriever = None
llm = None
custom_prompt = None
whoswho_names = set()


def create_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )


def cleanup_expired_sessions(ttl_minutes: int = 90):
    now = datetime.now()
    expired_ids = [
        sid for sid, (created_at, _) in memory_sessions.items()
        if now - created_at > timedelta(minutes=ttl_minutes)
    ]
    for sid in expired_ids:
        del memory_sessions[sid]
        print(f"[Session Cleanup] Removed expired session: {sid}")

# ========== Initialization ==========
def _build_chroma_store(collection_name: str) -> Chroma:
    """
    Connect to the remote Chroma server; local persistence is no longer supported.
    """
    if not CHROMA_HOST:
        raise RuntimeError("CHROMA_HOST is required. Local Chroma fallback has been removed.")

    logger.info(
        "Connecting to remote Chroma at %s:%s (ssl=%s)",
        CHROMA_HOST,
        CHROMA_PORT,
        CHROMA_SSL,
    )
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=False,
        )
        if CHROMA_TENANT:
            client.set_tenant(CHROMA_TENANT)
        if CHROMA_DATABASE:
            client.set_database(CHROMA_DATABASE)
        client.heartbeat()
    except Exception as exc:
        logger.exception("Failed to connect to remote Chroma.")
        raise RuntimeError(
            f"Failed to connect to Chroma at {CHROMA_HOST}:{CHROMA_PORT}"
        ) from exc
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )


def initialize_components():
    global embedding_model, info_vectordb, info_retriever
    global  llm, custom_prompt

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    info_vectordb = _build_chroma_store(INFO_COLLECTION_NAME)

#    info_vectordb = Chroma(
#        persist_directory=CHROMA_PERSIST_DIR,
#        embedding_function=embedding_model,
#        collection_name=INFO_COLLECTION_NAME
#    )
#    info_retriever = info_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = OllamaLLM(
    model="llama3.2:3b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2, 
   )



custom_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template="""
You are **MoSPI AI**, an AI assistant for the Ministry of Statistics and Programme Implementation (MoSPI), Government of India.

Your role is to answer user queries strictly using the information provided in the context. The `context` includes documents and data retrieved by a RAG (Retrieval-Augmented Generation) pipeline. Additionally, the `chat_history` includes the last two user-chatbot interactions and should only be used for reference — not summarized or repeated.  

- **Strictly Do not mention any challenges, Concern, problems, gaps, or negative aspects** even if present in the context.  

### Guidelines:

1. **Use Context Only**: Provide answers **strictly from the context**. Do not use external or general knowledge. Do not guess. If the answer is not clearly and directly supported by the context, respond with:
   > "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."

2. **Concise Answers (Default)**  
   - Keep responses **maximum 2 lines (under 50 words)**.  
   - Never exceed this limit, even if more context is available. Summarize concisely.

3. **Officer/Personnel Queries (Who’s Who)**  
    - When the user asks about an officer/person, such as:  
        - "Who is <Name>?"  
        - "Who is <Designation> of <Division>?"  
        - "Who is <Designation> from <Division>?"  
        - "List all <Designation> from <Division>"  
        - "give me all  <Designation> from  <Department>"    

    Use the officer data available in the provided context to accurately match it with the user's query.
    ---

    ### 🧍 Single Officer Query

    - If the query clearly refers to **one specific person**, include the following fields **if present**:
    - **Name**
    - **Designation**
    - **Division**
    - **Email**
    - **Contact No**
    - **Address**

    - Present the response in a **natural, conversational** tone (not a rigid list).  

    **Example 1 – Name Query:**  
    User: *Who is <Name>?*  
    Answer:  
    > <Name> is serving as a **<Designation>** in the **<Division>**.  
    > Their office is located at *<Address>*.  
    > You can reach them at **<Email>** or call their office line at **<Contact No>**.

    **Example 2 – Designation + Division Query:**  
    User: *Who is the <Designation> from <Division>?*  
    Answer:  
    > The <Designation> from the **<Division>** is **<Name>**.  
    > Their office is at *<Address>*.  
    > You can contact them by email at **<Email>** or by phone at **<Contact Number>**.

    ---

    ### 👥 Multiple Officers Query (Plural Response)

    - If the query requests **multiple officers** (e.g., “all directors”, “list of officers”, “give me all secretaries”, “show me all officers from X department”),  
    then:
    1. Identify all matching officers from the context.  
    2. For each matching officer, include available fields: **Name**, **Designation**, **Division**, **Email**, **Contact Number**, **Address**.  
    3. Present them in a **clear, natural list style** — short paragraph per officer (not a table).  

    **Example – List Query:**  
    User: *List all <Designation> from the <Department/Division>*  
    Answer:  
    > Here are the <Designation> currently listed under the **<Department/Division>**:  
    >
    > - **<Name 1>** — Email: **<Email 1>**, Contact: **<Contact Number 1>**  
    > - **<Name 2>** — Email: **<Email 2>**, Contact: **<Contact Number 2>**  
    > - **<Name 3>** — Email: **<Email 3>**, Contact: **<Contact Number 3>**

    ---

    ### ⚠️ Important Rules
    - Do **not assume or fabricate** missing information. Only include details found and correctly match in the provided context.  
    - Do not confuse between <Designation> and <Division>. Respond strictly based on what is mentioned in the user’s query. Do not mix similar designations (e.g., Director, Deputy Director, Director General), and provide information only for the exact designation or division requested. Avoid adding or fabricating unrelated details. 
    - If no confident match is found, reply clearly:  
    > "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."  

    
4. **No Subjective Judgments or Proof Requests**  
   - If the user asks for proof, judgment, or evaluation (e.g., "India is doing well," "India is poor," "why India is poor," "why literacy is low"),  
   - Do not provide any statistics, explanations, or reasoning.  
   - Directly respond only with:  
     > "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."

5. **No Justifications or Reasons Beyond Context**  
   - If the user asks "why," "what is the reason," ,"justify," or requests causes for a problem,  
   - Do not attempt to answer. Do not provide statistics, causes, or interpretations.  
   - Directly respond only with:  
     > "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."

6. **On Data-Related Questions (e.g., domain-specific stats from MoSPI)**  
    - If the user’s query relates to MoSPI domains like Indian education, labour, employment, Sustainable Development Goals, or similar:
        - Provide the latest available year among 2024, 2025, or 2026 (whichever is present in the context) with key statistical figures clearly.
        - Do **not extrapolate, estimate, or assume** missing data.

7. **On Visualization-Related Questions**  
   - If the user’s question involves or implies a graph, chart, or visual representation, you must not attempt to generate or describe visuals.  
   - Only extract and present textual or numerical facts from the context if they are clearly available and related to query.  
   - Do **not** say anything about being a language model, or about being unable to generate visuals.

8. **On Indian Economy–Related Questions**  
   - Provide a **short summary** (1-2 lines).  
   - Do not provide challenges, limitations, reasons, justifications, or negative aspects — only neutral or positive factual information.
   - Do not explain reasons, causes, or justifications.  

9. For NSS Round–Related Questions:
    - Provide a concise, structured response of the requested NSS round strictly using the context.
    - Include only verified information:
        - Round/Survey Name
        - Year of Survey
        - Key highlights (1–2 lines)
    - Do not:
        - Fabricate or assume missing information
        - Include explanations, reasons, or interpretations
        - Mention negative aspects or gaps

10. **Tone and Structure**  
   - Keep responses short, do not elaborate, just 1–2 lines.  
   - Keep responses **factual, positive or neutral**; do not include any negative statements or challenges even if present in the context.  
   - Avoid introductory phrases like “According to the context,” “Based on the data provided,” etc. Start directly with the answer.  
   - Do not include negative opinions about  unemployment, poverty, inequality, or deficits.
   - Do not include greetings unless the user is clearly greeting you first. Do not prepend greetings for factual queries.  
   - Keep answers concise, neutral, and focused on progress, achievements, or factual data only.

11. Do not repeat the answers fetched from the previous Chat History, instead regenerate based on the use query.

---

### Chat History:
{chat_history}

### Context:
{context}

---

Now, using the above context and referring to the previous interactions (chat history should only be used for reference and **must not be repeated**), answer the following question:

Question: {question}

"""
)



def preserve_short_forms(original: str, expanded: str) -> str:
    # Extract all-uppercase words or acronyms from original
    original_acronyms = re.findall(r'\b[A-Z]{2,}\b', original)
    for acronym in original_acronyms:
        # Replace altered forms in expanded query back to original acronym
        if acronym.lower() != acronym:  # skip lowercase accidental matches
            pattern = re.compile(re.escape(acronym), re.IGNORECASE)
            expanded = pattern.sub(acronym, expanded)
    return expanded


# ========== Core Functionality ==========
def expand_query_with_llm(query: str, llm, memory: ConversationBufferMemory) -> str:
    history = memory.chat_memory.messages if memory.chat_memory.messages else []
    last_user_msg, last_ai_msg = "", ""

    if len(history) >= 2:
        for i in range(len(history) - 1, -1, -1):
            if history[i].type == "ai" and not last_ai_msg:
                last_ai_msg = history[i].content
            elif history[i].type == "human" and not last_user_msg:
                last_user_msg = history[i].content
            if last_user_msg and last_ai_msg:
                break

    # Ensure all context is in English
    translated_query = translate_to_english(query)
    translated_last_user_msg = translate_to_english(last_user_msg) if last_user_msg else ""
    translated_last_ai_msg = translate_to_english(last_ai_msg) if last_ai_msg else ""

    if translated_last_user_msg and translated_last_ai_msg:
        prompt = f"""
You are a query rewriter, not a chatbot.

🎯 Your task is to only correct the grammar of the user’s question to make it clearer and more readable, for the purpose of Ministry of Statistics and Programme Implementation (MoSPI)-related document or information retrieval
    
🔒 STRICT RULES:
- You must NOT expand or interpret abbreviations, acronyms, or short forms like "SSO", "ESD", SSD, PLFS and many more.
- DO NOT assume full forms based on common usage or general knowledge.
- DO NOT expand short forms unless they are already expanded in the current or previous question/answer.
- DO NOT fabricate names, titles, departments, dates, or data types.
- DO NOT add or infer meaning beyond what is stated.
- DO NOT greet, explain, apologize, or act like a chatbot.
- DO NOT add any extra explanation.
- Strictly DO NOT expand short forms.

✅ DO:
- Keep the rewritten query **factually aligned** and based ONLY on the given inputs.
- If a short form is used retain it as-is — without any changes, expansions, or interpretation.
- If the current question clearly continues the previous question-answer pair, use relevant references from them.
- If not, treat it as a standalone query.
- Keep acronyms or short forms exactly as provided — do NOT guess or expand them.
- If the query is unrelated to mospi domain then also just correct its grammar and spellings do not add any explaination.

Previous Question: "{translated_last_user_msg}"
Previous Answer: "{translated_last_ai_msg}"
Current Question: "{translated_query}"

Rewritten Query:"""
    else:
        prompt = f"""
You are a query rewriter, not a chatbot.

🎯 Your task is to only correct the grammar of the user’s question to make it clearer and more readable, for the purpose of Ministry of Statistics and Programme Implementation (MoSPI)-related document or information retrieval

🔒 STRICT RULES:
- You must NOT expand or interpret abbreviations, acronyms, or short forms like "SSO", "ESD", SSD, PLFS and many more.
- DO NOT assume full forms based on common usage or general knowledge.
- DO NOT expand short forms unless they are already expanded in the current or previous question/answer.
- DO NOT fabricate names, titles, departments, dates, or data types.
- DO NOT add or infer meaning beyond what is stated.
- DO NOT greet, explain, apologize, or act like a chatbot.
- DO NOT add any extra explanation.
- Strictly DO NOT expand short forms.

✅ DO:
- Keep the rewritten query **factually aligned** and based ONLY on the user input.
- If a short form is used retain it as-is — without any changes, expansions, or interpretation.
- Keep acronyms or short forms exactly as provided — do NOT guess or expand them.
- If the query is unrelated to mospi domain then also just correct its grammer and spellings do not add any explaination.
Current Question: "{translated_query}"

Rewritten Query:"""

    #print(f"[Query Expansion] Prompt to LLM:\n{prompt}")
    expanded = llm.invoke(prompt).strip()
    expanded = preserve_short_forms(translated_query, expanded)

    #print(f"[Query Expansion] Expanded Query: {expanded}")
    return expanded




# Load BAAI/bge-reranker-base
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_model.eval()

from typing import List, Tuple
#from langchain.schema import Document
from langchain_core.documents import Document

import torch
import torch.nn.functional as F
from typing import List, Tuple
#from langchain.schema import Document
from langchain_core.documents import Document

def rerank_documents(query: str, docs: List[Document], top_k: int = 7) -> List[Tuple[Document, float]]:
    pairs = [(query, doc.page_content) for doc in docs]
    
    inputs = reranker_tokenizer.batch_encode_plus(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    scored_docs = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


# Build a country set (except India)
COUNTRIES = {name.lower() for code, name in countries_for_language('en')}
COUNTRIES.discard("india")

def contains_foreign_country(query: str) -> bool:
    q = query.lower()
    for country in COUNTRIES:
        if country in q:
            return True
    return False

def get_all_documents_from_vectordb(vectordb) -> List[Document]:
    result = vectordb.get(include=["documents", "metadatas"], limit=10000)  # Adjust limit as needed
    documents = result["documents"]
    metadatas = result["metadatas"]

    # Zip and rebuild Document objects
    return [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]

def get_bm25_retriever():
    # Load all your documents
    docs = get_all_documents_from_vectordb(info_vectordb)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 10
    return bm25_retriever


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text: str) -> str:
    if detect_language(text) == "en":
        return text  # No translation needed
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_hindi(text: str) -> str:
    if detect_language(text) == "hi":
        return text  # No translation needed
    return GoogleTranslator(source='auto', target='hi').translate(text)

def is_visualization_query(query: str) -> bool:
    keywords = ["graph", "trend", "chart", "visual", "plot", "line graph", "bar chart", "pie chart","visualization","visuals",]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def clean_labels(text: str) -> str:
    patterns = [
        r"\bQ:\s*",
        r"\bA:\s*",
        r"\bAnswer:\s*",
        r"\bMoSPI AI Answer:\s*",
        '''r"(?i)\bAccording to( the)?( provided)? (context|data|information)[:,]?\s*",
        r"(?i)\bBased on( the)?( provided)? (context|data|information)[:,]?\s*",
        r"(?i)\bFrom( the)?( provided)? (context|data|information)[:,]?\s*",
        r"(?i)\bAs per( the)?( provided)? (context|data|information)[:,]?\s*",
        r"(?i)\bIn the (given|provided) (context|data|information)[:,]?\s*",
        r"(?i)\bprovided context,\s*",  # explicitly remove "provided context,"'''
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def is_fallback_response(text: str) -> bool:

    fallback_patterns = [
        # Existing...
        re.compile(r"\bno (relevant|available)?\s*information\b", re.I),
        re.compile(r"\bi (don['’]t|do not) have (any )?(info|information|details)\b", re.I),
        re.compile(r"\bnot (in|within) (my )?(knowledge|data|records)\b", re.I),
        re.compile(r"\b(unable|cannot|can't) (to )?(find|provide|locate|answer|share)\b", re.I),
        re.compile(r"\bno (data|details|record|entry)\b", re.I),
        re.compile(r"\bi('m| am)? not sure\b", re.I),
        re.compile(r"\b(outside|beyond) (my )?(scope|knowledge|ability)\b", re.I),
        re.compile(r"\b(not|no) (matching|relevant) (info|information|data)\b", re.I),
        re.compile(r"\bnot (covered|documented)\b", re.I),
        re.compile(r"\bsorry[, ]? i (don['’]t|do not) have\b", re.I),
        re.compile(r"\b(i don['’]t|i do not) understand\b", re.I),
        re.compile(r"\b(could you please|can you) rephrase\b", re.I),
        re.compile(r"\bi (couldn['’]t|cannot|can't) find (any )?(info|information|details)\b", re.I),
        re.compile(r"\bit'?s possible that\b", re.I),  
        re.compile(r"\bi couldn['’]t find any information\b", re.I),
        re.compile(r"\bnot available in the provided context\b", re.I),
        re.compile(r"\bnot mentioned in the provided context\b", re.I),
        re.compile(r"\bi (don['’]t|do not) have\b.*\b(data|information|details|info)\b", re.I)

    ]

    if not text:
        return False
    return any(pattern.search(text) for pattern in fallback_patterns)
import re

import re

def is_blocked_query(query: str) -> bool:
    """
    Check if query is asking for proof, justification, reasons,
    or statistics in a negative/degrading way about India/MoSPI.
    """
    query = query.lower()

    BLOCKED_PATTERNS = [
        # Poverty / Poor
        r"\bwhy\s+is\s+india\s+poor\b",
        r"\bwhy\s+india\s+is\s+poor\b",
        r"why.*india.*poor",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(poor)",

        # Unemployment
        r"\bwhy\s+is\s+there\s+high\s+unemployment\b",
        r"\bwhy\s+is\s+unemployment\s+high\b",
        r"why.*india.*unemployment",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(unemployment)",

        # Literacy
        r"\bwhy\s+india\s+has\s+low\s+literacy\s+rate\b",
        r"\bwhy\s+is\s+literacy\s+low\s+in\s+india\b",
        r"why.*literacy.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(literacy|low)",

        # Poverty (general wording)
        r"\bwhy\s+poverty\s+is\s+high\s+in\s+india\b",
        r"why.*poverty.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(poverty).*(india)",

        # General degrading
        r"\bwhy\s+is\s+india\s+not\s+doing\s+well\b",
        r"\bwhy\s+does\s+india\s+lag\b",
        r"why.*india.*lag",
        r"why.*india.*not doing well",
        r"justify.*india",
        r"(statistics|stats|data).*(prove|proves|show|justify).*(india).*(not\s+doing\s+well|lag|failure)",
    ]

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query):
            return True
    return False


from sentence_transformers import SentenceTransformer, util
economy_model = SentenceTransformer("all-MiniLM-L6-v2")
# ─────────────────────────────
# Economy KPI values
# ─────────────────────────────
ECONOMY_KPIS = {
    "gdp": "**GDP Growth**: 7.8% (Q1, 2025-26)",
    "iip": "**Index of Industrial Production (IIP)**: 4.0% (August 2025)",
    "cpi": "**Inflation (CPI)**: 1.54% (September 2025)",
    "unemployment": "**Urban Unemployment Rate**: 5.2% (September 2025)"
}

# ─────────────────────────────
# KPI keyword mapping (expanded)
# ─────────────────────────────
KPI_KEYWORDS = {
    "gdp": [
        "gdp", "gross domestic product", "india gdp", "indian gdp", 
        "gdp growth", "gdp rate", "india gdp growth", "gdp of india"
    ],
    "iip": [
        "iip", "industrial production", "index of industrial production", 
        "industrial output", "industrial growth", "iip rate","iip growth rate"
    ],
    "cpi": [
        "cpi", "inflation", "consumer price index", "price rise", 
        "inflation in india", "current inflation", "inflation rate", "price increase"
    ],
    "unemployment": [
        "unemployment", "jobless rate", "urban unemployment", 
        "employment rate", "unemployment rate", "joblessness", "unemployment in india"
    ]
}

# ─────────────────────────────
# General economy keywords (expanded)
# ─────────────────────────────
GENERAL_ECONOMY_KEYWORDS = [
    "indian economy", "india economy", "economic growth", "economic health", 
    "economic condition", "progress of indian economy", "financial condition", 
    "latest economic growth", "current economic condition", "economic update", 
    "state of economy", "economic situation", "india economic report", 
    "recent economic growth", "india economic performance","economical outlook","financial outlook"
]


def get_kpi_by_keyword(query: str) -> str | None:
    """
    Return KPI response based on keyword matching.
    1️⃣ Specific KPI match → only matched KPIs
    2️⃣ General economy match → all KPIs
    3️⃣ No match → None
    """
    query_lower = query.lower()

    # Specific KPI match
    matched_kpis = [kpi for kpi, kws in KPI_KEYWORDS.items() if any(kw in query_lower for kw in kws)]
    if matched_kpis:
        kpi_lines = [ECONOMY_KPIS[kpi] for kpi in matched_kpis]
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(kpi_lines)

    # General economy match → all KPIs
    if any(kw in query_lower for kw in GENERAL_ECONOMY_KEYWORDS):
        return "📊 **Latest Indian Economy KPIs**\n\n" + "\n".join(ECONOMY_KPIS.values())

    return None


def fallback_response(is_hindi: bool = False) -> str:
    """Return generic fallback response."""
    return (
        "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"
        if is_hindi else
        "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding."
    )


def safe_subjective_response(is_hindi: bool = False) -> str:
    """Return a safe message for subjective queries along with KPI data."""
    neutral_response = (
        "As an AI assistant, I don’t provide opinions. Here is the latest official data:\n\n"
        if not is_hindi else
        "मैं किसी राय नहीं दे सकता। यहाँ नवीनतम आधिकारिक डेटा है:\n\n"
    )
    return neutral_response + "\n".join(ECONOMY_KPIS.values())


def is_economy_related(query: str) -> bool:
    """Check if the query is related to the Indian economy."""
    query_lower = query.lower()
    return bool(
        any(kw in query_lower for kws in KPI_KEYWORDS.values() for kw in kws) or
        any(kw in query_lower for kw in GENERAL_ECONOMY_KEYWORDS)
    )


SUBJECTIVE_KEYWORDS = [
    "what is your opinion on indian economy",
    "is india doing good or bad in economy",
    "how do you rate india economy",
    "what do you think about indian economy",
    "rate india economy", "your opinion on india economy"
]

def is_subjective_query(query: str) -> bool:
    """Check if query is opinion-based using keywords."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in SUBJECTIVE_KEYWORDS)

# Negative / degrading queries → keyword-based
# ─────────────────────────────
NEGATIVE_KEYWORDS = [
    "why india is poor", "india economy is weak", "india has high unemployment",
    "india has low literacy rate", "why india is behind", "why india is failing",
    "prove india is poor", "why india is not doing well", "justify india economy is weak",
    "india is poor", "india is failing", "weak economy"
]

def is_negative_query(query: str) -> bool:
    """Check if query is negative/degrading about India using keywords."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in NEGATIVE_KEYWORDS)



# For NSS Round Quries

import re


def is_nss_round_query(query: str) -> bool:
    q = query.lower()
    pattern = (
        r"\b("
        r"(nss|nss\s+survey|national\s+sample\s+survey)\s*\d+\s*(st|nd|rd|th)?\s*round|"
        r"\d+\s*(st|nd|rd|th)?\s*(nss|nss\s+survey|national\s+sample\s+survey)\s*round"
        r")\b"
    )
    result = bool(re.search(pattern, q))
    logger.debug(f"is_nss_round_query('{query}') -> {result}")
    return result


def extract_nss_round(query: str) -> str | None:
    q = query.lower()
    patterns = [
        r"(?:nss|nss\s+survey|national\s+sample\s+survey)\s*(\d+)\s*(st|nd|rd|th)?\s*round",
        r"(\d+)\s*(st|nd|rd|th)?\s*(?:nss|nss\s+survey|national\s+sample\s+survey)\s*round",
        r"(?:nss|nss\s+survey|national\s+sample\s+survey)\s*survey\s*round\s*(\d+)"
    ]
    for pat in patterns:
        match = re.search(pat, q)
        if match:
            logger.debug(f"extract_nss_round('{query}') -> {match.group(1)}")
            return match.group(1)
    logger.debug(f"extract_nss_round('{query}') -> None")
    return None


def get_nss_documents() -> list[str]:
    all_docs = list_all_documents()
    if isinstance(all_docs, dict) and "error" in all_docs:
        logger.warning("get_nss_documents: failed to list documents")
        return []

    nss_docs = [doc for doc in all_docs if re.search(r"\b\d+(st|nd|rd|th)?\s*round\b", doc.lower())]
    logger.info(f"get_nss_documents: found {len(nss_docs)} NSS documents")
    return sorted(nss_docs)


def get_nss_docs_for_round(round_no: str) -> list[str]:
    nss_docs = get_nss_documents()
    pattern = fr"\b{round_no}(st|nd|rd|th)?\s*round\b"
    matched_docs = [doc for doc in nss_docs if re.search(pattern, doc.lower())]
    logger.info(f"get_nss_docs_for_round({round_no}): found {len(matched_docs)} documents")
    return matched_docs


async def handle_nss_round_query(query: str, retriever, memory, session_id: str, top_k: int = 5):
    logger.info(f"handle_nss_round_query: received query '{query}'")
    round_no = extract_nss_round(query)
    if not round_no:
        fallback_msg = "I could not determine which NSS round you are asking about."
        logger.warning(f"handle_nss_round_query: {fallback_msg}")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(fallback_msg)
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")

    matched_docs = get_nss_docs_for_round(round_no)
    if not matched_docs:
        fallback_msg = f"No documents found for NSS {round_no}th round."
        logger.warning(f"handle_nss_round_query: {fallback_msg}")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(fallback_msg)
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")

    logger.info(f"handle_nss_round_query: fetching context from {len(matched_docs)} docs")
    contact_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    ).invoke(query, filter={"doc_name": {"$in": matched_docs}})

    if not contact_docs:
        fallback_msg = f"I could not retrieve context from NSS {round_no}th round documents."
        logger.warning(f"handle_nss_round_query: {fallback_msg}")
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(fallback_msg)
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")

    context = "\n\n".join([doc.page_content for doc in contact_docs])
    chat_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in memory.chat_memory.messages[-4:]])
    full_prompt = custom_prompt.format(context=context, question=query, chat_history=chat_history)

    async def generate_stream() -> AsyncGenerator[str, None]:
        async for token in stream_llm_response(full_prompt):
            yield token

    logger.info(f"handle_nss_round_query: streaming response for NSS {round_no}th round")
    return StreamingResponse(generate_stream(), media_type="text/markdown")




import re

def is_contact_query(query: str) -> bool:
    """
    Detects if the query is about 'who is head/incharge/contact/leads' for a division/unit.
    """
    query_lower = query.lower()

    patterns = [
        r"\bwho\s+is\s+(the\s+)?(head|chief|in[- ]?charge|lead)\b",
        r"\bwho\s+leads\b",
        r"\bwho\s+heads\b",
        r"\bwho\s+is\s+(the\s+)?contact\b",
        r"\bwho\s+should\s+i\s+contact\b",
        r"\bwho\s+is\s+(the\s+)?in[- ]?charge\b",
    ]

    return any(re.search(p, query_lower) for p in patterns)




def get_contact_us_chunks(query: str, retriever, top_k: int = 5):
    """
    Retrieves only Contact_Us.json chunks when query is about head/incharge/contact.
    """
    # Force retrieval from only Contact_Us.json
    contact_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    ).invoke(query, filter={"doc_name": "Contact_Us.json"})

    return contact_docs


async def handle_contact_query(query: str, retriever, memory, session_id: str):
    """
    Handles contact-us type queries exclusively using Contact_Us.json data.
    """
    contact_chunks = get_contact_us_chunks(query, retriever)

    if not contact_chunks:
        fallback_msg = "I could not find contact information for that division/unit."
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(fallback_msg)
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")

    # Prepare context only from Contact_Us.json
    context = "\n\n".join([doc.page_content for doc in contact_chunks])
    chat_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in memory.chat_memory.messages[-4:]])
    full_prompt = custom_prompt.format(context=context, question=query, chat_history=chat_history)

    async def generate_stream() -> AsyncGenerator[str, None]:
        async for token in stream_llm_response(full_prompt):
            yield token

    return StreamingResponse(generate_stream(), media_type="text/markdown")


from typing import Set

from typing import Set
import re
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

# -----------------------------
# 1️⃣ Load officer names
# -----------------------------
def load_whoswho_names(whoswho_chunks) -> Set[str]:
    """
    Extracts all officer names from Who's who.json chunks for fast lookup.
    """
    names = set()
    for chunk in whoswho_chunks:
        match = re.search(r"name:\s*([^,]+),", chunk.page_content, re.IGNORECASE)
        if match:
            name = match.group(1).strip().lower()
            names.add(name)
    print(f"✅ Loaded {len(names)} officer names: {names}")
    return names


# -----------------------------
# 2️⃣ Detect Who's Who type query
# -----------------------------
import re
from typing import Set
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Load once globally (for efficiency)
# -----------------------------
who_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example semantic intent patterns for Who's Who queries
WHOSWHO_INTENT_EXAMPLES = [
    "who is the officer responsible for",
    "list of officers in ministry",
    "show me all directors from department",
    "who are the secretaries in the ministry",
    "give me names of officials working under",
    "officers from ministry of statistics",
    "personnel list of department",
    "who reports to secretary",
    "list of joint secretaries from MoSPI",
    "heads of divisions and sections",
]

# -----------------------------
# Extended Regex Patterns
# -----------------------------
BASIC_PATTERNS = [
    r"\bwho\s+is\b",
    r"\bwho\s+holds\b",
    r"\bwho\s+reports\s+to\b",
    r"\bwho\s+is\s+responsible\s+for\b",
]

EXTENDED_PATTERNS = [
    # action + officer words
    r"\b(list|show|give|provide|display|fetch)\b.*\b(officer|official|personnel|employee|director|secretary|head|chief|advisor|minister|chairman|member|officers)\b",
    # department-based
    r"\bfrom\s+(ministry|department|division|section|office|bureau|cell|wing)\b",
    # designation-based
    r"\b(all|list\s+of|who\s+are\s+the|names\s+of)\b.*\b(director|secretary|advisor|officer|joint\s+secretary|deputy\s+secretary|chairman|chief)\b",
    # working relationships
    r"\b(who\s+(works|reports)\s+(under|to|in|for))\b",
]


# -----------------------------
# Semantic similarity check
# -----------------------------
def semantic_whoswho_check(query: str, threshold: float = 0.55) -> bool:
    """
    Uses embedding similarity to detect if query is semantically related to 'Who's Who' queries.
    """
    query_emb = who_model.encode(query, convert_to_tensor=True)
    examples_emb = who_model.encode(WHOSWHO_INTENT_EXAMPLES, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, examples_emb)
    max_score = float(scores.max())

    print(f"🔍 Semantic similarity score: {max_score:.2f}")
    return max_score >= threshold


# -----------------------------
# Main detection function
# -----------------------------
def is_whoswho_query(query: str, whoswho_names: Set[str]) -> bool:
    """
    Determines if a query is related to 'Who's Who' information using
    (1) regex patterns, (2) officer name detection, and (3) embedding similarity.
    """
    query_lower = query.lower().strip()

    # 1️⃣ Regex and keyword detection
    if any(re.search(p, query_lower) for p in BASIC_PATTERNS + EXTENDED_PATTERNS):
        print("✅ Matched Who's Who pattern via regex.")
        return True

    # 2️⃣ Known officer name detection
    for name in whoswho_names:
        if name in query_lower:
            print(f"✅ Query matches known officer name: {name}")
            return True

    # 3️⃣ Semantic similarity check (embedding-based)
    if semantic_whoswho_check(query_lower):
        print("✅ Detected Who's Who intent via semantic similarity.")
        return True

    print("❌ Not detected as Who's Who query.")
    return False



# -----------------------------
# 3️⃣ Retrieve chunks from Who's who.json
# -----------------------------
def get_whoswho_chunks(query: str, retriever, top_k: int = 10):
    print(f"🔄 Retrieving chunks for query: '{query}' with top_k={top_k}")
    whoswho_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    ).invoke(query, filter={"doc_name": "Who's Who.json"})

    
    print(f"📄 Retrieved {len(whoswho_docs)} chunks")
    #for i, doc in enumerate(whoswho_docs, start=1):
       # print(f"Chunk {i}: {doc.page_content[:200]}...")  # show first 200 chars
    return whoswho_docs




# -----------------------------
# 4️⃣ Handle Who's Who query
# -----------------------------
async def handle_whoswho_query(query: str, retriever, memory, session_id: str):
    whoswho_chunks = get_whoswho_chunks(query, retriever)

    if not whoswho_chunks:
        fallback_msg = "I could not find information about this officer/person."
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(fallback_msg)
        return StreamingResponse(iter([fallback_msg]), media_type="text/plain")

    # ------------------------------------------
    # 🧩 Parse officers from chunks
    # ------------------------------------------
    officers = []
    for doc in whoswho_chunks:
        try:
            data = json.loads(doc.page_content)
            if isinstance(data, dict):
                officers.append(data)
            elif isinstance(data, list):
                officers.extend(data)
        except Exception:
            officers.append({"raw": doc.page_content})

    # Combine all officer data into one context
    if officers:
        print(f"✅ Parsed {len(officers)} officers from chunks.")
        context = "\n".join([json.dumps(o, ensure_ascii=False) for o in officers])
    else:
        print("⚠️ No structured officers found — using raw chunks.")
        context = "\n\n".join([doc.page_content for doc in whoswho_chunks])

    # ------------------------------------------
    # 🧠 Continue existing logic
    # ------------------------------------------
    chat_history = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in memory.chat_memory.messages[-4:]
    ])
    full_prompt = custom_prompt.format(context=context, question=query, chat_history="")

    memory.chat_memory.add_user_message(query)

    async def generate_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        buffer = ""
        try:
            async for token in stream_llm_response(full_prompt):
                buffer += token
                cleaned_buffer = clean_labels(buffer)
                full_response = cleaned_buffer

            memory.chat_memory.add_ai_message(full_response)

            # Add source + save interaction
            sources = ["Who's Who"]
            interaction = Interaction(
                session_id=session_id,
                timestamp=datetime.now(),
                query=query,
                response=full_response,
                sources=sources,
            )
            await interaction.insert()
            interaction_id = str(interaction.id)

            yield full_response
            yield "\n\n📄 **Sources:** Who's Who"
            yield f"\n\n🆔 Interaction ID: {interaction_id}"

        except Exception:
            logger.exception(f"[Session {session_id}] ❌ Error generating Who's Who response.")
            yield "\n\n❌ Error generating response."

    return StreamingResponse(generate_stream(), media_type="text/markdown")




def retrieve_priority_chunks(
    query, retriever,
    top_n_json=7,
    top_n_text=10,
    top_n_table=10,
    top_n_bm25=10,
    final_top_k=10,
    top_n_url=3
):
    # Step 1: Vector search - JSON
    json_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_n_json}
    ).invoke(query, filter={"chunk_type": "json"})

    # Step 2: Vector search - URL
    url_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_n_url}
    ).invoke(query, filter={"chunk_type": "url"})

    # Step 3: Vector search - TEXT
    text_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_n_text}
    ).invoke(query, filter={"chunk_type": "text"})

    # Step 4: Vector search - TABLE
    table_docs = retriever.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_n_table}
    ).invoke(query, filter={"chunk_type": "table"})

    # Step 5: Visualization docs (optional)
    visualization_docs = []
    if is_visualization_query(query):
        visualization_docs = retriever.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        ).invoke(query, filter={"chunk_type": "visualization"})

    # Combine vector results
    vector_results = json_docs + url_docs + text_docs + table_docs

    # Step 6: BM25 retrieval
    bm25_retriever = get_bm25_retriever()
    bm25_docs = bm25_retriever.invoke(query)

    # Step 7: Merge + deduplicate
    seen = set()
    combined_docs = []
    for doc in vector_results + bm25_docs:
        unique_id = (doc.metadata.get("chunk_id"), doc.metadata.get("doc_name"))
        if unique_id not in seen:
            seen.add(unique_id)
            combined_docs.append(doc)

    # Step 8: Re-rank and return with scores
    reranked_with_scores = rerank_documents(query, combined_docs, top_k=final_top_k)

    return reranked_with_scores, visualization_docs if visualization_docs else None



def run_llm_stream(prompt):
    for token in llm._stream(prompt):
        yield token

async def stream_llm_response(prompt) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def producer():
        try:
            for token in run_llm_stream(prompt):
                text = token.text if hasattr(token, "text") else token
                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    # Run producer in a background thread
    import threading
    threading.Thread(target=producer).start()

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token




# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logging.basicConfig(
    filename="logs/chatbot.log",          # log file
    filemode="a",                         # append mode
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO                    # set default level
)

logger = logging.getLogger("chatbot")


# Handle Question
# ─────────────────────────────
async def handle_question(request):
    session_id = request.session_id
    original_query = request.query.strip()

    # ─────────────────────────────
    # Predefined responses
    # ─────────────────────────────
    normalized_query = original_query.lower()
    if normalized_query in PREDEFINED_RESPONSES:
        predefined_response = PREDEFINED_RESPONSES[normalized_query]
        if session_id in memory_sessions:
            _, memory = memory_sessions[session_id]
            memory.chat_memory.add_user_message(original_query)
            memory.chat_memory.add_ai_message(predefined_response)

        def stream_predefined():
            for word in predefined_response.split():
                yield word + " "

        logger.info(f"[Session {session_id}] Returned predefined response for query: {original_query}")
        return StreamingResponse(stream_predefined(), media_type="text/plain")

    # ─────────────────────────────
    # Session validation
    # ─────────────────────────────
    if session_id not in memory_sessions:
        logger.warning(f"[Session {session_id}] Invalid session_id.")
        raise HTTPException(status_code=404, detail="Invalid session_id. Please create a new session.")

    is_hindi = detect_language(original_query) == "hi"
    translated_query = translate_to_english(original_query)

    _, memory = memory_sessions[session_id]
    expanded_query = expand_query_with_llm(translated_query, llm, memory)

    # ─────────────────────────────
    # Regex block / foreign country check
    # ─────────────────────────────
    if is_blocked_query(expanded_query) or contains_foreign_country(expanded_query):
        fallback_final = fallback_response(is_hindi)
        logger.warning(f"[Session {session_id}] 🚫 Blocked or foreign-country query → Fallback.")

        memory.chat_memory.add_user_message(original_query)
        memory.chat_memory.add_ai_message(fallback_final)

        try:
            interaction = Interaction(
                session_id=session_id,
                timestamp=datetime.now(),
                query=original_query,
                response=fallback_final,
                sources=[],
            )
            await interaction.insert()
        except Exception as e:
            logger.error(f"[Session {session_id}] Failed to store fallback interaction: {e}", exc_info=True)

        return StreamingResponse(iter([fallback_final]), media_type="text/plain")

    # ─────────────────────────────
    # Contact Us Handling
    # ─────────────────────────────
    if is_contact_query(expanded_query):
        logger.info(f"[Session {session_id}] Detected Contact-Us type query.")
        return await handle_contact_query(original_query, info_vectordb, memory, session_id)

    # ─────────────────────────────
    # NSS  Handling
    # ─────────────────────────────
    if is_nss_round_query(expanded_query):
        logger.info(f"[Session {session_id}] Detected NSS type query.")
        return await handle_nss_round_query(original_query, info_vectordb, memory, session_id)
        
    # ─────────────────────────────
    # whoswho  Handling
    # ─────────────────────────────
    # Updated handling in handle_question
    if is_whoswho_query(expanded_query, whoswho_names):
        logger.info(f"[Session {session_id}] Detected Who's Who type query.")
        return await handle_whoswho_query(expanded_query, info_vectordb, memory, session_id)



    # ─────────────────────────────
    # Economy-specific handling (keyword-based)
    # ─────────────────────────────
    extra_economy_response = None
    extra_subjective_response = None

    if is_economy_related(expanded_query):

        # 1️⃣ Negative economy queries → fallback
        if is_negative_query(expanded_query):
            fallback_final = fallback_response(is_hindi)
            logger.warning(f"[Session {session_id}] 🚫 Negative economy query detected → Fallback.")

            memory.chat_memory.add_user_message(original_query)
            memory.chat_memory.add_ai_message(fallback_final)

            try:
                interaction = Interaction(
                    session_id=session_id,
                    timestamp=datetime.now(),
                    query=original_query,
                    response=fallback_final,
                    sources=[],
                )
                await interaction.insert()
            except Exception as e:
                logger.error(f"[Session {session_id}] Failed to store negative-economy interaction: {e}", exc_info=True)

            return StreamingResponse(iter([fallback_final]), media_type="text/plain")

        # 2️⃣ Subjective economy queries → safe disclaimer + all KPIs
        if is_subjective_query(expanded_query):
            extra_subjective_response = safe_subjective_response(is_hindi)
            logger.info(f"[Session {session_id}] ℹ️ Subjective economy query detected → Appending safe disclaimer.")

        # 3️⃣ Keyword-based KPI response
        extra_economy_response = get_kpi_by_keyword(expanded_query)

    
    logger.info(f"[Session {session_id}] No economy/KPI match → falling back to retrieval pipeline.")

    logger.info("************* Query Logs *************")
    logger.info(f"🔹 Original Query: {original_query}")
    logger.info(f"🔹 Translated Query: {translated_query}" if is_hindi else "🔹 Translated Query: [English Input]")
    logger.info(f"🔹 Expanded Query: {expanded_query}")
    logger.info("**************************************")

    reranked_with_scores, visualization_docs = retrieve_priority_chunks(expanded_query, info_vectordb)
    visualization_docs = visualization_docs or []

    reranked_docs, scores = zip(*reranked_with_scores) if reranked_with_scores else ([], [])

    # === Hybrid Decision Logic ===
    SIMILARITY_THRESHOLD = 0.2
    has_good_text_match = bool(scores) and max(scores) >= SIMILARITY_THRESHOLD
    has_visualization = bool(visualization_docs)

    if has_visualization and not has_good_text_match:
        visualization_data = [
            doc.metadata.get("embed_code")
            for doc in visualization_docs
            if doc.metadata.get("category") == "visualization" and doc.metadata.get("embed_code")
        ] or []

        if visualization_data:
            vis_msg = (
                "यहाँ आपके प्रश्न से संबंधित एक दृश्य प्रतिनिधित्व प्रस्तुत किया गया है।"
                if is_hindi else
                "Here is a visual representation related to your query:"
            )
            output_response = vis_msg + "\n\n📊 **Visualizations:**\n" + "\n".join(visualization_data)
            memory.chat_memory.add_user_message(original_query)
            memory.chat_memory.add_ai_message(output_response)

            try:
                interaction = Interaction(
                    session_id=session_id,
                    timestamp=datetime.now(),
                    query=original_query,
                    response=output_response,
                    sources=["[Visualizations Only]"],
                )
                await interaction.insert()
                logger.info(f"[Session {session_id}] Stored visualization-only interaction.")
            except Exception as e:
                logger.error(f"[Session {session_id}] Failed to store interaction: {e}", exc_info=True)

            logger.info(f"[Session {session_id}] Final Answer: {output_response}")
            return StreamingResponse(iter([output_response]), media_type="text/plain")

    docs = list(reranked_docs)

    visualization_data = [
        doc.metadata.get("embed_code")
        for doc in visualization_docs
        if doc.metadata.get("category") == "visualization" and doc.metadata.get("embed_code")
    ] or []

    context = "\n\n".join([doc.page_content for doc in docs])
    chat_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in memory.chat_memory.messages[-4:]])
    full_prompt = custom_prompt.format(context=context, question=expanded_query, chat_history=chat_history)
    #print("Full Prompt:"  ,full_prompt)
    logger.info("************* Chat History *************")
    logger.info(chat_history)
    logger.info("\n=== Selected Context Chunks for LLM ===")
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n🔹 Chunk {i}")
        logger.info(f"📄 Source: {doc.metadata.get('doc_name') or doc.metadata.get('url', 'N/A')}")
        logger.info(f"📁 Category: {doc.metadata.get('category', 'unknown')}")
        logger.info(f"🆔 Chunk ID: {doc.metadata.get('chunk_id')}")
        logger.info(f"📝 Content: {doc.page_content}")
    logger.info("=======================================")

    memory.chat_memory.add_user_message(original_query)

    async def generate_streamed_response() -> AsyncGenerator[str, None]:
        logger.info(f"[Session {session_id}] 🚀 Starting LLM streaming...")
        full_response = ""
        buffer = ""

        try:
            # 👇 Collect everything first (don’t yield yet)
            async for text in stream_llm_response(full_prompt):
                buffer += text
         
                cleaned_buffer = clean_labels(buffer)
                full_response = cleaned_buffer
  

            fallback_final_en = (
                "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. "
                "Thank you for your understanding."
            )
            fallback_final_hi = (
                "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"
            )

            # 👇 Fallback check AFTER full response is collected
            fallback_detected = (
                is_fallback_response(full_response)
                or full_response.strip() in [fallback_final_en, fallback_final_hi]
            )

            if fallback_detected:
                fallback_final = fallback_final_hi if is_hindi else fallback_final_en
                logger.warning(f"[Session {session_id}] 🚫 Fallback triggered, suppressing original content.")
                yield fallback_final   # 👈 only fallback goes to UI
                memory.chat_memory.add_ai_message(fallback_final)

                # Save fallback interaction
                try:
                    interaction = Interaction(
                        session_id=session_id,
                        timestamp=datetime.now(),
                        query=original_query,
                        response=fallback_final,
                        sources=[],
                    )
                    await interaction.insert()
                except Exception as e:
                    logger.error(f"[Session {session_id}] Failed to store fallback interaction: {e}", exc_info=True)
                return

            # 👇 If no fallback, now stream actual response to UI
            if is_hindi:
                full_response = translate_to_hindi(full_response)

            memory.chat_memory.add_ai_message(full_response)

            # Append economy KPI response
            if extra_economy_response:
                yield extra_economy_response + "\n\n"

            # Append subjective disclaimer
            if extra_subjective_response:
                yield "\n\n" + extra_subjective_response + "\n\n"


            # Yield final answer text
            yield full_response


            # Append visualizations
            if visualization_data:
                yield "\n\n📊 **Visualizations:**\n" + "\n".join(visualization_data)

            # Append sources
            if docs:
                doc_names = {
                    doc.metadata.get("doc_name") or doc.metadata.get("url", "")
                    for doc in docs
                }

                clean_doc_names = set()
                for name in doc_names:
                    base_name, ext = os.path.splitext(name)
                    if ext.lower() == ".json":
                        clean_doc_names.add("MOSPI FAQs")
                    else:
                        clean_doc_names.add(base_name)

                yield "\n\n📄 **Sources:**\n" + "\n".join(f"- {name}" for name in clean_doc_names)

            # Save interaction
            try:
                interaction = Interaction(
                    session_id=session_id,
                    timestamp=datetime.now(),
                    query=original_query,
                    response=full_response,
                    sources=list(doc_names) if docs else [],
                )
                await interaction.insert()
                interaction_id = str(interaction.id)
                logger.info(f"[Session {session_id}] Stored full interaction with ID {interaction_id}")

                # ✅ Yield the interaction_id so frontend can use it
                yield f"\n\n🆔 Interaction ID: {interaction_id}"

                logger.info(f"[Session {session_id}] Stored full interaction.")
            except Exception as e:
                logger.error(f"[Session {session_id}] Failed to store interaction: {e}", exc_info=True)

            logger.info(f"[Session {session_id}] Final Answer: {full_response}")

        except Exception:
            logger.exception(f"[Session {session_id}] ❌ Error generating response.")
            yield "\n\n❌ Error generating response."

    return StreamingResponse(generate_streamed_response(), media_type="text/markdown")



def list_all_documents():
    try:
        results = info_vectordb._collection.get(include=["metadatas"])
        doc_names = {
            meta.get("doc_name")
            for meta in results["metadatas"]
            if meta.get("category") == "doc" or meta.get("doc_name")
        }
        return sorted(filter(None, doc_names))  # Remove None values
    except Exception as e:
        return {"error": f"Failed to list document names: {str(e)}"}


def get_chunks_for_doc(doc_name: str):
    try:
        results = info_vectordb._collection.get(include=["documents", "metadatas", "embeddings"])
        chunks = [
            {
                "chunk_id": metadata.get("chunk_id"),
                "doc_id": metadata.get("doc_id"),
                "content": doc,
                "chunk_type": metadata.get("chunk_type", "text"),  # default to text if not present
                "uploaded_at": metadata.get("uploaded_at")  # added date field
            }
            for doc, metadata in zip(results["documents"], results["metadatas"])
            if metadata.get("doc_name") == doc_name
        ]
        return {
            "document_name": doc_name,
            "total_chunks": len(chunks),
            "chunks": chunks
        } if chunks else {"message": f"No chunks found for document: {doc_name}"}
    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}


def delete_chunks_for_doc(doc_names):

    if isinstance(doc_names, str):
        doc_names = [doc_names]  # convert single string to list

    summary = {}
    for doc_name in doc_names:
        try:
            results = info_vectordb._collection.get(include=["metadatas"], where={"doc_name": doc_name})

            if not results["ids"]:
                summary[doc_name] = {"message": f"No chunks found for document: {doc_name}"}
                continue

            # Delete using actual vector DB IDs
            info_vectordb._collection.delete(ids=results["ids"])

            summary[doc_name] = {
                "message": f"Deleted {len(results['ids'])} chunks for document: {doc_name}",
                "deleted_ids": results["ids"]
            }

        except Exception as e:
            summary[doc_name] = {"error": f"Failed to delete chunks: {str(e)}"}

    return summary



def list_all_urls():
    try:
        results = info_vectordb._collection.get(include=["metadatas"])
        urls = {
            meta.get("url")
            for meta in results["metadatas"]
            if meta.get("category") == "url" and meta.get("url")
        }
        return sorted(urls)
    except Exception as e:
        return {"error": f"Failed to list URLs: {str(e)}"}

def get_chunks_for_url(url: str):
    """
    Fetch all chunks stored in the vector DB for a given URL.
    """
    try:
        results = info_vectordb._collection.get(include=["documents", "metadatas"])
        chunks = [
            {
                "chunk_id": metadata.get("chunk_id"),
                "url": metadata.get("url"),
                "page_title": metadata.get("page_title"),
                "content": doc
            }
            for doc, metadata in zip(results["documents"], results["metadatas"])
            if metadata.get("url") == url and metadata.get("category") == "url"
        ]

        return {
            "url": url,
            "total_chunks": len(chunks),
            "chunks": chunks
        } if chunks else {"message": f"No chunks found for URL: {url}"}

    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}


import re
from datetime import datetime
from typing import List, Optional, Dict, Any

# Predefined domain list
VALID_DOMAINS = ["NSS Rounds", "GDP", "CPI", "IIP"]

def update_doc_metadata(
    doc_names: List[str], 
    uploaded_at: Optional[str] = None,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update metadata for all chunks belonging to given document names.
    - If `uploaded_at` not provided → use current UTC time.
    - If only a date (YYYY-MM-DD) is given → auto-set time to 00:00:00.
    - If full datetime is given → use as is.
    - If `domain` is provided → update domain metadata for all chunks (must be in VALID_DOMAINS).
    """

    # Validate domain
    if domain and domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain. Choose from: {', '.join(VALID_DOMAINS)}")

    # Process uploaded_at timestamp
    if uploaded_at:
        try:
            if re.match(r"^\d{4}-\d{2}-\d{2}$", uploaded_at):
                timestamp = datetime.fromisoformat(uploaded_at + "T00:00:00").isoformat()
            else:
                timestamp = datetime.fromisoformat(uploaded_at).isoformat()
        except ValueError:
            raise ValueError("Invalid datetime format. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
    else:
        timestamp = datetime.now().isoformat()

    updated_docs = {}

    for doc_name in doc_names:
        results = info_vectordb._collection.get(
            include=["metadatas"],
            where={"doc_name": doc_name}
        )

        if not results["ids"]:
            updated_docs[doc_name] = "No chunks found"
            continue

        metadata_update = {"uploaded_at": timestamp}
        if domain:
            metadata_update["domain"] = domain

        info_vectordb._collection.update(
            ids=results["ids"],
            metadatas=[metadata_update for _ in results["ids"]]
        )

        updated_docs[doc_name] = {
            "updated_chunk_count": len(results["ids"]),
            "uploaded_at": timestamp,
            "domain": domain if domain else "unchanged"
        }

    return {
        "status": "success",
        "available_domains": VALID_DOMAINS,
        "details": updated_docs
    }
