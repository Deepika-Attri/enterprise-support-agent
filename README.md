# 🤖 Enterprise Support Agent

An agentic RAG application that acts as an internal support and knowledge assistant. A LangGraph agent routes each question, retrieves answers from company documents with source citations, looks up orders with tools, and escalates to a human ticket when it can't answer confidently. Runs entirely on free, local components.

## 🚀 Features

* 🧭 LangGraph agent with a router (knowledge / account / escalate / out of scope)
* 🔁 Corrective RAG: relevance grading and automatic query rewriting (max 2 retries)
* 🔎 Semantic search with ChromaDB and metadata filtering by department
* 🔢 Free local embeddings using Sentence Transformers
* 🛠️ Tool use: order lookup and support-ticket creation backed by SQLite
* 🙋 Human escalation when no relevant answer is found
* 📌 Source citations on every document-based answer
* 📤 Upload your own PDF, Markdown or text files and organize them by department
* 🧪 "Try with sample data" toggle to demo with the included NovaCart documents
* 🔌 Switchable LLM provider: Ollama (local) or Gemini (free tier)
* 🐞 Developer mode in the UI plus terminal logging of routes and tickets
* ✅ Unit tests for tools and routing logic

## 🛠️ Technologies Used

* Python
* LangChain
* LangGraph
* ChromaDB
* Sentence Transformers (HuggingFace)
* Ollama / Google Gemini API
* SQLite
* Streamlit
* pytest

## 📁 Project Structure

```text
enterprise-support-agent/
│
├── app.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── tools.py
│   ├── agent.py
│   └── prompts.py
│
├── data/
│   └── sample_documents/
│       ├── support/
│       │   ├── refund_policy.md
│       │   └── shipping_faq.md
│       └── hr/
│           └── leave_policy.md
│
└── tests/
    └── test_agent.py
```

Uploaded documents are stored in `data/documents/` (git-ignored). The vector index and mock database are created automatically on first run.

## 🔄 How It Works

```text
User question
    ↓
Router (knowledge / account / escalate / out of scope)
    ├── account ──→ Order lookup tool (SQLite) ──→ Answer
    ├── escalate ─→ Create support ticket
    ├── out of scope ─→ Polite refusal
    └── knowledge
           ↓
        Retrieve (ChromaDB + metadata filters)
           ↓
        Relevance grader
           ├── relevant ─→ Generate cited answer
           └── weak ─→ Rewrite query ─→ Retrieve again (max 2 retries)
                                          └── still weak ─→ Escalate to human ticket
```

Each node reads from and writes to a shared LangGraph state, and conditional edges decide the next step. The vector store sits behind a single module (`retrieval.py`), so swapping ChromaDB for Pinecone only means editing that file.

**Try these questions with the sample data:**

* "How long do refunds take?"
* "Where is my order ORD-1001?"
* "How many paid leaves do I get?"
* "What is the capital of France?" (out-of-scope refusal)

**Run the tests:**

```bash
pytest
```

## ⚠️ Known Limitations

* **Model size affects grading quality.** Small local models (for example 3B parameters) can be inconsistent as relevance graders. An overly strict grade triggers extra query rewrites and an unnecessary escalation ticket. Larger models or Gemini behave more reliably.
* **Routing is LLM-based**, so a small model can occasionally misclassify a question. Routing accuracy will be measured in Phase 2.
* **No OCR.** Scanned PDFs without selectable text can't be indexed.
* **Supported formats:** PDF, Markdown and plain text only.
* **Mock data.** Orders and tickets use a small SQLite demo database, and the sample documents are intentionally short.
* **Free-tier model names change.** Set `GEMINI_MODEL` in `.env` to a model currently available in Google AI Studio.

## 📁 Future Enhancements

* Multi-agent architecture: supervisor with Knowledge, Account, Escalation and Critic agents
* RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall) comparing baseline RAG, single agent and multi-agent
* Agent-level metrics: routing accuracy, tool selection accuracy, escalation accuracy
* Hybrid keyword + semantic retrieval and reranking
* LangSmith tracing
* Support for DOCX files
* Docker packaging and a hosted demo