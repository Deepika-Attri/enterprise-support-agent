# 🤖 Enterprise Support Agent

An agentic RAG application that acts as an internal support and knowledge assistant. A LangGraph agent routes each question, retrieves answers from company documents with source citations, looks up orders with tools, and escalates to a human ticket when the user explicitly requests human assistance. Runs entirely on free, local components.

## 🚀 Features

* 🧭 LangGraph agent with a router (knowledge / account / escalate / out of scope)
* 🔁 Corrective RAG: relevance grading and automatic query rewriting (max 2 retries)
* 🔎 Semantic search with ChromaDB and metadata filtering by department
* 🔢 Free local embeddings using Sentence Transformers
* 🛠️ Tool use: order lookup and support-ticket creation backed by SQLite
* 🙋 Human escalation when the user explicitly requests human assistance
* 📌 Source citations on document-based answers
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
├── pytest.ini                # tells pytest how to run/configure tests
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

## 🧪 Try It

Try these questions with the sample data:

* "How long do refunds take?"
* "Where is my order ORD-1001?"
* "How many paid leaves do I get?"
* "Do you offer a student discount?"
* "What is the capital of France?" (out-of-scope refusal)
* "I want to talk to a manager." (human escalation)

## ✅ Run the Tests

```bash
pytest
```

## 📁 Future Enhancements

* Multi-agent architecture: supervisor with Knowledge, Account, Escalation and Critic agents
* RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall) comparing baseline RAG, single-agent and multi-agent approaches
* Agent-level metrics: routing accuracy, tool selection accuracy, escalation accuracy
* Hybrid keyword + semantic retrieval and reranking
* LangSmith tracing
* Support for DOCX files
* Docker packaging and a hosted demo
```
