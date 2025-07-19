# 🧠 Mini RAG-Powered QA System using Mistral-7B

This project implements a mini **Retrieval-Augmented Generation (RAG)** pipeline to build a question-answering system over a medical knowledge document (e.g., clinical guidelines or mental health protocols). It leverages **Mistral-7B** as the LLM backbone, FAISS for similarity search, and sentence-transformers for embeddings.

## 📌 Objective

The goal is to process a large document and allow users to ask natural language questions. The system retrieves the most relevant chunks using semantic search and passes them to the Mistral-7B model to generate contextualized answers.

---

## 🔧 Tools & Technologies Used

| Component               | Technology/Model                          |
|------------------------|-------------------------------------------|
| Embedding Model        | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store           | `FAISS`                                    |
| LLM                    | `mistralai/Mistral-7B-Instruct-v0.1`       |
| Tokenizer              | HuggingFace Transformers                  |
| Backend Language       | Python 3.11 (Colab/Ubuntu compatible)     |

---

## 📁 Project Structure

```bash
├── main_mistral.py            # Full RAG pipeline for Mistral
├── requirements.txt           # All dependencies
├── sample_outputs.txt         # Output from the test questions
├── README.md                  # This file
└── data/
    └── icd10_full_text.pdf    # Medical classification text (source document)
```

##Setup Instructions:
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mini-rag-mistral.git
cd mini-rag-mistral

# 2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Setup:
```bash
# Make sure to export your HuggingFace token
export HUGGINGFACEHUB_API_TOKEN=your_token_here

# Run the end-to-end QA pipeline using Mistral
python main_mistral.py
```

## Sample Usage:
Query:
```bash
Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission
```
LLM Response:
```bash
F33.4
```

Query:
```bash
What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?
```
LLM Response :
```bash
- Obsessional thoughts or compulsive acts must be present on most days for at least 2 successive weeks.
- They must be recognized as the individual's own thoughts.
- At least one thought or act must still be resisted, even if unsuccessfully.
- The symptoms must cause distress or interfere with activities.
```

## Design Decision
Chunk Size: ~200–300 tokens with 20-token overlap to retain context without redundancy.

Embedding Model: all-MiniLM-L6-v2 for its lightweight footprint and strong performance.

LLM: Mistral-7B-Instruct was selected for its open-source license and reasoning strength.

Pipeline Strategy: Built from scratch using HuggingFace + FAISS for full transparency and control.

## AI Tool Usage
This implementation received assistance for:

    ChatGPT: Debugging CUDA errors, streamlining pipeline integration, and resolving tokenizer mismatch issues.

    Copilot: Auto-completing imports, docstrings, and helping build query interfaces.

All AI-generated help was reviewed and edited for clarity and correctness.

## Limitations
Out-of-Memory: Mistral-7B requires ~14GB VRAM. May crash on low-resource devices.

No UI: Streamlit or Gradio interface not implemented due to time constraints.

No Reranking: MMR-style relevance reranking was not added.

Hardcoded Paths: File and model paths are relative and not parameterized.

## Completed Requirements
Document loading & chunking

 Embedding generation & FAISS indexing

 Top-k chunk retrieval

 Query interface (CLI)

 LLM-based contextual answering

 Sample input/output included

 Clean README with setup instructions
