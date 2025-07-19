from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# ✅ Upload and load PDF
pdf_path = "/content/9241544228_eng.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages")

# ✅ Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print(f"✅ Chunked into {len(chunks)} sections")

# ✅ Convert to embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ Embeddings ready\n✅ FAISS index built")

# ✅ Hugging Face token (replace or input securely)
HF_TOKEN = input("Enter your Hugging Face token: ").strip()

# ✅ Load Mistral model with accelerate & torch_dtype
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
print("✅ Model loaded and ready")

# ✅ Inference pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ Query Function with context-restricted prompting
def query_pdf(question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful and accurate medical assistant. Use ONLY the context below to answer the question.
If the answer is not present in the context, say "Not found in the document."

Context:
{context}

Question: {question}
Answer:"""

    response = llm(prompt, max_new_tokens=256, do_sample=False)
    print("\n🧠 LLM Answer:\n", response[0]["generated_text"].split("Answer:")[-1].strip())

# ✅ Sample Query
query_pdf("Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission")

query_pdf("What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?")