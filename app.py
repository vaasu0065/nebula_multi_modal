import os
import tempfile
import streamlit as st
import fitz
from PIL import Image
import io
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Client, Settings
from tabulate import tabulate
from cohere import Client as CohereClient
import email

# Setup directories
PDF_DIR = "pdf_files"
OUTPUT_DIR = "extracted_data"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize clients
COHERE_API_KEY = "nvLVzWCKT8LSltrRIk6YEy93N27rEyknXI3YRgss"
cohere_client = CohereClient(COHERE_API_KEY)
text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = Client(settings=Settings(persist_directory="chroma_db", anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="pdf_rag", metadata={"hnsw:space": "cosine"})

SUPPORTED_TYPES = ["pdf", "csv", "xlsx", "eml"]

class MultiModalProcessor:
    def __init__(self):
        # use the global embedding model
        self.text_embedding_model = text_embedding_model
        self.all_text_chunks = []
        self.all_tables = []
        self.all_images = []

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            self._chunk_and_store(page_text, source=f"{pdf_path}#page{page_num+1}")
            # 👉 Extract tables
            try:
                page_tables = page.find_tables()
                if page_tables.tables:
                    for i, table in enumerate(page_tables.tables):
                        df = table.to_pandas()
                        self.all_tables.append({
                            "source": pdf_path,
                            "page": page_num + 1,
                            "table_num": i + 1,
                            "data": df
                        })
            except Exception as e:
                print(f"Table extraction failed on page {page_num+1}: {e}")
            # 👉 Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    if image.mode in ("RGBA", "P"):
                        image = image.convert("RGB")
                    img_path = os.path.join(
                        OUTPUT_DIR,
                        f"{os.path.basename(pdf_path)}_p{page_num+1}_i{img_index}.png"
                    )
                    image.save(img_path)
                    self.all_images.append({
                        "source": pdf_path,
                        "page": page_num + 1,
                        "img_index": img_index,
                        "path": img_path
                    })
                except Exception as e:
                    print(f"Image extraction failed on page {page_num+1}: {e}")
        doc.close()

    def process_csv_or_xlsx(self, path):
        # 👉 Handle spreadsheets
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            for idx, row in df.iterrows():
                row_text = " | ".join([f"{col}:{val}" for col, val in row.items()])
                self._chunk_and_store(row_text, source=f"{path}#row{idx}")
            self.all_tables.append({"source": path, "data": df})
        except Exception as e:
            print(f"Spreadsheet processing failed: {e}")

    def process_eml(self, path):
        # 👉 Handle email files (.eml)
        try:
            with open(path, 'rb') as f:
                msg = email.message_from_binary_file(f)
            subject = msg.get('subject', '')
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body += part.get_payload(decode=True).decode(errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            full_text = f"Subject: {subject}\n\n{body}"
            self._chunk_and_store(full_text, source=path)
        except Exception as e:
            print(f"Email processing failed: {e}")

    def _chunk_and_store(self, text, source):
        # 👉 Break text into overlapping chunks
        words = text.split()
        chunk_size, overlap = 500, 50
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            self.all_text_chunks.append({"text": chunk, "source": source})

    def process_files(self, file_paths):
        # 👉 Process all uploaded files
        for path in file_paths:
            if path.endswith(".pdf"):
                self.process_pdf(path)
            elif path.endswith(".csv") or path.endswith(".xlsx"):
                self.process_csv_or_xlsx(path)
            elif path.endswith(".eml"):
                self.process_eml(path)

        # 👉 Store embeddings in ChromaDB
        if collection.count() > 0:
            collection.delete(ids=collection.get()["ids"])

        embeddings = self.text_embedding_model.encode(
            [c["text"] for c in self.all_text_chunks],
            convert_to_tensor=False
        )

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(self.all_text_chunks))],
            embeddings=embeddings.tolist(),
            documents=[c["text"] for c in self.all_text_chunks],
            metadatas=[{"source": c["source"]} for c in self.all_text_chunks]
        )

        return self.all_tables, self.all_images
class SmartPDFRAG:
    def retrieve_relevant_content(self, query):
        query_embedding = text_embedding_model.encode(query, convert_to_tensor=False)
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3, include=["documents"])
        return {"text": results["documents"][0] if results["documents"] else []}

    def generate_response(self, query, context, tables, images):
        table_refs = "\n\n".join([f"Table (source: {t['source']}) Columns: {list(t['data'].columns)}" for t in tables[:3]]) if tables else "No tables found."
        img_refs = "\n\n".join([f"Image (source: {img['source']}) Path: {img['path']}" for img in images[:3]]) if images else "No images found."
        prompt = f"""
        You are an AI document assistant. Answer using:
        Context: {context}
        {table_refs}
        {img_refs}
        Question: {query}
        """
        response = cohere_client.generate(model="command-r-plus", prompt=prompt, max_tokens=300, temperature=0.3)
        return response.generations[0].text.strip()

st.title(" Smart Multi-Modal Document Q&A App")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, CSV, XLSX, EML)", type=["pdf", "csv", "xlsx", "eml"], accept_multiple_files=True
)
question = st.text_input("Ask a question about the documents")

if st.button("Submit"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            # Save uploaded files to temp paths
            temp_paths = []
            for uf in uploaded_files:
                ext = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uf.read())
                    temp_paths.append(tmp.name)

            # Process all files
            processor = MultiModalProcessor()
            tables, images = processor.process_files(temp_paths)

            # Retrieve and generate answer
            rag = SmartPDFRAG()
            context_text = rag.retrieve_relevant_content(question)["text"]
            answer = rag.generate_response(question, "\n".join(context_text), tables, images)

            st.success("✅ Answer generated successfully!")
            st.text_area("Answer:", value=answer, height=200)
