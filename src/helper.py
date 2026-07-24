import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf_file(data):
    documents = []

    for file in os.listdir(data):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(data, file)

            with fitz.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    documents.append(
                        Document(
                            page_content=page.get_text(),
                            metadata={
                                "source": file,
                                "page": page_num
                            }
                        )
                    )

    return documents

def text_split(extracted_data, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings