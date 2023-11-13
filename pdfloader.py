

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os


class PDFProcessor:
    def __init__(self, pdf_directory = '', text_splitter= RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)):
        self.text_splitter = text_splitter
        self.pdf_directory = pdf_directory

    def get_file_path(self, filename):
        return os.path.join(self.pdf_directory, filename)

    def load_and_split_document_by_title(self, title):
        path = self.get_file_path(title)
        loader = PyPDFLoader(path)
        document = loader.load()
        doc_chucks = self.text_splitter.split_documents(document)
        return doc_chucks

    def load_from_directory(self, path=''):
        if path == '':
            path = self.pdf_directory
        loader = PyPDFDirectoryLoader(path)
        doc_chunks = loader.load_and_split(self.text_splitter)
        return doc_chunks

def main():
    # Directory where the PDF files are located
    pdf_directory = "/content/drive/MyDrive/LLM/ExamplePDFsForLLM/"
    
    # Initialize the PDFProcessor with the directory
    pdf_processor = PDFProcessor(store_directory=pdf_directory)
    
    # Filename of the PDF to process
    pdf_title = "ExamplePDFTitle.pdf"
    
    # Load and split the document by title
    document_chunks = pdf_processor.load_and_split_document_by_title(pdf_title)
    
    # For demonstration purposes, let's just print the first 200 characters of the first chunk
    # of the document to show that it's been loaded and split properly.
    print(document_chunks[0].page_content[:200])

    # If you want to load all PDFs from a directory and process them
    all_docs_chunks = pdf_processor.load_from_directory(pdf_directory)
    
    # Here you can add additional processing for the document chunks

if __name__ == "__main__":
    main()