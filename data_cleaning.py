from typing import List, Optional
import fire
from llama import Llama
from pdfloader import PDFProcessor
from database import ChromaDBManager

def generate_cleaning_prompts(document_chunks: List[str]) -> List[List[dict]]:
    # Generate prompts for each document chunk
    dialogs = []
    for chunk in document_chunks:
        dialog = [
            {"role": "system", "content": "Your task is to clean the following text, correcting any errors and improving clarity."},
            {"role": "user", "content": chunk},
        ]
        dialogs.append(dialog)
    return dialogs

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    query: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
    load_pdf_file_path: str = None,
    load_pdf_file_name: str = None,
    load_pdf_directory_path: str = None,
    pdf_directory: str = '/gpfs/scratch/yh2563/ExamplePDFsForLLM/',
    database: str = 'my_database.db',
    vector_database: str = './chroma_db'
):
    # Initialize PDFProcessor, SQLiteDBManager, and ChromaDBManager
    # ... [Your existing initialization code]
    pdf_processor = PDFProcessor(pdf_directory)
    chroma_manager = ChromaDBManager(database, vector_database)

    # Loading document chunks
    document_chunks = []
    if load_pdf_file_name:
        path = pdf_directory + load_pdf_file_name
        document_chunks = pdf_processor.load_and_split_document_by_title(path)

    if load_pdf_file_path:
        document_chunks = pdf_processor.load_and_split_document_by_title(load_pdf_file_path)

    if load_pdf_directory_path:
        document_chunks = pdf_processor.load_from_directory(load_pdf_directory_path)

    # Generate cleaning prompts
    cleaning_prompts = generate_cleaning_prompts(document_chunks)

    # Initialize Llama2 for data cleaning
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )

    # Clean document chunks with Llama2
    cleaned_chunks = []
    for dialog in cleaning_prompts:
        result = generator.chat_completion(
            dialog,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        cleaned_chunk = result[0]['generation']['content']
        cleaned_chunks.append(cleaned_chunk)

    # Adding cleaned documents to database
    chroma_manager.add_documents_to_chroma(cleaned_chunks)

if __name__ == "__main__":
    fire.Fire(main)