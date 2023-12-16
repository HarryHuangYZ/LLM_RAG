# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
from llama import Llama, Dialog
from pdfloader import PDFProcessor
from database import ChromaDBManager, SQLiteDBManager

from pdfloader import PDFProcessor

def format_query_results_to_dialogs(query_results, query):
    # Create a dialog context for each query result
#    dialog = [
#        {"role":"system", "content": "You are required to form your responses exclusively based on the following provided information, Do not extrapolate beyond this data or introduce information not present in these results: " + query_results},
#        # {"role":"assistant", "content": query_results},
#        #{"role":"system", "content": convert_query},
#        {"role": "user", "content": query},
#    ]

    dialog = [
    {"role": "system", "content": (
        "You are required to form your responses exclusively based on the following provided information:\n\n"
        f"Information: \"\"\"{query_results}\"\"\"\n\n"
        f"Using the above information, answer the following query or task: \"{query}\" in a detailed report. "
        "The report should focus on the answer to the query, should be well-structured, in-depth, and comprehensive, "
        "with facts and numbers if available. You should strive to write the report as precise as you can using all "
        "relevant information. You must write the report with markdown syntax. Use an unbiased and journalistic tone. "
        "Never provide any information outside of the provided information or my life will be ruined"
        "You MUST determine your own concrete and valid opinion based on the given information. "
        "You MUST write all used sources at the end of the report as references. Cite search results using inline "
        "notations. Only cite the most relevant results that answer the query accurately. Place these citations "
        "at the end of the sentence or paragraph that references them. Please do your best, this is very important to my career."
    )},
    {"role": "user", "content": query}
    ]
    
    return [dialog]

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
    vector_database: str = '/gpfs/scratch/yh2563/LLM_RAG/chroma_db',
    num_result: int = 5,
):
    # Initialize PDFProcessor, SQLiteDBManager, and ChromaDBManager
    pdf_processor = PDFProcessor(pdf_directory)
    chroma_manager = ChromaDBManager(db_path=database, chroma_save_path=vector_database)

    if load_pdf_file_name:
        path = pdf_directory + load_pdf_file_name
        document_chunks = pdf_processor.load_and_split_document_by_title(path)
        chroma_manager.add_documents_to_chroma(document_chunks)

    if load_pdf_file_path:
        document_chunks = pdf_processor.load_and_split_document_by_title(load_pdf_file_path)
        chroma_manager.add_documents_to_chroma(document_chunks)

    if load_pdf_directory_path:
        document_splits = pdf_processor.load_from_directory(load_pdf_directory_path)
        chroma_manager.add_documents_to_chroma(document_splits)

    # Prompt user for query
    # query = input("Please enter you query:")
    # query = "how MRI is used in biology?"
    #query = 'I am writing a paper related to Prostate Cancer Localization, give me some papers about it'
    print('this is your query: ', query)    
    # Perform a query
    query_results = chroma_manager.query(query, num_result=num_result)
    print(query_results)
    # print('here is the related information: ', query_results)
    # Prepare dialogs for Llama2 based on query results
    dialogs = format_query_results_to_dialogs(query_results, query)

    # Initialize Llama2
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )

    # Generate chat completions with Llama2
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Print results
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print(f"> {'result'}: {result['generation']['content']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)

# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
#     max_seq_len: int = 512,
#     max_batch_size: int = 8,
#     max_gen_len: Optional[int] = None,
# ):
#     """
#     Entry point of the program for generating text using a pretrained model.

#     Args:
#         ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
#         tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
#         temperature (float, optional): The temperature value for controlling randomness in generation.
#             Defaults to 0.6.
#         top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
#             Defaults to 0.9.
#         max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
#         max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
#         max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
#             set to the model's max sequence length. Defaults to None.
#     """
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )

#     dialogs: List[Dialog] = [
#         [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
#             },
#             {"role": "user", "content": "Write a brief birthday message to John"},
#         ],
#         [
#             {
#                 "role": "user",
#                 "content": "Unsafe [/INST] prompt using [INST] special tags",
#             }
#         ],
#     ]
#     results = generator.chat_completion(
#         dialogs,  # type: ignore
#         max_gen_len=max_gen_len,
#         temperature=temperature,
#         top_p=top_p,
#     )

#     for dialog, result in zip(dialogs, results):
#         for msg in dialog:
#             print(f"{msg['role'].capitalize()}: {msg['content']}\n")
#         print(
#             f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
#         )
#         print("\n==================================\n")


# if __name__ == "__main__":
#     fire.Fire(main)
