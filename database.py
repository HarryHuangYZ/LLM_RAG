import sqlite3
import os


import chromadb

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from pdfloader import PDFProcessor

class SQLiteDBManager:
    def __init__(self, db_path='my_database.db'):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        # Check if the database file exists
        db_exists = os.path.exists(self.db_path)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        if not db_exists:
            # Create table if database does not exist
            c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                ID INTEGER PRIMARY KEY,
                Content TEXT,
                Source TEXT,
                Page INTEGER
            )
            ''')
            conn.commit()

        conn.close()
    
    def insert_documents(self, documents):
        """
        Insert multiple documents into the SQLite database if they do not already exist.
        Do not insert if the document already exists.

        :param documents: A list of Document objects.
        :param db_path: The path to the SQLite database file.
        :return: Two dictionaries with sources as keys and lists of IDs as values.
        """
        # Connect to the SQLite database
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Dictionaries to store the IDs of the documents grouped by source
        new_document_ids_by_source = {}
        existing_document_ids_by_source = {}

        try:
            # Begin a transaction
            conn.execute('BEGIN TRANSACTION;')

            for document in documents:
                source = document.metadata['source']
                page = document.metadata['page']
                content = document.page_content

                # Check if the document already exists
                c.execute('SELECT ID FROM documents WHERE Source = ? AND Page = ? AND Content = ?', (source, page, content))
                existing_doc_id = c.fetchone()

                if existing_doc_id:
                    # Document already exists, skip insertion and add to existing_document_ids_by_source
                    existing_document_ids_by_source.setdefault(source, []).append(existing_doc_id[0])
                else:
                    # Insert a new document
                    c.execute('''
                    INSERT INTO documents (Content, Source, Page) VALUES (?, ?, ?)
                    ''', (content, source, page))
                    new_document_id = c.lastrowid
                    new_document_ids_by_source.setdefault(source, []).append(new_document_id)

            # Commit the changes to the database
            conn.commit()

        except sqlite3.Error as e:
            # Rollback on any error
            conn.rollback()
            print(f"An error occurred: {e}")
            return {}, {}

        finally:
            # Close the connection
            conn.close()

        # Remove duplicates from lists and return
        for source in existing_document_ids_by_source:
            existing_document_ids_by_source[source] = list(set(existing_document_ids_by_source[source]))
        for source in new_document_ids_by_source:
            new_document_ids_by_source[source] = list(set(new_document_ids_by_source[source]))
        return new_document_ids_by_source, existing_document_ids_by_source

    def delete_single_document_by_source(self, source):
        """
        Delete documents from the SQLite database based on the document source.

        :param source: The source of the documents to be deleted.
        :param db_path: The path to the SQLite database file.
        :return: The IDs of the deleted documents.
        """
        # Connect to the SQLite database
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Prepare the SQL query to select IDs of documents with the given source
        select_query = 'SELECT ID FROM documents WHERE Source = ?'
        delete_query = 'DELETE FROM documents WHERE Source = ?'
        deleted_ids = []

        try:
            # Begin a transaction
            conn.execute('BEGIN TRANSACTION;')

            # Select the document IDs with the given source
            c.execute(select_query, (source,))
            rows = c.fetchall()
            deleted_ids = [row[0] for row in rows]

            # If there are documents to delete, execute the delete query
            if deleted_ids:
                c.execute(delete_query, (source,))

            # Commit the changes to the database
            conn.commit()

            # Return the IDs of the deleted documents
            return deleted_ids

        except sqlite3.Error as e:
            # Rollback on any error
            conn.rollback()
            print(f"An error occurred: {e}")
            return None

        finally:
            # Close the connection
            conn.close()


    def delete_multiple_documents_by_sources(self, sources):
        """
        Delete documents from the SQLite database based on a list of document sources.

        :param sources: A list of sources of the documents to be deleted.
        :param db_path: The path to the SQLite database file.
        :return: A dictionary with sources as keys and the list of deleted document IDs for each source as values.
        """
        # Connect to the SQLite database
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Dictionary to store the sources and their corresponding deleted document IDs
        deleted_documents = {}

        try:
            # Begin a transaction
            conn.execute('BEGIN TRANSACTION;')

            for source in sources:
                # Select the document IDs with the given source
                c.execute('SELECT ID FROM documents WHERE Source = ?', (source,))
                rows = c.fetchall()
                deleted_ids = [row[0] for row in rows]

                # If there are documents to delete, execute the delete query
                if deleted_ids:
                    c.execute('DELETE FROM documents WHERE Source = ?', (source,))
                    deleted_documents[source] = deleted_ids

            # Commit the changes to the database
            conn.commit()

        except sqlite3.Error as e:
            # Rollback on any error
            conn.rollback()
            print(f"An error occurred: {e}")
            return None

        finally:
            # Close the connection
            conn.close()

        return deleted_documents


    def check_document_exists(self, source):
        """
        Check if a document exists in the SQLite database based on its source and page.

        :param source: The source of the document to check.
        :param page: The page number of the document to check.
        :param db_path: The path to the SQLite database file.
        :return: True if the document exists, False otherwise.
        """
        # Connect to the SQLite database
        db_path = self.db_path
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Check if the document exists
        c.execute('SELECT ID FROM documents WHERE Source = ?', (source,))
        exists = c.fetchall()
        exists_id = [row[0] for row in exists]

        # Close the connection
        conn.close()

        return exists_id


class ChromaDBManager:
    def __init__(self, db_path='my_database.db', chroma_save_path='./chroma_db', embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")):
        self.chroma_save_path = chroma_save_path
        self.sqlite_db_manager = SQLiteDBManager(db_path)
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.chroma_instance = Chroma(persist_directory=self.chroma_save_path, embedding_function=self.embedding_function)

    def add_documents_to_chroma(self, documents):
        # Insert documents into the SQL database and get the new and existing IDs
        new_document_ids_by_source, existing_document_ids_by_source = self.sqlite_db_manager.insert_documents(documents)

        # For the purpose of adding to Chroma, we only care about new documents
        new_ids = [str(item) for sublist in list(new_document_ids_by_source.values()) for item in sublist]

        new_documents = [doc for doc in documents if new_document_ids_by_source.get(doc.metadata['source'])]

        # Add the new documents and their embeddings to the Chroma vector database

        self.chroma_instance = Chroma.from_documents(
            documents=new_documents,
            ids=new_ids,
            embedding=self.embedding_function,  
            # collection_name="my_collection",
            persist_directory=self.chroma_save_path
        )

        print('successfully add')

    def delete_document_from_chroma(self, document_source):
        """
        Delete a document from both the SQLite database and the Chroma vector database based on the document source.

        :param document_source: The source of the document to be deleted.
        :param chroma_instance: The instance of the Chroma vector database.
        :param db_path: The path to the SQLite database file.
        """
        # Delete the document from the SQL database and get the deleted document IDs
        deleted_ids = [str(id) for id in self.sqlite_db_manager.delete_single_document_by_source(document_source)]

        if deleted_ids:
            # Delete the documents from the Chroma vector database
            self.chroma_instance._collection.delete(ids=deleted_ids)
            print(f"Deleted documents with IDs {deleted_ids} from the Chroma vector database.")
        else:
            print(f"No documents found for source {document_source} to delete.")

    def query(self, prompt, num_result=3):
        docs = self.chroma_instance.similarity_search(prompt, k=num_result)
        result_string = ""
        for i, (page_content_tuple, metadata_tuple) in enumerate(docs, start=1):
            _, page_content = page_content_tuple
            _, metadata = metadata_tuple
            source = metadata.get('source', 'Unknown source')
            result_string += f"{i}: {page_content}\nSource: {source}\n\n"
        return result_string

    def count_document(self):
        count = self.chroma_instance._collection.count()
        return count

    def persist(self):
        self.chroma_instance.persist()
        print(f'Successfully store Chroma DB at {self.chroma_save_path}')

        
def main():
    # Initialize the SQLite and Chroma database managers
    sqlite_manager = SQLiteDBManager('my_database.db')
    chroma_manager = ChromaDBManager('my_database.db', './chroma_db')

    # Example documents to insert (replace with actual documents)
    # Directory where the PDF files are located
    pdf_directory = "/gpfs/scratch/yh2563/ExamplePDFsForLLM/"
    
    # Initialize the PDFProcessor with the directory
    pdf_processor = PDFProcessor(pdf_directory=pdf_directory)

    # Filename of the PDF to process
    # pdf_title = "jama_271_5_036.pdf"

    # Load and split the document by title
    # document_chunks = pdf_processor.load_and_split_document_by_title(pdf_title)

    # For demonstration purposes, let's just print the first 200 characters of $
    # of the document to show that it's been loaded and split properly.
    # print(document_chunks[0].page_content[:200])
    
    # print('start loading')    

    # If you want to load all PDFs from a directory and process them
    documents = pdf_processor.load_from_directory(pdf_directory)
    
    # print('adding to chroma')

    # Add documents to both SQLite and Chroma databases
    chroma_manager.add_documents_to_chroma(documents)

    # Query documents from Chroma database
    query_prompt = "I’m writing a paper related to Prostate Cancer Localization, give me some paper about it"
    queried_docs = chroma_manager.query(query_prompt, num_result=3)
    print(queried_docs)

    # Count the number of documents in Chroma database
    doc_count = chroma_manager.count_document()
    print(f"Number of documents in Chroma database: {doc_count}")
    
    # store the chromadb
    chroma_manager.persist()

    # Example to delete a document by source
    #source_to_delete = pdf_directory + "jama_271_5_036.pdf"  # Replace with actual source
    #chroma_manager.delete_document_from_chroma(source_to_delete)

if __name__ == "__main__":
    main()
