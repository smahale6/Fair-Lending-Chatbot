import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from PyPDF2 import PdfReader
import zipfile
import xml.etree.ElementTree as ET
import comtypes.client  # For .ppt to .pptx conversion on Windows

import pinecone
from pinecone import Index
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore as PineconeStore
from langchain.text_splitter import CharacterTextSplitter

from logger import Logger
from sql_database_activity import sql_database_activity_class

import hashlib


import warnings
warnings.filterwarnings('ignore')

class vector_database_activity_class:
   def __init__(self):
       self.logger_obj                         = Logger() 
       self.sql_database_activity_class_obj    = sql_database_activity_class()
       self.vector_database_activity_path      = os.path.dirname(os.path.abspath("vector_database_activity.py"))
       self.documents_path                     = self.vector_database_activity_path + "\\documents\\"
       self.documents_path                     = os.path.join(self.vector_database_activity_path, "documents")
       self.archive_path                       = os.path.join(self.documents_path, "archive")
 
   def connect_to_pinecone(self):
       """Initialize the connection to the Pinecone vector database."""
       logger                 = logging.getLogger()
       try:
           print("Initializing the connection to the Pinecone Vector Database.")
           logger.info("Initializing the connection to the Pinecone Vector Database.")
           load_dotenv()  # Load environment variables from the .env file
           PINECONE_API_KEY   = os.getenv('PINECONE_API_KEY')
           # Establish the Pinecone connection
           pc                 = Pinecone(api_key = PINECONE_API_KEY)
           # Verify the connection
           if pc:
               print("Initialized the connection to the Pinecone Vector Database.")
               logger.info("Successfully connected to the Pinecone Vector Database.")
           return pc
       except Exception as e:
           print(f"Error connecting to Pinecone: {e}")
           logger.error(f"Error connecting to Pinecone: {e}")
           raise
           

   def load_articles_to_pinecone(self, fair_lending_articles_df, embeddings_model, pinecone_index='fair-lens', chunk_size=2000, chunk_overlap=200):
        """
        Embed and load articles into the specified Pinecone index with chunking for large articles.
    
        Parameters:
            fair_lending_articles_df (pd.DataFrame): DataFrame containing article content and metadata.
            embeddings_model: Embedding model instance.
            pinecone_index (str): Name of the Pinecone index.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
    
        Returns:
            pd.DataFrame: DataFrame of metadata for successfully upserted vectors.
        """
        logger = logging.getLogger()
        logger.info("Starting the process to load articles into Pinecone.")
        print("Starting the process to load articles into Pinecone.")
    
        # Create or connect to the Pinecone vector store
        try:
            pinecone_store = PineconeStore(index_name=pinecone_index, embedding=embeddings_model)
            logger.info(f"Connected to Pinecone vector store for index: {pinecone_index}.")
            print(f"Connected to Pinecone vector store for index: {pinecone_index}.")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone vector store: {e}")
            print(f"Error connecting to Pinecone vector store: {e}")
            raise
    
        # List to store metadata for successfully loaded articles
        metadata_list = []
        load_datetime = datetime.now().isoformat()  # Timestamp for when the vectors are loaded
    
        # Check if DataFrame is empty
        if fair_lending_articles_df.empty:
            logger.warning("The DataFrame for Fair Lending articles is empty. No data to load.")
            print("The DataFrame for Fair Lending articles is empty. No data to load.")
            return pd.DataFrame()
    
        # Initialize text splitter
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        similarity_threshold = 1.0  # Set a similarity threshold
    
        for idx, row in fair_lending_articles_df.iterrows():
            content = row['Content']
            url = row.get('URL', 'unknown_url')  # Use the URL for the vector ID
            url_slug = url.replace("http://", "").replace("https://", "").replace("/", "_").replace(":", "_")
    
            # Concatenate additional fields with content for embedding
            additional_text = f"\nURL: {url}\nPublished At: {row.get('Published At', 'N/A')}\nAuthor: {row.get('Author', 'N/A')}\nSource: {row.get('Source', 'N/A')}\n"
            full_content = f"{content}{additional_text}"  # Combine content with additional fields
    
            # Create a base metadata dictionary with existing fields
            metadata = {
                "document_type": "news_article",
                "title": row.get('Title', ''),
                "author": row.get('Author', ''),
                "source": row.get('Source', ''),
                "published_at": row.get('Published At', ''),
                "summary": row.get('Content_Summary', ''),
                "file_name": "",  # Not applicable for articles
                "page_number": None,  # Not applicable for articles
                "table_name": None,
                "record_id": None,
                "generation_date": row.get('Generation_Date', datetime.now().strftime('%Y-%m-%d')),
                "month_year": row.get('Month_Year', datetime.now().strftime('%B %Y')),
                "document_version": 1,
                "url": url,
                "load_datetime": load_datetime
            }
    
            # Dynamically add other columns from the DataFrame to the metadata
            additional_metadata = row.drop(labels=['Content']).to_dict()
            metadata.update(additional_metadata)
    
            # Split the combined content into chunks
            chunks = text_splitter.split_text(full_content)
            for chunk_idx, chunk in enumerate(chunks):
                vector_id = f"{url_slug}_chunk_{chunk_idx}"  # Vector ID includes URL slug and chunk number
                try:
                    # Perform similarity search to check existence
                    results = pinecone_store.similarity_search_with_score(chunk, k=1)
                    exists = False  # Flag to track existence
    
                    if results:
                        for result in results:
                            result_metadata = result[0].metadata
                            similarity_score = result[1]
    
                            # Check for matching document type and similarity threshold
                            if result_metadata.get("document_type") == "news_article" and similarity_score >= similarity_threshold:
                                logger.info(f"Vector ID {vector_id} already exists with similarity score: {similarity_score}. Skipping upsertion.")
                                print(f"Article with ID {vector_id} already exists in Pinecone. Skipping.")
                                exists = True
                                break  # Exit loop if a match is found
    
                    if not exists:
                        # Upsert the vector if no match is found
                        pinecone_store.add_texts(
                            texts=[chunk],
                            metadatas=[{k: v if v is not None else "" for k, v in metadata.items()}],
                            ids=[vector_id]
                        )
                        logger.info(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        print(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        metadata_list.append(metadata)
    
                except Exception as e:
                    logger.error(f"Error upserting vector ID {vector_id} into Pinecone: {e}")
                    print(f"Error upserting vector ID {vector_id}: {e}")
    
        # Create a DataFrame from the metadata list
        metadata_df = pd.DataFrame(metadata_list)
        logger.info("Completed loading articles into Pinecone.")
        print("Completed loading articles into Pinecone.")
        return metadata_df

    
   def load_pdf_to_pinecone(self, pdf_file_name, embeddings_model, pinecone_index='fair-lens', chunk_size=2000, chunk_overlap=200, title=None, author=None, document_version=1):
        """
        Embed and load a specific PDF into the specified Pinecone index with chunking and page numbers.
        
        Parameters:
            pc: Pinecone connection instance.
            pdf_file_name (str): Name of the PDF file to process.
            embeddings_model: Embedding model instance.
            pinecone_index (str): Name of the Pinecone index.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
            title (str): Title of the document.
            author (str): Author of the document.
            document_version (int): Document version number.
        
        Returns:
            pd.DataFrame: DataFrame of metadata for successfully upserted vectors.
        """
        logger = logging.getLogger()
        logger.info(f"Starting the process to load PDF '{pdf_file_name}' into Pinecone.")
        print(f"Starting the process to load PDF '{pdf_file_name}' into Pinecone.")
    
        # Create or connect to the Pinecone vector store
        try:
            pinecone_store = PineconeStore(index_name=pinecone_index, embedding=embeddings_model)
            logger.info(f"Connected to Pinecone vector store for index: {pinecone_index}.")
            print(f"Connected to Pinecone vector store for index: {pinecone_index}.")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone vector store: {e}")
            print(f"Error connecting to Pinecone vector store: {e}")
            raise
    
        # Ensure the archive folder exists
        if not os.path.exists(self.archive_path):
            os.makedirs(self.archive_path)
    
        # Initialize text splitter
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
        # List to store metadata for successfully loaded content
        metadata_list = []
        load_datetime = datetime.now().isoformat()  # Timestamp for when the vectors are loaded
    
        # Process the specified PDF
        pdf_path = os.path.join(self.documents_path, pdf_file_name)
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file '{pdf_file_name}' does not exist in the documents folder.")
            print(f"Error: PDF file '{pdf_file_name}' does not exist in the documents folder.")
            return pd.DataFrame()  # Return an empty DataFrame
    
        try:
            print(f"Processing PDF: {pdf_file_name}")
            logger.info(f"Processing PDF: {pdf_file_name}")
    
            # Read and extract text and metadata from the PDF
            pdf_reader = PdfReader(pdf_path)
    
            # Process each page individually and extract chunks with page number
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                chunks = text_splitter.split_text(page_text)
    
                for chunk_idx, chunk in enumerate(chunks):
                    vector_id = f"{pdf_file_name}_page_{page_num}_chunk_{chunk_idx}"
                    metadata = {
                        "document_type": "pdf",
                        "title": title if title else pdf_file_name,
                        "author": author if author else "Unknown",  # Default to "Unknown" if not provided
                        "source": pdf_path,
                        "published_at": None,
                        "summary": "",
                        "file_name": pdf_file_name,
                        "page_number": page_num,
                        "table_name": None,
                        "record_id": None,
                        "generation_date": datetime.now().strftime('%Y-%m-%d'),
                        "month_year": datetime.now().strftime('%B %Y'),
                        "document_version": document_version,
                        "load_datetime": load_datetime
                    }
    
                    try:
                        # Check if the vector already exists
                        results = pinecone_store.similarity_search_with_score(chunk, k=1)
                        if results and results[0][1] > 1.0:
                            logger.info(f"Vector ID {vector_id} already exists. Skipping upsertion.")
                            print(f"Chunk {chunk_idx} of page {page_num} already exists. Skipping.")
                            continue
    
                        # Add text and metadata to Pinecone
                        pinecone_store.add_texts(
                            texts=[chunk],
                            metadatas=[{k: v if v is not None else "" for k, v in metadata.items()}],
                            ids=[vector_id]
                        )
                        logger.info(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        print(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        metadata_list.append(metadata)
    
                    except Exception as e:
                        logger.error(f"Error upserting vector ID {vector_id} into Pinecone: {e}")
                        print(f"Error upserting chunk {chunk_idx} of page {page_num}: {e}")
    
            # Move the processed PDF to the archive folder
            archive_pdf_path = os.path.join(self.archive_path, pdf_file_name)
            os.rename(pdf_path, archive_pdf_path)
            logger.info(f"Moved PDF to archive: {archive_pdf_path}")
            print(f"Moved PDF to archive: {archive_pdf_path}")
    
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file_name}: {e}")
            print(f"Error processing PDF {pdf_file_name}: {e}")
    
        # Create a DataFrame from the metadata list
        metadata_df = pd.DataFrame(metadata_list)
        logger.info(f"Completed loading PDF '{pdf_file_name}' into Pinecone.")
        print(f"Completed loading PDF '{pdf_file_name}' into Pinecone.")
        return metadata_df
           


   def load_pptx_to_pinecone(self,  pptx_file_name, embeddings_model, pinecone_index='fair-lens', chunk_size=2000, chunk_overlap=200, title=None, author=None, document_version=1):
        """
        Embed and load a specific PowerPoint (.pptx) into the specified Pinecone index with chunking and slide numbers.
        
        Parameters:
            pc: Pinecone connection instance.
            pptx_file_name (str): Name of the PowerPoint file to process.
            embeddings_model: Embedding model instance.
            pinecone_index (str): Name of the Pinecone index.
            chunk_size (int): Maximum number of tokens per chunk.
            chunk_overlap (int): Number of overlapping tokens between chunks.
            title (str): Title of the document.
            author (str): Author of the document.
            document_version (int): Document version number.
        
        Returns:
            pd.DataFrame: DataFrame of metadata for successfully upserted vectors.
        """
        logger = logging.getLogger()
        logger.info(f"Starting the process to load PowerPoint '{pptx_file_name}' into Pinecone.")
        print(f"Starting the process to load PowerPoint '{pptx_file_name}' into Pinecone.")
    
        # Create or connect to the Pinecone vector store
        try:
            pinecone_store = PineconeStore(index_name=pinecone_index, embedding=embeddings_model)
            logger.info(f"Connected to Pinecone vector store for index: {pinecone_index}.")
            print(f"Connected to Pinecone vector store for index: {pinecone_index}.")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone vector store: {e}")
            print(f"Error connecting to Pinecone vector store: {e}")
            raise
    
        # Ensure the archive folder exists
        if not os.path.exists(self.archive_path):
            os.makedirs(self.archive_path)
    
        # Initialize text splitter
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
        # List to store metadata for successfully loaded content
        metadata_list = []
        load_datetime = datetime.now().isoformat()  # Timestamp for when the vectors are loaded
    
        # Process the specified PowerPoint file
        pptx_path = os.path.join(self.documents_path, pptx_file_name)
        if not os.path.exists(pptx_path):
            logger.error(f"PowerPoint file '{pptx_file_name}' does not exist in the documents folder.")
            print(f"Error: PowerPoint file '{pptx_file_name}' does not exist in the documents folder.")
            return pd.DataFrame()  # Return an empty DataFrame
    
        try:
            print(f"Processing PowerPoint: {pptx_file_name}")
            logger.info(f"Processing PowerPoint: {pptx_file_name}")
    
            # Extract text from .pptx file (direct XML parsing)
            slide_texts = []
            with zipfile.ZipFile(pptx_path, 'r') as pptx_zip:
                for item in pptx_zip.namelist():
                    if item.startswith('ppt/slides/slide') and item.endswith('.xml'):
                        with pptx_zip.open(item) as slide_file:
                            tree = ET.parse(slide_file)
                            root = tree.getroot()
                            namespace = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
                            text_elements = root.findall('.//a:t', namespace)
                            slide_text = " ".join(t.text for t in text_elements if t.text)
                            slide_texts.append(slide_text)
    
            # Process each slide and extract chunks with slide number
            for slide_num, slide_text in enumerate(slide_texts, start=1):
                chunks = text_splitter.split_text(slide_text)
    
                for chunk_idx, chunk in enumerate(chunks):
                    vector_id = f"{pptx_file_name}_slide_{slide_num}_chunk_{chunk_idx}"
                    metadata = {
                        "document_type"     : "pptx",
                        "title"             : title if title else pptx_file_name,
                        "author"            : author if author else "Unknown",  # Default to "Unknown" if not provided
                        "source"            : pptx_path,
                        "published_at"      : None,  # Not applicable for most PowerPoints
                        "summary"           : "",  # Summarization can be added later
                        "file_name"         : pptx_file_name,
                        "page_number"       : slide_num,  # Store the slide number
                        "table_name"        : None,
                        "record_id"         : None,
                        "generation_date"   : datetime.now().strftime('%Y-%m-%d'),
                        "month_year"        : datetime.now().strftime('%B %Y'),
                        "document_version"  : document_version,
                        "url"               : None,
                        "load_datetime"     : load_datetime
                    }
    
                    try:
                        # Check if the vector already exists
                        results = pinecone_store.similarity_search_with_score(chunk, k=1)
                        if results and results[0][1] > 0.9:  # Skip similar chunks
                            logger.info(f"Vector ID {vector_id} already exists. Skipping upsertion.")
                            print(f"Chunk {chunk_idx} of slide {slide_num} already exists. Skipping.")
                            continue
    
                        # Add text and metadata to Pinecone
                        pinecone_store.add_texts(
                            texts=[chunk],
                            metadatas=[{k: v if v is not None else "" for k, v in metadata.items()}],
                            ids=[vector_id]
                        )
                        logger.info(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        print(f"Successfully upserted vector with ID {vector_id} into Pinecone.")
                        metadata_list.append(metadata)
    
                    except Exception as e:
                        logger.error(f"Error upserting vector ID {vector_id} into Pinecone: {e}")
                        print(f"Error upserting chunk {chunk_idx} of slide {slide_num}: {e}")
    
            # Move the processed PowerPoint to the archive folder
            archive_pptx_path = os.path.join(self.archive_path, pptx_file_name)
            os.rename(pptx_path, archive_pptx_path)
            logger.info(f"Moved PowerPoint to archive: {archive_pptx_path}")
            print(f"Moved PowerPoint to archive: {archive_pptx_path}")
    
        except Exception as e:
            logger.error(f"Error processing PowerPoint {pptx_file_name}: {e}")
            print(f"Error processing PowerPoint {pptx_file_name}: {e}")
    
        # Create a DataFrame from the metadata list
        metadata_df = pd.DataFrame(metadata_list)
        logger.info(f"Completed loading PowerPoint '{pptx_file_name}' into Pinecone.")
        print(f"Completed loading PowerPoint '{pptx_file_name}' into Pinecone.")
        return metadata_df

   def load_file_to_pinecone(self, file_name, embeddings_model, pinecone_index='fair-lens', chunk_size=3000, chunk_overlap=200, title=None, author=None, document_version=1):
        """
        Load a file into Pinecone based on its type (PDF or PPTX).
    
        Parameters:
            file_name (str): Name of the file to process.
            embeddings_model: Embedding model instance.
            pinecone_index (str): Name of the Pinecone index.
            chunk_size (int): Maximum number of tokens per chunk (for PDF and PPTX).
            chunk_overlap (int): Number of overlapping tokens between chunks (for PDF and PPTX).
            title (str): Title of the document.
            author (str): Author of the document.
            document_version (int): Document version number.
    
        Returns:
            pd.DataFrame: DataFrame of metadata for successfully upserted vectors.
        """
        logger = logging.getLogger()
        file_extension = os.path.splitext(file_name)[1].lower()
    
        try:
            if file_extension == '.pdf':
                logger.info(f"File type detected: PDF. Loading '{file_name}' into Pinecone.")
                print(f"File type detected: PDF. Loading '{file_name}' into Pinecone.")
                metadata_df = self.load_pdf_to_pinecone(
                    pdf_file_name=file_name,
                    embeddings_model=embeddings_model,
                    pinecone_index=pinecone_index,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    title=title,
                    author=author,
                    document_version=document_version
                )
            elif file_extension == '.pptx':
                logger.info(f"File type detected: PPTX. Loading '{file_name}' into Pinecone.")
                print(f"File type detected: PPTX. Loading '{file_name}' into Pinecone.")
                metadata_df = self.load_pptx_to_pinecone(
                    pptx_file_name=file_name,
                    embeddings_model=embeddings_model,
                    pinecone_index=pinecone_index,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    title=title,
                    author=author,
                    document_version=document_version
                )
            else:
                logger.warning(f"Unsupported file type '{file_extension}'. Skipping file '{file_name}'.")
                print(f"Unsupported file type '{file_extension}'. Skipping file '{file_name}'.")
                return pd.DataFrame()  # Return an empty DataFrame for unsupported file types
    
            logger.info(f"Successfully loaded file '{file_name}' into Pinecone.")
            print(f"Successfully loaded file '{file_name}' into Pinecone.")
            return metadata_df
    
        except Exception as e:
            logger.error(f"Error processing file '{file_name}': {e}")
            print(f"Error processing file '{file_name}': {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of failure


    
if __name__ == "__main__":
    from llm_activity import llm_activity_class
    from fetch_news_articles import fetch_news_class
    
    
    
    llm_activity_class_obj = llm_activity_class()
    llm, embeddings_model  = llm_activity_class_obj.get_llm_embedding()
    
    news_fetcher = fetch_news_class()
    #fair_lending_articles_df = news_fetcher.filter_fair_lending_articles(llm)
    fair_lending_articles_df = pd.read_csv(r"C:\GenAI\FairLens\research\updated_fair_lending_articles.csv")
    
    vector_database_activity_class_obj = vector_database_activity_class()
    pc = vector_database_activity_class_obj.connect_to_pinecone()
    #article_metadata_df = vector_database_activity_class_obj.load_articles_to_pinecone( fair_lending_articles_df, embeddings_model, pinecone_index='fair-lens', chunk_size=3000, chunk_overlap=200)
    
    # pdf_file_name = "201409_cfpb_report_proxy-methodology.pdf"
    # pdf_metadata_df = vector_database_activity_class_obj.load_pdf_to_pinecone( pdf_file_name, embeddings_model, pinecone_index='fair-lens', chunk_size=2000, chunk_overlap=200)
    
    # pptx_file_name = "cfpb_arc-meeting_implementing-dodd-frank-1071_presentation_2020-11.pptx"
    # pptx_metadata_df = vector_database_activity_class_obj.load_pptx_to_pinecone(  pptx_file_name, embeddings_model, pinecone_index='fair-lens', chunk_size=2000, chunk_overlap=200)
    
    
    file_name = "Regulation F.pdf"  # or "example_presentation.pptx"
    metadata_df = vector_database_activity_class_obj.load_file_to_pinecone(
        file_name=file_name,
        embeddings_model=embeddings_model,
        pinecone_index='fair-lens',
        chunk_size=3000,
        chunk_overlap=200,
        title="DEBT COLLECTION PRACTICES (REGULATION F)",
        author="CFPB",
        document_version=1
    )
    
    