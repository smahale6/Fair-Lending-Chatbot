from langchain.chat_models import AzureChatOpenAI
`from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import logging
import os
from logger import Logger

import warnings
warnings.filterwarnings('ignore')

class llm_activity_class:
   def __init__(self):
       self.logger_obj     = Logger() 
       
   def get_llm_embedding(self):
      """Connect to Azure OpenAI LLM and embeddings, and return them."""
      logger                             = logging.getLogger()
      print("Loading environment variables for llm and embedding model.")
      logger.info("Loading environment variables for llm and embedding model.")
      load_dotenv()  # Load environment variables from the .env file

      # Fetch environment variables
      deployment_name                    = "CART"
      AZURE_OPENAI_API_KEY               = os.getenv("AZURE_OPENAI_API_KEY")
      AZURE_OPENAI_ENDPOINT              = os.getenv("AZURE_OPENAI_API_BASE")
      AZURE_OPENAI_API_VERSION           = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")
      AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING")

      # Set up environment variables for Azure OpenAI
      os.environ["OPENAI_API_VERSION"]    = AZURE_OPENAI_API_VERSION
      os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
      os.environ["AZURE_OPENAI_API_KEY"]  = AZURE_OPENAI_API_KEY
      logger.info("Loaded environment variables for llm and embedding model.")
      print("Loaded environment variables for llm and embedding model.")

      try:
          print("Establishing connection with GPT-4 Turbo OpenAI LLM.")
          logger.info("Establishing connection with GPT-4 Turbo OpenAI LLM.")
          llm = AzureChatOpenAI(
                                deployment_name = deployment_name,
                                temperature     = 0.0
                               )
          logger.info("Successfully connected to GPT-4 Turbo OpenAI LLM.")
          print("Successfully connected to GPT-4 Turbo OpenAI LLM.")
      except Exception as e:
          logger.error(f"Error connecting to LLM: {e}")
          print(f"Error connecting to LLM: {e}")
          raise

      try:
          print("Fetching GPT-4 OpenAI Embeddings.")
          logger.info("Fetching GPT-4 OpenAI Embeddings.")
          embeddings_model = AzureOpenAIEmbeddings(
                                                    model              = "CART_Embedding",
                                                    azure_endpoint     = AZURE_OPENAI_ENDPOINT,
                                                    api_key            = AZURE_OPENAI_API_KEY,
                                                    openai_api_version = AZURE_OPENAI_API_VERSION_EMBEDDING
                                                  )
          logger.info("Successfully fetched GPT-4 OpenAI Embeddings.")
          print("Successfully fetched GPT-4 OpenAI Embeddings.")
      except Exception as e:
          logger.error(f"Error fetching embeddings: {e}")
          print(f"Error fetching embeddings: {e}")
          raise
      return llm, embeddings_model
  
if __name__ == "__main__":
    llm_activity_class_obj = llm_activity_class()
  