import os
import time
import logging
import pandas as pd

from logger import Logger
from llm_activity import llm_activity_class
from sql_database_activity import sql_database_activity_class
from vector_database_activity import vector_database_activity_class
from prompt_retriever_chain import prompt_retrieval_chain_class


import warnings
warnings.filterwarnings('ignore')

class fairlens_intersect_class:
   def __init__(self):
       self.fairlens_path                      = os.path.dirname(os.path.abspath("fairlens.py"))
       self.version                            = 1.0
       self.user_id                            = 'shrikanth_mahale'
       self.llm_activity_class_obj             = llm_activity_class()
       self.sql_database_activity_class_obj    = sql_database_activity_class()
       self.vector_database_activity_class_obj = vector_database_activity_class()
       self.prompt_retrieval_chain_class_obj   = prompt_retrieval_chain_class()
       self.logger_obj                         = Logger() 
       self.logger_obj.logger() # Initialize the logger

   def fairlens(self, user_id, password):
       start_time            = time.time()
       logger                = logging.getLogger()
       ############################################################# Connecting to SQL Server ################################################################
       conn, db              = self.sql_database_activity_class_obj.connect_to_database()
       
       ############################################################# Fetching User Info ######################################################################
       user_info_df      = pd.read_sql_query("Select * from [FairLens].[dbo].[fairlens_user_profile] where user_id = '{}'".format(self.user_id),conn)
       if len(user_info_df) == 1:
           user_id           = user_info_df.loc[:,'User_ID'][0]
           password          = user_info_df.loc[:,'Password'][0]
           user_first_name   = user_info_df.loc[:,'First_Name'][0]
           logger.info("First Name: {}".format(user_first_name))
           user_last_name    = user_info_df.loc[:,'Last_Name'][0]
           logger.info("Last Name: {}".format(user_last_name))
           permissions       = user_info_df.loc[:,'Permissions'][0]
           logger.info("Permissions: {}".format(permissions))
           email_address     = user_info_df.loc[:,'EmailAddress'][0]
           logger.info("Email Address: {}".format(email_address))
       else:
           logger.error("Invalid User")
           
       
      
       if permissions == 1:
           print("User {} has the permissions to use FairLens ChatBot.")
           logger.info("User {} has the permissions to use FairLens ChatBot.")
           
           ############################################################# Generating Log Id for For the user ########################################################
           table_action          = 'insert'
           update_column         = None
           update_column_value   = None
           fairlens_log_id       = self.sql_database_activity_class_obj.fairlens_log_entry(conn,table_action,update_column,update_column_value,self.version,user_id,fairlens_log_id = None)
           
           ############################################################# Fetching LLM and Embedding Model ########################################################
           llm, embeddings_model = self.llm_activity_class_obj.get_llm_embedding()
           
           ############################################################# Prompt Template and Retriever for text chain  ########################################################
           pinecone_index        = "fair-lens"
           prompt_template       = self.prompt_retrieval_chain_class_obj.create_prompt_template()
           retriever             = self.prompt_retrieval_chain_class_obj.create_retriever(pinecone_index, embeddings_model, k=3)
           
           ############################################################# Prompt Template and Retriever for text chain  ########################################################
           memory_window_size    = 10
           combined_chain        = self.prompt_retrieval_chain_class_obj.create_combined_chain(user_id, llm, retriever, db, memory_window_size = 10)
           
       else:
           print("User {} does not have the permissions to use FairLens ChatBot.")
           logger.info("User {} does not have the permissions to use FairLens ChatBot.")
       
       conn.dispose()
       print("Connection to the FairLens Database in SQL Server has been closed.")
       logger.info("Connection to the FairLens Database in SQL Server has been closed.")
       elapsed_time = (time.time() - start_time) / 60
       print('Time taken to run this code {} mins'.format(elapsed_time))
       logger.info('Time taken to run this code {} mins'.format(elapsed_time))
       return None
# Example usage
if __name__ == "__main__":
    fairlens_instance = fairlens_intersect_class()
    fairlens_instance.fairlens()

