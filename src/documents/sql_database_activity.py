import pandas as pd
import pyodbc
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from dotenv import load_dotenv
import os
import urllib
from logger import Logger
from langchain.sql_database import SQLDatabase
from sqlalchemy.orm import Session
from sqlalchemy import update
from sqlalchemy import text

import warnings
warnings.filterwarnings('ignore')

class sql_database_activity_class:
   def __init__(self):
       self.logger_obj     = Logger() 
       
   def connect_to_database(self):
        logger = logging.getLogger()
    
        # Database connection details
        server = 'DESKTOP-VONKKUH'
        database = 'FairLens'
        driver = 'ODBC Driver 17 for SQL Server'
    
        try:
            print("Establishing connection to SQL Server.")
            logger.info("Connecting to SQL Server.")
    
            conn_str = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
                                                                                    f"DRIVER={{{driver}}};"
                                                                                    f"SERVER={server};"
                                                                                    f"DATABASE={database};"
                                                                                    "Trusted_Connection=yes;"
                                                                                )
            conn = create_engine(conn_str)
            db = SQLDatabase(conn)
            print("Established connection to the FairLens Database in SQL Server.")
            logger.info("Successfully connected to the FairLens Database.")
            return conn, db
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            logger.error(f"Error connecting to the database: {e}")
            raise
            
   def fairlens_log_entry(self,conn,table_action,update_column,update_column_value,version,user_id,fairlens_log_id = None):
       if table_action == 'insert':
           logger = logging.getLogger()
           print("Generating the Fairlens Log Id for this run for the user {}.".format(user_id))
           logger.info("Generating the Fairlens Log Id for this run for the user {}.".format(user_id))
           fairlens_log_df = pd.read_sql("select * from dbo.FairLens_Log",conn)
           if len(fairlens_log_df) == 0:
               fairlens_log_id       = 1000
           else:
               fairlens_log_df       = pd.read_sql("select max(FairLens_ID) as FairLens_ID from dbo.FairLens_Log",conn)
               fairlens_log_id       = fairlens_log_df.loc[0,'fairlens_log_id'] + 1
           print("The Fairlens Log Id for this run is {}.".format(fairlens_log_id))
           fairlens_log                   = pd.DataFrame([{'FairLens_ID':fairlens_log_id,'Version':version}])
           fairlens_log['User_ID']        = user_id
           fairlens_log['Log_Date']       = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
           fairlens_log['Total_Prompts']  = 0
           fairlens_log = fairlens_log.loc[:,["FairLens_ID","Log_Date","Total_Prompts","User_ID","Version"]]
           print("Creating a new log record for the fairlens log id {}.".format(fairlens_log_id))
           logger.info("Creating a new log record for the fairlens log id {}.".format(fairlens_log_id))
           try:
               fairlens_log.to_sql(name="FairLens_Log",con=conn,schema="dbo",if_exists="append",index=False)
               print("Record successfully appended to dbo.FairLens_Log.")
               logger.info("Record successfully appended to dbo.FairLens_Log.")
           except Exception as e:
               print(f"An error occurred: {e}")
               logger.error(f"An error occurred: {e}")
           return fairlens_log_id
       elif table_action == 'update' and type(update_column_value) == str:
            print("Updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            logger.info("Updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            with conn.connect() as connection:
                update_query = text(f"UPDATE dbo.FairLens_Log SET {update_column} = :update_column_value WHERE FairLens_ID = :fairlens_log_id")
                connection.execute(update_query,{"update_column_value": update_column_value, "fairlens_log_id": fairlens_log_id})
                connection.commit()
            print("Completed updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            logger.info("Completed updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            return None
       elif table_action == 'update' and type(update_column_value) == int:
            print("Updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            logger.info("Updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            with conn.connect() as connection:
                update_query = text(f"UPDATE dbo.FairLens_Log SET {update_column} = :update_column_value WHERE FairLens_ID = :fairlens_log_id")
                connection.execute(update_query,{"update_column_value": update_column_value, "fairlens_log_id": fairlens_log_id})
                connection.commit()
            print("Completed updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            logger.info("Completed updating the column {} in the dbo.FairLens_Log table.".format(update_column))
            return None
        
if __name__ == "__main__":
    sql_database_activity_class_obj = sql_database_activity_class()
    conn, db = sql_database_activity_class_obj.connect_to_database()
    
    
    
    