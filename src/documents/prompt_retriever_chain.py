from langchain import PromptTemplate, LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore as PineconeStore
from logger import Logger
import logging

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from operator import itemgetter

import warnings
warnings.filterwarnings('ignore')



class prompt_retrieval_chain_class:
    def __init__(self):
        self.logger_obj       = Logger()
        self.prompt_template  = self.create_prompt_template()
        self.user_memory_dict = {}  # Dictionary to store user-specific memories

    def create_prompt_template(self):
        """
        Create a PromptTemplate using the saved prompt for FairLens use case.
        
        Returns:
            PromptTemplate: A LangChain prompt template object.
        """
        logger = logging.getLogger()
        logger.info("Creating the PromptTemplate for the RetrievalQA chain.")
        print("Creating the PromptTemplate for the RetrievalQA chain.")
        
        try:
            # Saved prompt for FairLens
            prompt_text = """
                                You are a knowledgeable and avery friendly assistant helping with Fair Lending analysis. You have access to the following types of data sources, each with detailed metadata:
                                
                                1. **Articles, PDFs, and PPTs**:
                                   - `document_type`: Type of document (e.g., article, pdf, pptx).
                                   - `title`: Title of the document.
                                   - `author`: Author of the document.
                                   - `source`: Path or URL of the document.
                                   - `published_at`: Publication date, if available.
                                   - `summary`: Concise summary of the content.
                                   - `file_name`: Name of the file.
                                   - `page_number`: Page or slide number (if applicable).
                                   - `generation_date`: Date when this data was ingested into the system.
                                   - `month_year`: Month and year of ingestion.
                                   - `document_version`: Version of the document.
                                
                                
                                ### Your Task:
                                1. **Identify the Data Source**: 
                                   Based on the question, determine whether the answer is likely to come from:
                                   - Articles, PDFs, or PPTs (if the question references documents or summaries).
                                   - SQL Table (`UW_Odds_Ratio`) (if the question involves structured data like years, areas, or demographic analysis).
                                
                                2. **Retrieve Relevant Information**:
                                   - If the query references a document title, file name, or author, retrieve the document and summarize its content.
                                   - If the query involves numerical data (e.g., odds ratios, years, quarters, or segments), retrieve it from the SQL table.
                                
                                3. **Dynamic Responses**:
                                  
                                   - If the query is incomplete, ask clarifying questions to ensure accurate retrieval.
                                
                                
                                Now, answer the query by identifying the appropriate data source and retrieving the relevant information.
                                {context}
                                
                                Question: {question}
                                
                                You need to answer the question in the language it is asked.

                        """
            
            # Define the prompt template
            prompt_template = PromptTemplate(
                                                input_variables=["question"],
                                                template=prompt_text.strip()
                                            )
            logger.info("Successfully created the PromptTemplate.")
            print("Successfully created the PromptTemplate.")
            return prompt_template
        except Exception as e:
            logger.error(f"Error creating the PromptTemplate: {e}")
            print(f"Error creating the PromptTemplate: {e}")
            raise
            
    def create_retriever(self, pinecone_index, embeddings_model, k=3):
        """
        Create a retriever using PineconeVectorStore.

        Parameters:
            pinecone_index (str): The name of the Pinecone index.
            embeddings_model: Embedding model instance (e.g., OpenAIEmbeddings).
            k (int): Number of results to return for each query.

        Returns:
            retriever: A retriever instance created from the specified Pinecone index.
        """
        logger = logging.getLogger()
        logger.info("Creating the retriever for Pinecone index.")
        print("Creating the retriever for Pinecone index.")

        try:
            # Initialize Pinecone Store
            pinecone_store = PineconeStore(index_name=pinecone_index, embedding=embeddings_model)

            # Create retriever using PineconeStore
            retriever = pinecone_store.as_retriever(search_kwargs={"k": k})

            logger.info(f"Successfully created retriever for Pinecone index: {pinecone_index}.")
            print(f"Successfully created retriever for Pinecone index: {pinecone_index}.")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            print(f"Error creating retriever: {e}")
            raise
            
    def get_user_memory(self, user_id, memory_window_size=5):
        """
        Retrieve or create a memory instance for a specific user.

        Parameters:
            user_id (str): The unique identifier for the user.
            memory_window_size (int): Number of interactions to remember in memory.

        Returns:
            ConversationBufferWindowMemory: A memory instance for the user.
        """
        if user_id not in self.user_memory_dict:
            # Create a new memory instance for the user
            self.user_memory_dict[user_id] = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=memory_window_size,
                return_messages=True
            )
            
        return self.user_memory_dict[user_id]
    
    def create_text_chain(self, user_id, llm, retriever, memory_window_size=5):
        """
        Create a ConversationalRetrievalChain with limited memory for a specific user.

        Parameters:
            user_id (str): The unique identifier for the user.
            llm: The language model instance (e.g., OpenAI or Azure GPT).
            retriever: The retriever instance (e.g., from Pinecone or other vector databases).
            memory_window_size (int): Number of interactions to remember in memory.

        Returns:
            ConversationalRetrievalChain: A LangChain ConversationalRetrievalChain instance.
        """
        logger = logging.getLogger()
        logger.info(f"Creating the ConversationalRetrievalChain with memory for user: {user_id}.")
        print(f"Creating the ConversationalRetrievalChain with memory for user: {user_id}.")

        # Get user-specific memory with custom memory_window_size
        memory = self.get_user_memory(user_id, memory_window_size)

        # Define the chain with memory
        chain = ConversationalRetrievalChain.from_llm(
                                                        llm                       = llm,
                                                        retriever                 = retriever,
                                                        memory                    = memory,
                                                        combine_docs_chain_kwargs = {"prompt": self.prompt_template},
                                                        output_key                = "answer"  # Specify the key to store in memory
                                                    )

        logger.info(f"Successfully created the ConversationalRetrievalChain for user: {user_id}.")
        print(f"Successfully created the ConversationalRetrievalChain for user: {user_id}.")
        return chain
    
    
    def create_sql_chain(self, user_id, llm,db, memory_window_size=5):
        write_query = create_sql_query_chain(llm, db)
        execute_query = QuerySQLDataBaseTool(db=db)
        
        answer_prompt = PromptTemplate.from_template(
            """
                Given the SQL query and its results, provide the answer to the user's question. 
                Give a short answer. Present the explanation in a user-friendly manner.
                Answer the questions only if it is available in dbo.UW_Odds_Ratio table.
        
                Given below is the explanation of each field in the dbo.UW_Odds_Ratio table.
                year: Represents the year in which the loans and applications were issued.
                quarter : The Quarter in which the loans and applications were issued. There are 4 quarters i.e. Q1 ,Q2, Q3, Q4.
                area: Area represents the business unit or type of loans. Like Mortgage, Consumer Lending , Education Lending etc.
                product: One area can have one or multiple product.
                segment: One product can have multiple segments.
                demographics: Demographics represents race/ethnicity of an applicant
                
                If there are multiple rows in the result, explain why there are multiple rows 
                and which columns differentiate them. Present the explanation in a user-friendly manner.
                
        
                Question: {question}
                SQL Query: {query}
                SQL Result: {result}
                Answer: 
            """
        )

        answer = answer_prompt | llm | StrOutputParser()
        
        chain = (
                     RunnablePassthrough.assign(query=write_query).assign
                    (
                         result=itemgetter("query") | execute_query
                    )
                    | answer
                )
        return chain

    
    

    def create_combined_chain(self, user_id, llm, retriever, db, memory_window_size=10):
        """
        Create a combined chain that routes questions to either the text chain or the SQL chain dynamically.
        """
        logger = logging.getLogger()
        logger.info(f"Creating the combined chain with memory for user: {user_id}.")
        print(f"Creating the combined chain with memory for user: {user_id}.")
    
        # Create individual chains
        text_chain = self.create_text_chain(user_id, llm, retriever, memory_window_size)
        sql_chain = self.create_sql_chain(user_id, llm, db, memory_window_size)
    
        # Define router prompt
        router_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
                You are a routing assistant. Decide the type of chain to route the query to.
    
                If the question references:
                - Regulations, general information, or documents like articles, PDFs, or PPTs -> route to "text".
                
                If the question is regarding odds ratios, focal points then  -> route to "sql".
                If Structured data (like odds ratios, years, demographics, or SQL keywords) -> route to "sql".
                
                If the question is ambigous then -> route to "text"
    
                Question: {input}
                Route to: (answer with either "text" or "sql")
            """
        )
    
        router_chain = LLMChain(llm=llm, prompt=router_prompt, output_key="route")
    
        def combined_chain_run(inputs):
            """
            Custom logic to execute the combined chain by routing the query.
            """
            logger.info(f"Routing the query: {inputs['input']}")
        
            # Step 1: Use the router chain to decide
            routing_decision = router_chain.run({"input": inputs["input"]}).strip()
        
            # Step 2: Execute the appropriate chain
            if routing_decision == "sql":
                logger.info(f"Routing to SQL chain for query: {inputs['input']}")
                return sql_chain.invoke({"question": inputs["input"]})
            elif routing_decision == "text":
                logger.info(f"Routing to Text chain for query: {inputs['input']}")
                return text_chain.run({"question": inputs["input"]})
            else:
                logger.warning("Router could not determine the chain. Returning default response.")
                return "Unable to determine the appropriate chain for this query."
    
        logger.info(f"Successfully created the combined chain with memory for user: {user_id}.")
        print(f"Successfully created the combined chain with memory for user: {user_id}.")
    
        return combined_chain_run
    
    
if __name__ == "__main__":
    from llm_activity import llm_activity_class
    llm_activity_class_obj = llm_activity_class()
    llm, embeddings_model  = llm_activity_class_obj.get_llm_embedding()
    
    from sql_database_activity import sql_database_activity_class
    sql_database_activity_class_obj = sql_database_activity_class()
    conn, db = sql_database_activity_class_obj.connect_to_database()
    
    pinecone_index = "fair-lens"
    prompt_retrieval_chain_class_obj = prompt_retrieval_chain_class()
    prompt_template        = prompt_retrieval_chain_class_obj.create_prompt_template()
    retriever              = prompt_retrieval_chain_class_obj.create_retriever(pinecone_index, embeddings_model, k=3)
    
    # Create the prompt chain object
    prompt_chain_obj = prompt_retrieval_chain_class()

    # Create a combined chain
    user_id = "user_1"
    combined_chain = prompt_chain_obj.create_combined_chain(user_id, llm, retriever, db, memory_window_size=10)

    # Query examples
    response1 = combined_chain({"input": "What is regulation AA?"})
    print(' ')
    print(' ')
    print(' ')
    print(f"Response 1: {response1}")


    response2 = combined_chain({"input": "What is the credit model odds ratio for all Consumer Lending Direct Auto for African American?"})
    print(' ')
    print(' ')
    print(' ')
    print(f"Response 2: {response2}")
    
    
    