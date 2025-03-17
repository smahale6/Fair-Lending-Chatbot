import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
from langchain import PromptTemplate, LLMChain
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from logger import Logger

import warnings
warnings.filterwarnings('ignore')

class fetch_news_class:
    def __init__(self):
        """Initialize the logger."""
        self.logger_obj = Logger()  # Initialize Logger instance

    def fetch_articles(self):
        """Fetch fair lending related news articles using NewsAPI."""
        logger = logging.getLogger()
        # Pull the NEWS_API_KEY from environment variables
        logger.info("Loading NEWS_API_KEY from environment variables.")
        NEWS_API_KEY = os.getenv('NEWS_API_KEY')

        # Check if API key is available
        if not NEWS_API_KEY:
            logger.error("API key not found. Please set the 'NEWS_API_KEY' environment variable.")
            print("Error: Please set the 'NEWS_API_KEY' environment variable.")
            raise ValueError("Please set the 'NEWS_API_KEY' environment variable.")

        logger.info("NEWS_API_KEY loaded successfully.")

        # Define the base URL for NewsAPI
        url = 'https://newsapi.org/v2/everything'
        logger.info(f"Using NewsAPI endpoint: {url}")

        # Get the current date and the date from 7 days ago in YYYY-MM-DD format
        current_date   = datetime.now().strftime('%Y-%m-%d')
        last_week_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        logger.info(f"Fetching articles from {last_week_date} to {current_date}")

        # Define the search query with specific keywords and phrases
        search_query = (
                            '("fair lending" OR "disparate treatment" OR "overt discrimination" OR '
                            '"redlining" OR "mortgage discrimination" OR "banking discrimination" OR '
                            '"racial discrimination in banking" OR "lending bias" OR "credit discrimination")'
                        )

        # Set parameters for the request
        params = {
                    'q': search_query,
                    'from': last_week_date,
                    'to': current_date,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': NEWS_API_KEY
                }

        try:
            logger.info("Making request to NewsAPI.")
            print("Fetching articles from NewsAPI...")
            response = requests.get(url, params=params)

            # Handle the response
            if response.status_code == 200:
                logger.info("Successfully fetched articles from NewsAPI.")
                data     = response.json()
                articles = data.get('articles', [])
                
                # Extract relevant metadata and store it in a DataFrame
                if articles:
                    logger.info(f"Found {len(articles)} articles.")
                    articles_data = []

                    for article in articles:
                        article_data = {
                                            'Title': article.get('title'),
                                            'Author': article.get('author'),
                                            'Source': article.get('source', {}).get('name'),
                                            'Description': article.get('description'),
                                            'URL': article.get('url'),
                                            'Published At': article.get('publishedAt'),
                                        }

                        # Scrape the full content of the article
                        article_url = article.get('url')
                        try:
                            page            = requests.get(article_url)
                            soup            = BeautifulSoup(page.content, 'html.parser')

                            # Extract paragraphs from the content
                            paragraphs      = soup.find_all('p')
                            raw_content     = ' '.join([para.get_text() for para in paragraphs])

                            # Clean the text by removing non-ASCII characters
                            cleaned_content = ''.join(filter(lambda x: x in set(map(chr, range(32, 127))), raw_content))

                            article_data['Content'] = cleaned_content
                        except Exception as e:
                            logger.error(f"Error fetching content from URL: {article_url}. Error: {e}")
                            print(f"Error fetching content from URL: {article_url}. Error: {e}")
                            article_data['Content'] = f"Error fetching content: {e}"
                        articles_data.append(article_data)
                    # Create a DataFrame from the list of dictionaries
                    articles_df = pd.DataFrame(articles_data)
                    return articles_df
                else:
                    logger.info("No articles found for the last week.")
                    print("No articles found for the last week.")
                    return pd.DataFrame()  # Return an empty DataFrame if no articles found
            else:
                logger.error(f"Failed to fetch articles. Status code: {response.status_code}")
                raise Exception(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error during API request: {e}")
            print(f"Error during API request: {e}")
            raise
            

    def evaluate_article_content(self, article_content, llm_chain):
        """Evaluate if the article content is related to fair lending."""
        logger = logging.getLogger()
        try:
            # Split the content into manageable chunks
            text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
            chunks        = text_splitter.split_text(article_content)

            
            # Evaluate each chunk
            for chunk in chunks:
                response = llm_chain.run({"article_content": chunk})
                if "yes" in response.lower():
                    return True  # Mark as related if any chunk is related

            return False  # Mark as unrelated if no chunks are related
        except Exception as e:
            logger.error(f"Error while evaluating the article: {e}")
            print(f"Error while evaluating the article: {e}")
            return False

    def summarize_article_content(self, article_content,llm):
        """Summarize the article content using LLM."""
        logger               = logging.getLogger()
        try:
            summary_prompt   = (
                                "Please provide a concise summary of the following article in 4-5 sentences, "
                                "focusing on the Fair Lending aspects:\n\n{article_content}"
                               )
            summary_template = PromptTemplate(input_variables=["article_content"], template=summary_prompt)
            summary_chain    = LLMChain(llm=llm, prompt=summary_template)
            response         = summary_chain.run({"article_content": article_content})
            return response
        except Exception as e:
            logger.error(f"Error while summarizing the article: {e}")
            print(f"Error while summarizing the article: {e}")
            return "Error generating summary"

    def filter_fair_lending_articles(self, llm):
        """Filter and summarize fair lending articles."""
        logger = logging.getLogger()
        articles_df = self.fetch_articles()
        if articles_df.empty:
            logger.error("The articles DataFrame is empty.")
            print("The articles DataFrame is empty.")
            return pd.DataFrame()
        # Define the prompt template for evaluation
        prompt_template = (
                            "Determine if the following article content is related to fair lending practices, "
                            "a financial institution, a bank, CFPB regulations, or any type of discrimination. "
                            "Please respond with 'Yes' if it is related, or 'No' if it is not:\n\n{article_content}"
                          )
        prompt                         = PromptTemplate(input_variables=["article_content"], template=prompt_template)
        llm_chain                      = LLMChain(llm=llm, prompt=prompt)
        # Filter articles
        articles_df['Is_Related']      = articles_df['Content'].apply(lambda x: self.evaluate_article_content(x, llm_chain))
        # Summarize related articles
        articles_df['Content_Summary'] = articles_df.apply(lambda row: self.summarize_article_content(row['Content'], llm) if row['Is_Related'] else "", axis=1)
        # Filter only related articles for the final DataFrame
        fair_lending_articles          = articles_df.loc[articles_df['Is_Related'] == True, :]
        print("A total of {} Fair Lending articles have been identified.".format(len(fair_lending_articles)))
        logger.info("A total of {} Fair Lending articles have been identified.".format(len(fair_lending_articles)))
        return fair_lending_articles
            
    
# Example usage
if __name__ == "__main__":
    from llm_activity import llm_activity_class
    llm_activity_class_obj = llm_activity_class()
    llm, embeddings_model  = llm_activity_class_obj.get_llm_embedding()
    news_fetcher = fetch_news_class()
    fair_lending_articles = news_fetcher.filter_fair_lending_articles(llm)
    
