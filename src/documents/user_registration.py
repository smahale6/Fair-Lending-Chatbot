import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import sys
import os


from sql_database_activity import sql_database_activity_class


class user_registration_class:
    def __init__(self):
        self.sql_database_activity_class_obj = sql_database_activity_class()

    def insert_dataframe_to_db(self, df, table_name):
        """
        Insert the contents of a DataFrame into a specified SQL table.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing user data.
            table_name (str): The name of the SQL table.
        """
        conn, _ = self.sql_database_activity_class_obj.connect_to_database()
        if conn:
            try:
                # Use Pandas to insert the DataFrame into the SQL table
                df.to_sql(table_name, conn, if_exists="append", index=False)
                st.success("Thank you for registering. We will review your application and give you the permission to use FairLens")
            except Exception as e:
                st.error(f"An error occurred while inserting data: {e}")
            finally:
                conn.dispose()

    def create_user_registration_ui(self):
        """
        Render the Streamlit UI for user registration and load data into a DataFrame.
        """
        st.title("FairLens User Profile Form")

        # Input fields
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        email_address = st.text_input("Email Address")

        # Buttons
        if st.button("Submit"):
            if not user_id or not password or not first_name or not last_name or not email_address:
                st.error("All fields are required.")
            else:
                # Create a DataFrame from the form data
                data = {
                    "User_ID": [user_id],
                    "Password": [password],
                    "First_Name": [first_name],
                    "Last_Name": [last_name],
                    "Email_Address": [email_address],
                }
                df = pd.DataFrame(data)

                # Insert the DataFrame into the database
                self.insert_dataframe_to_db(df, "fairlens_user_profile")

        if st.button("Clear"):
            st.experimental_rerun()  # Reload the page to clear inputs


# Main Execution
if __name__ == "__main__":
    user_registration = user_registration_class()
    user_registration.create_user_registration_ui()
