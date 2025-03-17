import streamlit as st
import pandas as pd
from datetime import datetime
from sql_database_activity import sql_database_activity_class


class fairLens_login_class:
    def __init__(self):
        self.sql_database_activity_class_obj = sql_database_activity_class()

    def insert_log_record(self, conn, user_id, version):
        """
        Insert a new log record in the FairLens_Log table.
        Parameters:
            conn: SQLAlchemy connection object.
            user_id: The ID of the user logging in.
            version: The version of the system.
        Returns:
            fairlens_log_id: The generated log ID.
        """
        table_action = "insert"
        update_column = None
        update_column_value = None
        fairlens_log_id = self.sql_database_activity_class_obj.fairlens_log_entry(
            conn,
            table_action,
            update_column,
            update_column_value,
            version,
            user_id,
        )
        return fairlens_log_id

    def create_login_page(self):
        """
        Create a login page for FairLens.
        If the login is successful, insert a log record and return the FairLens_ID.
        """
        st.title("FairLens Login Page")

        # Login form
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if not user_id or not password:
                st.error("Please provide both User ID and Password.")
                return None

            # Connect to the database
            conn, db = self.sql_database_activity_class_obj.connect_to_database()

            try:
                # Validate user credentials
                user_query = f"""
                    SELECT * FROM dbo.Fairlens_User_Profile 
                    WHERE User_ID = '{user_id}' AND Password = '{password}'
                """
                user_info_df = pd.read_sql(user_query, conn)

                if len(user_info_df) == 1:
                    # Fetch user details
                    permissions = user_info_df.loc[0, "Permissions"]

                    if permissions == 1:
                        st.success(f"Welcome, {user_id}! You have permission to use FairLens.")
                        
                        # Insert a new log record
                        version = "1.0"  # Example version
                        fairlens_log_id = self.insert_log_record(conn, user_id, version)
                        st.info(f"Your session log ID is {fairlens_log_id}.")
                        return fairlens_log_id
                    else:
                        st.error("You do not have the required permissions to access FairLens.")
                        return None
                else:
                    st.error("Invalid User ID or Password.")
                    return None
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                conn.dispose()


# Main Execution
if __name__ == "__main__":
    fairlens_login = fairLens_login_class()
    fairlens_login.create_login_page()
