import os
import logging
from datetime import datetime

class Logger:
    def __init__(self):
        """Initialize the logger with a unique log file."""
        # Create the logs directory if it doesn't exist
        self.log_path = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_path, exist_ok=True)
        
        # Define the log file name with the current timestamp
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_filepath = os.path.join(self.log_path, log_file)

    def logger(self):
        """Sets up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level to INFO
            filename=self.log_filepath,
            format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        )
        logging.info("Logger initialized successfully.")
        print(f"Logging initialized. Logs are being saved to {self.log_filepath}")

# Example usage
if __name__ == "__main__":
    # Create an instance of the Logger class
    logger_instance = Logger()
    
    # Initialize the logger
    logger_instance.logger()

    # Test logging
    #logging.info("This is an info message.")
    #logging.warning("This is a warning message.")
    #logging.error("This is an error message.")