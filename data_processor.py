import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from rich.logging import RichHandler

DATABASE = 'lottery_db.sqlite'
CSV_FILE = 'data/Lottery_Powerball_Winning_Numbers__Beginning_2010.csv'


class CSVProcessor:
    def __init__(self, database_path, csv_file):
        self.database_path = database_path
        self.csv_file = csv_file
        self.setup_logging()

    @staticmethod
    def setup_logging():
        logger.remove()
        logger.add(RichHandler(), level="INFO")

    def create_db(self):
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS csvs
                                 (id INTEGER PRIMARY KEY, data TEXT, embeddings TEXT)''')
            logger.info("Database created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating database: {e}")

    @staticmethod
    def generate_embeddings(text):
        try:
            vectorizer = TfidfVectorizer()
            return vectorizer.fit_transform([text]).toarray()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def store_csv_data(self, data, embeddings):
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("INSERT INTO csvs (data, embeddings) VALUES (?, ?)",
                             (data, str(embeddings)))
            logger.info(f"CSV data stored.")
        except sqlite3.Error as e:
            logger.error(f"Error storing CSV data: {e}")

    def process_csv_file(self):
        try:
            logger.info("Processing CSV file.")
            df = pd.read_csv(self.csv_file)
            for index, row in df.iterrows():
                data = ', '.join(str(item) for item in row)
                embeddings = self.generate_embeddings(data)
                if embeddings is not None:
                    self.store_csv_data(data, embeddings)
            logger.info(f"CSV processed: {os.path.basename(self.csv_file)}")
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")


if __name__ == '__main__':
    csv_processor = CSVProcessor(DATABASE, CSV_FILE)
    csv_processor.create_db()
    csv_processor.process_csv_file()
