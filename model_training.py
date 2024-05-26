import logging
import sqlite3

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


class LotteryPredictor:
    def __init__(self, database, query, test_size=0.2, random_state=42):
        try:
            self.conn = sqlite3.connect(database)
            self.df = pd.read_sql_query(query, self.conn)
            self.conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            return
        except pd.errors.DatabaseError as e:
            logging.error(f"Error querying database: {e}")
            return

        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def preprocess_data(self):
        try:
            data_expanded = self.df['data'].str.split(',', expand=True)
            expected_columns = ['Draw Date', 'Winning Numbers', 'Multiplier']

            if data_expanded.shape[1] > len(expected_columns):
                data_expanded = data_expanded.iloc[:, :len(expected_columns)]
            elif data_expanded.shape[1] < len(expected_columns):
                raise ValueError(f"Expected {len(expected_columns)} columns, but got {data_expanded.shape[1]} columns")

            data_expanded.columns = expected_columns
            data_expanded = data_expanded.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            data_expanded['Winning Numbers'] = data_expanded['Winning Numbers'].apply(
                lambda x: list(map(int, x.split())))
            data_expanded['Multiplier'] = pd.to_numeric(data_expanded['Multiplier'], errors='coerce')
            data_expanded.dropna(inplace=True)

            # Adjust y to include all the lottery numbers you want to predict
            y = pd.DataFrame(data_expanded['Winning Numbers'].tolist())

            # Adjust X to include the features you want to use to predict the lottery numbers
            # Exclude 'Draw Date' column
            X = data_expanded.drop(columns=['Winning Numbers', 'Draw Date'])

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                    random_state=self.random_state)
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            return None, None, None, None

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        try:
            # Use MultiOutputRegressor to handle multiple class outputs
            self.model = MultiOutputRegressor(RandomForestRegressor(random_state=self.random_state), n_jobs=-1)
            self.model.fit(self.X_train, self.y_train)
            logging.info(f"Model trained successfully.")
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            return

    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test.values.ravel(), y_pred.ravel())
            mse = mean_squared_error(self.y_test.values.ravel(), y_pred.ravel())
            logging.info(f"Mean Absolute Error: {mae}")
            logging.info(f"Mean Squared Error: {mse}")
        except Exception as e:
            logging.error(f"Error in model evaluation: {e}")
            return

    def save_model(self, filename):
        try:
            joblib.dump(self.model, filename)
            logging.info(f"Model saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return

    def load_and_test_model(self, filename):
        try:
            loaded_model = joblib.load(filename)
            y_pred_loaded = loaded_model.predict(self.X_test)
            mae_loaded = mean_absolute_error(self.y_test.values.ravel(), y_pred_loaded.ravel())
            mse_loaded = mean_squared_error(self.y_test.values.ravel(), y_pred_loaded.ravel())
            logging.info(f"Mean Absolute Error (Loaded Model): {mae_loaded}")
            logging.info(f"Mean Squared Error (Loaded Model): {mse_loaded}")
        except Exception as e:
            logging.error(f"Error loading and testing model: {e}")
            return

        return mae_loaded, mse_loaded


if __name__ == "__main__":
    logging.info("Starting Lottery Predictor")

    predictor = LotteryPredictor('lottery_db.sqlite', "SELECT * FROM csvs")
    predictor.preprocess_data()
    predictor.train_model()
    predictor.evaluate_model()
    predictor.save_model('models/lottery_predictor_model.pkl')
    predictor.load_and_test_model('models/lottery_predictor_model.pkl')

    logging.info("Done")
