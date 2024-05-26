import joblib
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


class LotteryModel:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.model = None

    def predict(self, data):
        if self.model is None:
            raise Exception("Model is not available.")

        # Create DataFrame from input data with correct column order
        input_df = pd.DataFrame([data],
                                columns=['Multiplier'])

        # Make predictions
        predictions = self.model.predict(input_df)

        # Return a list of predictions for each output
        return predictions.tolist()
