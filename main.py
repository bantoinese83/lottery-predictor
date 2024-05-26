import pandas as pd
from fastapi import FastAPI, HTTPException
import random
import logging

from lottery_model import LotteryModel

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = "models/lottery_predictor_model.pkl"
model = LotteryModel(model_path)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Lottery Predictor!",
        "description": "Use this API to predict the next Powerball winning numbers based on historical data."
    }


@app.get("/predict")
def predict():
    try:
        # Generate random multiplier
        multiplier = random.randint(1, 26)

        # Create a DataFrame from the input data
        input_df = pd.DataFrame({'Multiplier': [multiplier]})

        # Convert DataFrame to 2D array
        input_data = input_df.values

        # Make predictions with the model
        predictions = model.predict([input_data])

        # Round predictions to nearest integer
        predictions_rounded = [round(num) for num in predictions[0]]

        return {"predictions": predictions_rounded}
    except Exception as e:
        error_message = "An error occurred while processing the request."
        logging.error(f"{error_message} Error Details: {e}")
        raise HTTPException(status_code=500, detail=error_message)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
