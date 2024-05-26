import streamlit as st
from lottery_model import LotteryModel
import random
import time

# Load the trained model
model_path = "models/lottery_predictor_model.pkl"
model = LotteryModel(model_path)

st.title("🎰 Lottery Predictor 🎲")

st.markdown("Click the button to predict the next Powerball 🏀 winning numbers.")

if st.button('Predict 🚀'):
    # Generate random numbers for the model
    multiplier = random.randint(1, 26)

    # Prepare the data in the format expected by the model
    data = [multiplier]

    # Add a progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    try:
        # Make predictions with the model
        predictions = model.predict(data)

        # Round predictions to nearest integer
        predictions_rounded = [round(num) for num in predictions[0]]

        # Display the predictions
        st.success(f"🎉 Predicted winning numbers: {', '.join(map(str, predictions_rounded))} 🎉")
        print(f"Predicted winning numbers: {', '.join(map(str, predictions_rounded))}")
    except Exception as e:
        st.error(f"Error: {str(e)} 😞")

st.markdown("Note: This is a simple model for demonstration purposes only. The predictions are not guaranteed. 🧐")