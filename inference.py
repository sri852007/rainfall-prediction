
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/best_model.pkl")

# Example weather data
sample = pd.DataFrame({
    "pressure": [1025.9],
    "temparature": [18.3],
    "dewpoint": [13.1],
    "humidity": [72],
    "cloud": [49],
    "sunshine": [9.3],
    "winddirection": [80.0],
    "windspeed": [26.3]
})

# Make rainfall prediction
prediction = model.predict(sample)

print(
    "Rainfall Prediction:",
    "Rainfall" if prediction[0] == 1 else "No Rainfall"
)
