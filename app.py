import gradio as gr
import joblib
import pandas as pd

# 1. Load the trained model globally (loads once when the server starts)
try:
    model = joblib.load('RidgeModel_new.joblib')
except FileNotFoundError:
    raise FileNotFoundError("Model file 'RidgeModel_new.joblib' not found. Ensure it is uploaded.")

# 2. Wrap your exact Streamlit prediction button logic inside a clean function
def predict_price(location, total_sqft, bath, bhk):
    # This matches exactly how your Streamlit code formats the dataframe for your pipeline
    input_data = pd.DataFrame([{
        'location': location,
        'total_sqft': float(total_sqft),
        'bath': float(bath),
        'bhk': int(bhk)
    }])
    
    # Predict using your model pipeline
    prediction = model.predict(input_data)[0]
    
    # Return the clean text response to show in the UI output box
    return f"🏠 Predicted Price: ₹{round(prediction, 2)} Lakhs"

# 3. Create the Gradio interface mapping your precise inputs to the function
interface = gr.Interface(
    fn=predict_price, # The function Gradio will call when a user interacts
    inputs=[
        # Dropdown with your exact list of locations
        gr.Dropdown(
            choices=['other', 'Whitefield', 'Sarjapur  Road', 'Electronic City', 'Kanakpura Road', 'Thanisandra', 'Yelahanka', 'Uttarahalli', 'Hebbal', 'Raja Rajeshwari Nagar', 'Marathahalli', 'Hennur Road', 'Banashankari', 'Haralur Road', 'Electronic City Phase II', 'Hosur Road', 'Old Airport Road', 'Kothanur', 'Rustam Bagh', 'Kadugodi', '7th Phase JP Nagar', 'Kaggadasapura', 'Chandapura', 'Kengeri', 'Domlur', 'Yeshwanthpur', 'Sarjapur', 'Kasavanhalli', 'Begur Road', 'Ramamurthy Nagar', 'Malleshwaram', 'Varthur', 'Bommasandra', 'BTM 2nd Stage', 'Tin Factory', 'Bellandur'], 
            label="Location", 
            value="other"
        ),
        # Number inputs and sliders matching your exact parameters
        gr.Number(label="Total Square Feet", value=1200.0),
        gr.Slider(minimum=1, maximum=16, step=1, label="Number of Bathrooms", value=2),
        gr.Slider(minimum=1, maximum=16, step=1, label="Number of BHK", value=2)
    ],
    outputs=gr.Textbox(label="Prediction Output"),
    title="Bengaluru House Price Prediction",
    description="Enter house details below to compute property valuations using our pre-trained Ridge regression machine learning model."
)

# 4. Launch the application
if __name__ == "__main__":
    interface.launch()