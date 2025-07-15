import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def get_health_advice(user_input, prediction):
    prompt = f"""
    Patient details:
    {user_input}

    Prediction: {prediction}

    Based on the above information:
    
    0. Remember, every patient is unique. The following advice is general and should not replace professional medical consultation. Always encourage the user to consult a healthcare professional for personalized advice.
    1. For each point below, keep your response concise, polite, and easy to understand.
    2. Briefly describe the possible severity or seriousness of the predicted condition.
    3. Clearly suggest when the user should see a doctor. If the risk is high, recommend immediate medical attention.
    4. Offer practical lifestyle and dietary tips to help reduce risk or manage the condition.
    5. End with a positive, encouraging message that motivates healthy habits and discourages harmful behaviors such as smoking or excessive drinking.

    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
