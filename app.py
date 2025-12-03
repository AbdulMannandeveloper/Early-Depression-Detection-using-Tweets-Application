import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

# ✅ Set Streamlit page config at the VERY TOP (before any UI commands)
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# Use your Hugging Face model repository path
MODEL_PATH = "AMGzz/mentalbert-depression-detector"  # Replace with your actual Hugging Face repo name

@st.cache_resource
def load_model():
    # Load the tokenizer and model directly from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

tokenizer, model = load_model()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit UI
st.title("🧠 Mental Health Text Classifier")
st.markdown("Enter a sentence or paragraph, and the model will classify it as **Normal** or **Depression**.")

user_input = st.text_area("📝 Your input text:")

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize the input and predict the label
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        # Map model output to label
        label_map = {0: "Normal", 1: "Depression"}
        result = label_map[prediction]

        st.success(f"🧾 **Prediction**: {result}")
