import streamlit as st
import pandas as pd
import joblib

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Heatwave RAG Chatbot", layout="centered")

st.title("🌡️ Heatwave Prediction + RAG Chatbot")
st.markdown("Ask weather conditions and get ML prediction + AI explanation")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("random_forest.pkl")

# =========================
# RAG SETUP (STATIC KNOWLEDGE)
# =========================
docs = [
    "Heatwaves occur due to high temperature, low humidity and high pressure.",
    "Low soil moisture increases surface heating.",
    "High solar radiation intensifies heatwaves.",
    "Weak winds and clear skies contribute to heatwaves.",
    "Urban heat islands increase temperature in cities."
]


openai_api_key=os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key,model="text-embedding-ada-002")

vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()

def get_context(query):
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

# =========================
# PREDICTION FUNCTION
# =========================
def predict(input_df):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return pred, prob

# =========================
# UI INPUTS
# =========================
st.subheader("📊 Enter Weather Parameters")

wind_u10 = st.number_input("WIND_U10", value=0.0)
wind_v10 = st.number_input("WIND_V10", value=0.0)
mslp = st.number_input("MSLP", value=1000.0)
blh = st.number_input("BLH", value=100.0)
geop=st.number_input("GEOP", value=500.0)
temp2m = st.number_input("TEMP2M", value=30.0)
dew2m = st.number_input("DEW2M", value=100.0)
cloud = st.number_input("CLOUD", value=0.0)

rain = st.number_input("RAIN", value=0.0)
srad = st.number_input("SRAD", value=200.0)
evap = st.number_input("EVAP", value=0.0)
soilt1=st.number_input("SOILT1", value=0.0)
soilm1= st.number_input("SOILM1", value=0.0)
lai=st.number_input("LAI", value=0.0)

if st.button("🔮 Predict Heatwave"):

    # create input dataframe
    input_df = pd.DataFrame([[
        wind_u10, wind_v10, mslp, blh,geop, dew2m, temp2m, cloud, rain, srad, evap, soilt1, soilm1, lai
    ]], columns=[
        "WIND_U10","WIND_V10","MSLP","BLH","GEOP","DEW2M","TEMP2M","CLOUD","RAIN","SRAD","EVAP","SOILT1","SOILM1","LAI"
    ])

    # ML prediction
    pred, prob = predict(input_df)

    # RAG retrieval
    query = f"temperature {temp2m} pressure {mslp} humidity cloud rain heatwave conditions"
    context = get_context(query)

    # OUTPUT
    st.subheader("🌡️ Prediction Result")

    if pred == 1:
        st.error(f"🔥 Heatwave Expected (Probability: {prob:.2f})")
    else:
        st.success(f"🌤️ No Heatwave (Probability: {prob:.2f})")

    st.subheader("📚 AI Explanation (RAG)")
    st.info(context)

    st.subheader("🧠 Insight")
    st.write(
        "Prediction is based on ML model + retrieved climate knowledge "
        "such as pressure, radiation, and humidity patterns."
    )