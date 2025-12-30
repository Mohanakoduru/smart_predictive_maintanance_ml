import streamlit as st
import joblib
import numpy as np
import os
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit




st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-image: url("https://tse4.mm.bing.net/th/id/OIP.0WLrVPhcXOyHT3BwGt3gLAHaEO?rs=1&pid=ImgDetMain&o=7&rm=3");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Transparent content container */
    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 12px;
    }

    /* Text readability */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #F9FAFB !important;
    }

    /* Input fields */
    input, textarea, select {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #E5E7EB !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# -------------------------------------------------
# Load ML assets
# -------------------------------------------------
model = joblib.load("maintenance_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------------------------------------
# PDF GENERATOR FUNCTION
# -------------------------------------------------
def generate_pdf_report(
    filename,
    inputs,
    status,
    confidence,
    explanation,
    recommendation
):
    c = canvas.Canvas(filename, pagesize=A4)

    width, height = A4

    # Margins (THIS FIXES THE ISSUE)
    left_margin = 40
    right_margin = 40
    top_margin = 40
    bottom_margin = 40

    usable_width = width - left_margin - right_margin
    y = height - top_margin

    def draw_heading(text):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, text)
        y -= 22

    def draw_text(text, font_size=10, gap=14):
        nonlocal y
        c.setFont("Helvetica", font_size)

        wrapped_lines = simpleSplit(
            text,
            "Helvetica",
            font_size,
            usable_width
        )

        for line in wrapped_lines:
            if y < bottom_margin:
                c.showPage()
                c.setFont("Helvetica", font_size)
                y = height - top_margin
            c.drawString(left_margin, y, line)
            y -= gap

    # ---------------- HEADER ----------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, y, "Predictive Maintenance Report")
    y -= 30

    c.setFont("Helvetica", 10)
    draw_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    draw_text("-" * 90, gap=18)

    # ---------------- MATERIAL DETAILS ----------------
    draw_heading("1. Material Identification")
    for k, v in inputs.items():
        draw_text(f"{k}: {v}")

    draw_text("-" * 90, gap=18)

    # ---------------- PREDICTION ----------------
    draw_heading("2. Machine Learning Prediction Summary")
    draw_text(f"Predicted Condition: {status}")
    draw_text(f"Risk Confidence: {confidence:.2f}%")

    risk_level = "HIGH" if confidence >= 75 else "MEDIUM" if confidence >= 40 else "LOW"
    draw_text(f"Risk Level: {risk_level}")

    draw_text("-" * 90, gap=18)

    # ---------------- AI EXPLANATION ----------------
    draw_heading("3. AI-Based Explanation")
    for paragraph in explanation.split("\n"):
        draw_text(paragraph)

    draw_text("-" * 90, gap=18)

    # ---------------- RECOMMENDATION ----------------
    draw_heading("4. Maintenance Recommendation")
    draw_text(recommendation)

    draw_text("-" * 90, gap=18)

    # ---------------- DISCLAIMER ----------------
    draw_heading("Disclaimer")
    draw_text(
        "This report is generated using machine learning and AI-based analysis "
        "from user-provided data. It is intended to support maintenance decisions "
        "and should be reviewed by a qualified engineer before implementation."
    )

    c.save()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Smart Predictive Maintenance System")
st.title("🏗 Smart Predictive Maintenance System")


st.markdown("---")

material_type = st.selectbox("Material Type", ["Steel", "Cement", "Brick"])
material_age_days = st.slider("Material Age (Days)", 1, 500, 60)
usage_frequency = st.selectbox("Usage Frequency", ["Low", "Medium", "High"])
humidity_exposure = st.selectbox("Humidity Exposure", ["Low", "Medium", "High"])
load_stress_level = st.selectbox("Load Stress Level", ["Low", "Medium", "High"])
cracks_visible = st.selectbox("Cracks Visible?", ["No", "Yes"])
last_maintenance_days = st.slider("Days Since Last Maintenance", 0, 300, 45)

if st.button("Predict & Generate Report"):
    input_data = np.array([[
        encoders["material_type"].transform([material_type])[0],
        material_age_days,
        encoders["usage_frequency"].transform([usage_frequency])[0],
        encoders["humidity_exposure"].transform([humidity_exposure])[0],
        encoders["load_stress_level"].transform([load_stress_level])[0],
        encoders["cracks_visible"].transform([cracks_visible])[0],
        last_maintenance_days
    ]])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    status = target_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities) * 100)

    recommendation = (
        "No maintenance required."
        if status == "Good"
        else "Schedule maintenance soon."
        if status == "Moderate"
        else "Immediate replacement required."
    )

    # Groq explanation
    explanation = "AI explanation unavailable."
    if groq_client:
        try:
            prompt = f"""
Material: {material_type}
Age: {material_age_days} days
Usage: {usage_frequency}
Humidity: {humidity_exposure}
Load stress: {load_stress_level}
Cracks visible: {cracks_visible}
Last maintenance: {last_maintenance_days} days

Prediction: {status}
Risk confidence: {confidence:.2f}%

Explain the condition, causes, and risks briefly.
"""
            res = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a construction maintenance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=200
            )
            explanation = res.choices[0].message.content
        except:
            pass

    inputs = {
        "Material Type": material_type,
        "Material Age (days)": material_age_days,
        "Usage Frequency": usage_frequency,
        "Humidity Exposure": humidity_exposure,
        "Load Stress Level": load_stress_level,
        "Cracks Visible": cracks_visible,
        "Days Since Last Maintenance": last_maintenance_days
    }

    pdf_file = "maintenance_report.pdf"
    generate_pdf_report(
        pdf_file,
        inputs,
        status,
        confidence,
        explanation,
        recommendation
    )

    st.success(f"Prediction: {status} ({confidence:.2f}%)")
    st.write(explanation)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Maintenance Report (PDF)",
            data=f,
            file_name="Maintenance_Report.pdf",
            mime="application/pdf"
        )

# ---------------------Footer ---------------------
# FOOTER
# ======================
st.markdown(
    """
    <div class="glass" style="text-align:center;">
        <p>Developed by @MOHAN KODURU | @MOKSHAGNA | @SAFIYA </p>
    </div>
    """,
    unsafe_allow_html=True
)
