import streamlit as st
import google.generativeai as genai
from langdetect import detect
from textblob import TextBlob
from fpdf import FPDF
from io import BytesIO
import concurrent.futures
import json

# Configure API Key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App Configuration
st.set_page_config(page_title="Client Email Analyzer", page_icon="📧", layout="wide")
st.title("📨 Client Email Analyzer & Insights")
st.write("Analyze incoming client emails with AI-powered insights.")

# Default Enabled Features
features = {
    "sentiment": True,
    "highlights": True,
    "response_suggestion": True,
    "export": True,
    "tone": True,
    "urgency": True,
    "task_extraction": True,
    "category": True,
    "emotion": True,
    "spam_check": True,
    "readability": True,
    "root_cause": True,  # Identifies reasons behind sentiment and tone
    "clarity": True,  # Rates clarity of the email
    "best_response_time": True,  # Suggests the best time to respond
}

# Email Input Section
email_content = st.text_area("📩 Paste the client's email content here:", height=200)
MAX_EMAIL_LENGTH = 2000  # Increased for better analysis

# Cache AI Responses for Performance
@st.cache_data(ttl=3600)
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return ""

# Additional Analysis Functions
def get_sentiment(email_content):
    return TextBlob(email_content).sentiment.polarity

def get_readability(email_content):
    return round(TextBlob(email_content).sentiment.subjectivity * 10, 2)  # Rough readability proxy

def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# Process Email When Button Clicked
if email_content and st.button("🔍 Analyze Client Email"):
    try:
        detected_lang = detect(email_content)
        if detected_lang != "en":
            st.error("⚠️ Only English language is supported.")
        else:
            with st.spinner("⚡ Processing client email insights..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # AI-Powered Analysis
                    future_summary = executor.submit(get_ai_response, "Summarize this client email concisely:\n\n", email_content)
                    future_highlights = executor.submit(get_ai_response, "Extract key points from this email:\n\n", email_content)
                    future_tone = executor.submit(get_ai_response, "Analyze the tone of this client email:\n\n", email_content)
                    future_urgency = executor.submit(get_ai_response, "Determine the urgency level of this client email:\n\n", email_content)
                    future_tasks = executor.submit(get_ai_response, "Identify any requests or tasks mentioned in this email:\n\n", email_content)
                    future_category = executor.submit(get_ai_response, "Categorize this client email:\n\n", email_content)
                    future_emotion = executor.submit(get_ai_response, "Analyze emotions conveyed in this email:\n\n", email_content)
                    future_spam = executor.submit(get_ai_response, "Determine if this client email is spam or genuine:\n\n", email_content)
                    future_root_cause = executor.submit(get_ai_response, "Analyze the root cause behind the sentiment and tone:\n\n", email_content)
                    future_clarity = executor.submit(get_ai_response, "Rate the clarity of this client email:\n\n", email_content)
                    future_best_time = executor.submit(get_ai_response, "Suggest the best time to respond to this client email:\n\n", email_content)

                    # Extract Results
                    summary = future_summary.result()
                    highlights = future_highlights.result()
                    tone = future_tone.result()
                    urgency = future_urgency.result()
                    tasks = future_tasks.result()
                    category = future_category.result()
                    emotion = future_emotion.result()
                    spam_status = future_spam.result()
                    root_cause = future_root_cause.result()
                    clarity_score = future_clarity.result()
                    best_response_time = future_best_time.result()
                    readability_score = get_readability(email_content)

                # Display Results
                st.subheader("📌 Client Email Summary")
                st.write(summary)

                st.subheader("🔑 Key Highlights")
                st.write(highlights)

                st.subheader("🎭 Tone Analysis")
                st.write(tone)

                st.subheader("⚠️ Urgency Level")
                st.write(urgency)

                st.subheader("📝 Identified Requests or Tasks")
                st.write(tasks)

                st.subheader("📂 Email Category")
                st.write(category)

                st.subheader("📖 Readability Score")
                st.write(f"{readability_score} / 10")

                st.subheader("🧐 Root Cause Analysis")
                st.write(root_cause)

                st.subheader("🔍 Email Clarity Score")
                st.write(clarity_score)

                st.subheader("🕒 Best Time to Respond")
                st.write(best_response_time)

                # Export Options
                export_data = json.dumps({
                    "summary": summary, "highlights": highlights,
                    "root_cause": root_cause, "clarity_score": clarity_score,
                    "best_response_time": best_response_time
                }, indent=4)
                st.download_button("📥 Download JSON", data=export_data, file_name="client_email_analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("✏️ Paste the client's email content and click 'Analyze Client Email' to begin.")
