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
st.set_page_config(page_title="Advanced Email AI", page_icon="📧", layout="wide")
st.title("📨 Advanced Email AI Analysis & Insights")
st.write("Extract insights, generate professional responses, and analyze emails with AI.")

# Default Enabled Features
features = {
    "sentiment": True,
    "highlights": True,
    "response": True,
    "export": True,
    "tone": True,
    "urgency": True,
    "task_extraction": True,
    "subject_recommendation": True,
    "category": True,
    "politeness": True,
    "emotion": True,
    "spam_check": True,
    "readability": True,
    "root_cause": True,  # NEW: Identifies the reason behind tone/sentiment.
    "grammar_check": True,  # NEW: Checks spelling & grammar.
    "clarity": True,  # NEW: Rates clarity of the email.
    "best_response_time": True,  # NEW: Suggests the best time to respond.
    "professionalism": True,  # NEW: Rates professionalism level.
}

# Email Input Section
email_content = st.text_area("📩 Paste your email content here:", height=200)
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
if email_content and st.button("🔍 Generate Insights"):
    try:
        detected_lang = detect(email_content)
        if detected_lang != "en":
            st.error("⚠️ Only English language is supported.")
        else:
            with st.spinner("⚡ Processing email insights..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # AI-Powered Analysis (executing based on feature flags)
                    future_summary = executor.submit(get_ai_response, "Summarize this email concisely:\n\n", email_content) if features["highlights"] else None
                    future_response = executor.submit(get_ai_response, "Generate a professional response to this email:\n\n", email_content) if features["response"] else None
                    future_highlights = executor.submit(get_ai_response, "Highlight key points:\n\n", email_content) if features["highlights"] else None
                    future_tone = executor.submit(get_ai_response, "Detect the tone of this email:\n\n", email_content) if features["tone"] else None
                    future_urgency = executor.submit(get_ai_response, "Analyze urgency level:\n\n", email_content) if features["urgency"] else None
                    future_tasks = executor.submit(get_ai_response, "List actionable tasks:\n\n", email_content) if features["task_extraction"] else None
                    future_subject = executor.submit(get_ai_response, "Suggest a professional subject line:\n\n", email_content) if features["subject_recommendation"] else None
                    future_category = executor.submit(get_ai_response, "Categorize this email:\n\n", email_content) if features["category"] else None
                    future_politeness = executor.submit(get_ai_response, "Evaluate politeness score:\n\n", email_content) if features["politeness"] else None
                    future_emotion = executor.submit(get_ai_response, "Analyze emotions in this email:\n\n", email_content) if features["emotion"] else None
                    future_spam = executor.submit(get_ai_response, "Detect if this email is spam/scam:\n\n", email_content) if features["spam_check"] else None
                    future_root_cause = executor.submit(get_ai_response, "Analyze the root cause of the email tone and sentiment:\n\n", email_content) if features["root_cause"] else None
                    future_grammar = executor.submit(get_ai_response, "Check spelling & grammar mistakes and suggest fixes:\n\n", email_content) if features["grammar_check"] else None
                    future_clarity = executor.submit(get_ai_response, "Rate the clarity of this email:\n\n", email_content) if features["clarity"] else None
                    future_best_time = executor.submit(get_ai_response, "Suggest the best time to respond to this email:\n\n", email_content) if features["best_response_time"] else None
                    future_professionalism = executor.submit(get_ai_response, "Rate the professionalism of this email on a scale of 1-10:\n\n", email_content) if features["professionalism"] else None

                    # Extract Results
                    summary = future_summary.result() if future_summary else None
                    response = future_response.result() if future_response else None
                    highlights = future_highlights.result() if future_highlights else None
                    tone = future_tone.result() if future_tone else None
                    urgency = future_urgency.result() if future_urgency else None
                    tasks = future_tasks.result() if future_tasks else None
                    subject_recommendation = future_subject.result() if future_subject else None
                    category = future_category.result() if future_category else None
                    politeness = future_politeness.result() if future_politeness else None
                    emotion = future_emotion.result() if future_emotion else None
                    spam_status = future_spam.result() if future_spam else None
                    root_cause = future_root_cause.result() if future_root_cause else None
                    grammar_issues = future_grammar.result() if future_grammar else None
                    clarity_score = future_clarity.result() if future_clarity else None
                    best_response_time = future_best_time.result() if future_best_time else None
                    professionalism_score = future_professionalism.result() if future_professionalism else None
                    readability_score = get_readability(email_content)

                # Display Results Based on Enabled Features
                if summary:
                    st.subheader("📌 Email Summary")
                    st.write(summary)

                if response:
                    st.subheader("✉️ Suggested Response")
                    st.write(response)

                if highlights:
                    st.subheader("🔑 Key Highlights")
                    st.write(highlights)

                if features["sentiment"]:
                    st.subheader("💬 Sentiment Analysis")
                    sentiment = get_sentiment(email_content)
                    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                    st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

                if tone:
                    st.subheader("🎭 Email Tone")
                    st.write(tone)

                if urgency:
                    st.subheader("⚠️ Urgency Level")
                    st.write(urgency)

                if tasks:
                    st.subheader("📝 Actionable Tasks")
                    st.write(tasks)

                if category:
                    st.subheader("📂 Email Category")
                    st.write(category)

                if features["readability"]:
                    st.subheader("📖 Readability Score")
                    st.write(f"{readability_score} / 10")

                if root_cause:
                    st.subheader("🧐 Root Cause Analysis")
                    st.write(root_cause)

                if grammar_issues:
                    st.subheader("🔎 Grammar & Spelling Check")
                    st.write(grammar_issues)

                if clarity_score:
                    st.subheader("🔍 Email Clarity Score")
                    st.write(clarity_score)

                if best_response_time:
                    st.subheader("🕒 Best Time to Respond")
                    st.write(best_response_time)

                if professionalism_score:
                    st.subheader("🏆 Professionalism Score")
                    st.write(f"Rated: {professionalism_score} / 10")

                # Export Options
                if features["export"]:
                    export_data = json.dumps({
                        "summary": summary, "response": response, "highlights": highlights,
                        "root_cause": root_cause, "grammar_issues": grammar_issues,
                        "clarity_score": clarity_score, "best_response_time": best_response_time,
                        "professionalism_score": professionalism_score
                    }, indent=4)
                    st.download_button("📥 Download JSON", data=export_data, file_name="analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("✏️ Paste email content and click 'Generate Insights' to begin.")
