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
    "root_cause": True,
    "grammar_check": True,
    "length": True,  # NEW: Length of the email
    "complexity": True,  # NEW: Complexity of the language
    "keyword_density": True,  # NEW: Keyword density analysis
    "entity_recognition": True,  # NEW: Named Entity Recognition
    "language_style": True,  # NEW: Language style analysis
    "positivity": True,  # NEW: Positivity level
    "negativity": True,  # NEW: Negativity level
    "neutrality": True,  # NEW: Neutrality level
    "word_count": True,  # NEW: Total word count
    "sentence_count": True,  # NEW: Total sentence count
    "paragraph_count": True,  # NEW: Total paragraph count
    "average_sentence_length": True,  # NEW: Average sentence length
    "average_word_length": True,  # NEW: Average word length
    "misspellings": True,  # NEW: Detect misspellings
    "sentiment_intensity": True,  # NEW: Sentiment intensity
    "mood": True,  # NEW: Mood analysis
    "purpose": True,  # NEW: Purpose of the email
    "importance": True,  # NEW: Importance level
    "confidence": True,  # NEW: Confidence level
    "honesty": True,  # NEW: Honesty level
    "persuasiveness": True,  # NEW: Persuasiveness level
    "engagement": True,  # NEW: Engagement level
    "relevance": True,  # NEW: Relevance level
    "coherence": True,  # NEW: Coherence level
    "precision": True,  # NEW: Precision level
    "vagueness": True,  # NEW: Vagueness level
    "passive_voice": True,  # NEW: Detect passive voice
    "active_voice": True,  # NEW: Detect active voice
    "attachment_analysis": True,  # NEW: Analyze attachments
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

def get_length(email_content):
    return len(email_content)

def get_complexity(email_content):
    return round(TextBlob(email_content).sentiment.subjectivity * 100, 2)

def get_keyword_density(email_content, keyword):
    words = email_content.split()
    keyword_count = words.count(keyword)
    return round((keyword_count / len(words)) * 100, 2)

def get_entity_recognition(email_content):
    # Placeholder for entity recognition logic
    return "Entities: [PLACEHOLDER]"

def get_language_style(email_content):
    # Placeholder for language style analysis logic
    return "Language Style: [PLACEHOLDER]"

def get_summary(email_content):
    return TextBlob(email_content).summarize()

def get_word_count(email_content):
    return len(email_content.split())

def get_sentence_count(email_content):
    return len(TextBlob(email_content).sentences)

def get_paragraph_count(email_content):
    return len(email_content.split('\n\n'))

def get_average_sentence_length(email_content):
    sentences = TextBlob(email_content).sentences
    word_count = sum([len(sentence.words) for sentence in sentences])
    return round(word_count / len(sentences), 2)

def get_average_word_length(email_content):
    words = email_content.split()
    return round(sum(len(word) for word in words) / len(words), 2)

def get_misspellings(email_content):
    # Placeholder for misspellings detection logic
    return "Misspellings: [PLACEHOLDER]"

def get_sentiment_intensity(email_content):
    # Placeholder for sentiment intensity logic
    return "Sentiment Intensity: [PLACEHOLDER]"

def get_mood(email_content):
    # Placeholder for mood analysis logic
    return "Mood: [PLACEHOLDER]"

def get_purpose(email_content):
    # Placeholder for purpose detection logic
    return "Purpose: [PLACEHOLDER]"

def get_importance(email_content):
    # Placeholder for importance detection logic
    return "Importance: [PLACEHOLDER]"

def get_confidence(email_content):
    # Placeholder for confidence detection logic
    return "Confidence: [PLACEHOLDER]"

def get_honesty(email_content):
    # Placeholder for honesty detection logic
    return "Honesty: [PLACEHOLDER]"

def get_persuasiveness(email_content):
    # Placeholder for persuasiveness detection logic
    return "Persuasiveness: [PLACEHOLDER]"

def get_engagement(email_content):
    # Placeholder for engagement detection logic
    return "Engagement: [PLACEHOLDER]"

def get_relevance(email_content):
    # Placeholder for relevance detection logic
    return "Relevance: [PLACEHOLDER]"

def get_coherence(email_content):
    # Placeholder for coherence detection logic
    return "Coherence: [PLACEHOLDER]"

def get_precision(email_content):
    # Placeholder for precision detection logic
    return "Precision: [PLACEHOLDER]"

def get_vagueness(email_content):
    # Placeholder for vagueness detection logic
    return "Vagueness: [PLACEHOLDER]"

def get_passive_voice(email_content):
    # Placeholder for passive voice detection logic
    return "Passive Voice: [PLACEHOLDER]"

def get_active_voice(email_content):
    # Placeholder for active voice detection logic
    return "Active Voice: [PLACEHOLDER]"

def analyze_attachments(attachments):
    # Placeholder for attachment analysis logic
    return "Attachments: [PLACEHOLDER]"

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
                    future_best_time = executor.submit(get_ai_response, "Suggest the best time to respond to this email:\n\n", email_content) if features["best_response_time"] else None
                    future_attachment_analysis = executor.submit(analyze_attachments, []) if features["attachment_analysis"] else None

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
                    best_response_time = future_best_time.result() if future_best_time else None
                    attachment_analysis = future_attachment_analysis.result() if future_attachment_analysis else None
                    readability_score = get_readability(email_content)
                    length = get_length(email_content)
                    complexity = get_complexity(email_content)
                    keyword_density = get_keyword_density(email_content, "your_keyword")
                    entity_recognition = get_entity_recognition(email_content)
                    language_style = get_language_style(email_content)
                    word_count = get_word_count(email_content)
                    sentence_count = get_sentence_count(email_content)
                    paragraph_count = get_paragraph_count(email_content)
                    average_sentence_length = get_average_sentence_length(email_content)
                    average_word_length = get_average_word_length(email_content)
                    misspellings = get_misspellings(email_content)
                    sentiment_intensity = get_sentiment_intensity(email_content)
                    mood = get_mood(email_content)
                    purpose = get_purpose(email_content)
                    importance = get_importance(email_content)
                    confidence = get_confidence(email_content)
                    honesty = get_honesty(email_content)
                    persuasiveness = get_persuasiveness(email_content)
                    engagement = get_engagement(email_content)
                    relevance = get_relevance(email_content)
                    coherence = get_coherence(email_content)
                    precision = get_precision(email_content)
                    vagueness = get_vagueness(email_content)
                    passive_voice = get_passive_voice(email_content)
                    active_voice = get_active_voice(email_content)

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

                if best_response_time:
                    st.subheader("🕒 Best Time to Respond")
                    st.write(best_response_time)

                if attachment_analysis:
                    st.subheader("📎 Attachment Analysis")
                    st.write(attachment_analysis)

                # New Features Display
                if features["length"]:
                    st.subheader("📏 Email Length")
                    st.write(f"{length} characters")

                if features["complexity"]:
                    st.subheader("🧠 Language Complexity")
                    st.write(f"{complexity} / 100")

                if features["keyword_density"]:
                    st.subheader("🔍 Keyword Density")
                    st.write(f"Keyword Density: {keyword_density}%")

                if features["entity_recognition"]:
                    st.subheader("🏷️ Named Entity Recognition")
                    st.write(entity_recognition)

                if features["language_style"]:
                    st.subheader("🖋️ Language Style")
                    st.write(language_style)

                if features["word_count"]:
                    st.subheader("📝 Word Count")
                    st.write(f"{word_count} words")

                if features["sentence_count"]:
                    st.subheader("🔢 Sentence Count")
                    st.write(f"{sentence_count} sentences")

                if features["paragraph_count"]:
                    st.subheader("🔠 Paragraph Count")
                    st.write(f"{paragraph_count} paragraphs")

                if features["average_sentence_length"]:
                    st.subheader("🔡 Average Sentence Length")
                    st.write(f"{average_sentence_length} words")

                if features["average_word_length"]:
                    st.subheader("🔤 Average Word Length")
                    st.write(f"{average_word_length} characters")

                if features["misspellings"]:
                    st.subheader("📚 Misspellings")
                    st.write(misspellings)

                if features["sentiment_intensity"]:
                    st.subheader("🌡️ Sentiment Intensity")
                    st.write(sentiment_intensity)

                if features["mood"]:
                    st.subheader("🤔 Mood")
                    st.write(mood)

                if features["purpose"]:
                    st.subheader("🎯 Purpose")
                    st.write(purpose)

                if features["importance"]:
                    st.subheader("⭐ Importance")
                    st.write(importance)

                if features["confidence"]:
                    st.subheader("💪 Confidence")
                    st.write(confidence)

                if features["honesty"]:
                    st.subheader("🤥 Honesty")
                    st.write(honesty)

                if features["persuasiveness"]:
                    st.subheader("💬 Persuasiveness")
                    st.write(persuasiveness)

                if features["engagement"]:
                    st.subheader("🤝 Engagement")
                    st.write(engagement)

                if features["relevance"]:
                    st.subheader("🔗 Relevance")
                    st.write(relevance)

                if features["coherence"]:
                    st.subheader("🧩 Coherence")
                    st.write(coherence)

                if features["precision"]:
                    st.subheader("🎯 Precision")
                    st.write(precision)

                if features["vagueness"]:
                    st.subheader("💭 Vagueness")
                    st.write(vagueness)

                if features["passive_voice"]:
                    st.subheader("🗣️ Passive Voice")
                    st.write(passive_voice)

                if features["active_voice"]:
                    st.subheader("🗣️ Active Voice")
                    st.write(active_voice)

                # Export Options
                if features["export"]:
                    export_data = json.dumps({
                        "summary": summary, "response": response, "highlights": highlights,
                        "root_cause": root_cause, "grammar_issues": grammar_issues,
                        "best_response_time": best_response_time, "attachment_analysis": attachment_analysis
                    }, indent=4)
                    st.download_button("📥 Download JSON", data=export_data, file_name="analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("✏️ Paste email content and click 'Generate Insights' to begin.")

