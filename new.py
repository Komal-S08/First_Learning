import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import base64
from datetime import datetime
from streamlit_chat import message
from fpdf import FPDF
import speech_recognition as sr
import google.generativeai as genai
import resend

# --------------------🔐 Configuration --------------------

model = pickle.load(open("model.pkl", "rb"))
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", "your-gemini-key"))
resend.api_key = st.secrets.get("RESEND_API_KEY", "your-resend-key")

# --------------------📁 Data for Manual Form --------------------
df = pd.read_csv(r"C:\Users\Komal\Documents\archive (4).csv")
df.dropna(inplace=True)

# --------------------⚙️ Streamlit Setup --------------------
st.set_page_config(page_title="Smart Course Assistant", layout="wide")
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Go to", ["💬 Smart Assistant", "🔮 Predict Completion"])

# --------------------🧠 NLP Feature Extraction --------------------
def extract_features(text):
    text = text.lower()
    data = {
        "course_category": 0,
        "device_type": 1,
        "time_spent": 0.0,
        "videos_watched": 0,
        "quizzes_taken": 0,
        "quiz_scores": 0.0,
        "completion_rate": 0.0,
        "forum_participation": 0,
        "peer_interaction": 0,
        "feedback_given": 0,
        "reminders_clicked": 0,
        "support_usage": 0,
    }

    patterns = {
        "time_spent": r"spent (\d+\.?\d*) hours",
        "videos_watched": r"watched (\d+)",
        "quizzes_taken": r"took (\d+)",
        "quiz_scores": r"scored (\d+\.?\d*)%",
        "completion_rate": r"completed (\d+\.?\d*)%",
        "forum_participation": r"forum.*?(\d+)",
        "peer_interaction": r"peer.*?(\d+)",
        "feedback_given": r"feedback.*?(\d+)",
        "reminders_clicked": r"reminder.*?(\d+)",
        "support_usage": r"support.*?(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            data[key] = val if key in ["time_spent", "quiz_scores", "completion_rate"] else int(val)

    return np.array([[data[k] for k in data]]), data

# --------------------🎙️ Voice --------------------
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... speak now")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, I didn't catch that."

# --------------------🔮 Gemini Explain --------------------
def ask_gemini(features, prediction, prob):
    prompt = f"""Student metrics:
{features}

Prediction: {'Completed' if prediction == 1 else 'Not Completed'}
Confidence Score: {prob:.2f}

Explain the prediction simply."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

# --------------------📤 Export + Email --------------------
def get_chat_df():
    users = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    bots = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]
    return pd.DataFrame({"User": users, "Assistant": bots[:len(users)]})

def export_predictions_to_pdf(predictions, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Course Completion Prediction Report", ln=True, align='C')
    pdf.ln(10)
    for i, pred in enumerate(predictions):
        pdf.multi_cell(0, 10, txt=f"Prediction #{i+1}: {'Completed' if pred == 1 else 'Not Completed'}")
        pdf.ln(5)
    pdf.output(filename)

def send_email_with_attachment(receiver, subject, body, attachment_path):
    with open(attachment_path, "rb") as f:
        encoded_file = base64.b64encode(f.read()).decode()
    resend.Emails.send({
        "from": "Assistant <onboarding@resend.dev>",
        "to": receiver,
        "subject": subject,
        "html": f"<p>{body}</p>",
        "attachments": [{
            "filename": attachment_path,
            "content": encoded_file,
            "contentType": "application/octet-stream"
        }]
    })

# --------------------💬 Smart Assistant Page --------------------
if page == "💬 Smart Assistant":
    st.title("🎓 Smart Course Completion Assistant")
    st.markdown("Ask in natural language or speak. I’ll predict and explain course completion.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "predictions" not in st.session_state:
        st.session_state.predictions = []

    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=str(i))

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("You:", key="input")
    with col2:
        if st.button("🎤 Speak"):
            user_input = record_audio()
            st.session_state.input = user_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        features, raw_data = extract_features(user_input)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        st.session_state.predictions.append(pred)
        response = ask_gemini(raw_data, pred, prob)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response, is_user=False, key=str(len(st.session_state.messages)))

        # Visuals
        engagement_score = np.sum(features[0][2:])
        max_engagement = 100 * len(features[0][2:])
        engagement_percent = (engagement_score / max_engagement) * 100
        st.progress(int(engagement_percent), text=f"Engagement Score: {engagement_percent:.1f}%")
        st.metric("Prediction Probability", f"{prob*100:.2f}%")

        fig, ax = plt.subplots()
        ax.barh(["Engagement"], [engagement_percent], color='skyblue')
        ax.set_xlim(0, 100)
        ax.set_title("Engagement vs. Prediction")
        st.pyplot(fig)

    # Sidebar tools
    st.sidebar.markdown("## 📤 Export Tools")
    if st.sidebar.button("💾 Save Chat as CSV"):
        df = get_chat_df()
        fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(fname, index=False)
        st.sidebar.success(f"Saved: {fname}")

    if st.sidebar.button("📄 Export Predictions to PDF"):
        export_predictions_to_pdf(st.session_state.predictions)
        st.sidebar.success("PDF Exported")

    email = st.sidebar.text_input("📧 Enter email to send report")
    if st.sidebar.button("📨 Send Email") and email:
        export_predictions_to_pdf(st.session_state.predictions)
        send_email_with_attachment(email, "Course Completion Report", "See attached report.", "report.pdf")
        st.sidebar.success(f"Email sent to {email}")

    st.sidebar.markdown("## 📁 Bulk Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV for Bulk Prediction", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        predictions = model.predict(data)
        st.write("### 🔢 Bulk Prediction Results")
        data['Prediction'] = predictions
        st.dataframe(data)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "bulk_predictions.csv", "text/csv")

# --------------------🔮 Predict Completion Page --------------------
elif page == "🔮 Predict Completion":
    st.title("🎓 Online Learning Effectiveness Predictor")
    st.markdown("Predict if a student will complete an online course based on their engagement data.")

    course_category = st.selectbox("📘 Course Category", ["Data Science", "Arts", "Commerce", "Technology"])
    category_dict = {"Data Science": 0, "Arts": 1, "Commerce": 2, "Technology": 3}
    course_category_encoded = category_dict[course_category]

    device_type = st.selectbox("💻 Device Type", ["Mobile Phone", "Laptop"])
    device_dict = {"Mobile Phone": 0, "Laptop": 1}
    device_encoded = device_dict[device_type]

    time_spent = st.slider("⏱️ Time Spent on Course (hours)", 0.0, 100.0, 20.0)
    videos_watched = st.number_input("🎥 Videos Watched", 0, 100, 10)
    quizzes_taken = st.number_input("📝 Quizzes Taken", 0, 50, 5)
    quiz_scores = st.slider("📊 Quiz Score (%)", 0.0, 100.0, 75.0)
    completion_rate = st.slider("📈 Completion Rate (%)", 0.0, 100.0, 60.0)

    with st.expander("➕ Additional Features"):
        forum_participation = st.number_input("💬 Forum Participation", 0, 100, 5)
        peer_interaction = st.slider("🤝 Peer Interaction (0-100)", 0, 100, 50)
        feedback_given = st.number_input("🗒️ Feedback Given", 0, 100, 3)
        reminders_clicked = st.number_input("🔔 Reminders Clicked", 0, 100, 2)
        support_usage = st.number_input("🆘 Support Usage", 0, 100, 1)

    if st.button("🔮 Predict"):
        features = np.array([[course_category_encoded, device_encoded, time_spent, videos_watched,
                              quizzes_taken, quiz_scores, completion_rate, forum_participation,
                              peer_interaction, feedback_given, reminders_clicked, support_usage]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.success(f"Prediction: {'✅ Will Complete' if prediction == 1 else '❌ Not Likely to Complete'}")
        st.metric("Confidence", f"{probability * 100:.2f}%")
