import os
import time
import sqlite3
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai

# 1. Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file!")

app = Flask(__name__)
client = genai.Client(api_key=api_key)

DB_NAME = "exam_progress.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            status TEXT,
            missed_questions TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate-exam", methods=["POST"])
def generate_exam():
    data = request.json or {}
    topic = data.get("topic", "All Domains (Full Exam)")
    count = int(data.get("count", 1))

    prompt = f"""Generate exactly {count} realistic Cisco CCNA (200-301) exam questions for topic: "{topic}".
    Return STRICTLY a JSON array matching this exact format with no markdown blocks:
    [
      {{
        "question": "Question text",
        "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
        "answer": "A",
        "explanation": "Clear CCNA technical explanation"
      }}
    ]"""

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return response.text, 200, {'Content-Type': 'application/json'}
            
        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg or "rate limit" in err_msg.lower():
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            return jsonify({"error": f"Gemini API Error: {err_msg}"}), 500

@app.route("/save-session", methods=["POST"])
def save_session():
    data = request.json or {}
    topic = data.get("topic", "Unknown")
    score = data.get("score", 0)
    total = data.get("total", 0)
    percentage = round((score / total) * 100, 2) if total > 0 else 0
    status = "PASSED" if percentage >= 85.0 else "FAILED"
    missed_questions = json.dumps(data.get("missed_questions", []))

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO exam_sessions (topic, score, total, percentage, status, missed_questions) VALUES (?, ?, ?, ?, ?, ?)",
            (topic, score, total, percentage, status, missed_questions)
        )
        conn.commit()
        conn.close()
        return jsonify({"status": status, "percentage": percentage}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ai-progress-agent", methods=["GET"])
def ai_progress_agent():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, topic, score, total, percentage, status, missed_questions FROM exam_sessions ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"report": "No exam history found yet. Take a practice exam to initialize your AI progress agent!"})

        history_summary = []
        for r in rows:
            history_summary.append({
                "timestamp": r[0],
                "topic": r[1],
                "score": r[2],
                "total": r[3],
                "percentage": r[4],
                "status": r[5],
                "missed_concepts": json.loads(r[6])
            })

        agent_prompt = f"""You are an expert CCNA (200-301) AI Mentor Agent. Analyze the student's exam history below and provide a structured progress report.
        
        Evaluate performance across the 6 official CCNA domains:
        1. Network Fundamentals
        2. Network Access
        3. IP Connectivity
        4. IP Services
        5. Security Fundamentals
        6. Automation & Programmability

        Student Exam History:
        {json.dumps(history_summary, indent=2)}

        Provide:
        1. Overall readiness assessment against the 85% passing standard.
        2. Domain-by-domain breakdown highlighting strengths and weaknesses.
        3. Specific actionable study advice for the weakest domain found.
        
        Keep formatting clean with clear headings.
        """

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=agent_prompt
        )
        return jsonify({"report": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
