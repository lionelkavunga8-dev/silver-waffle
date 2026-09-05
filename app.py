import os
import time
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate-exam", methods=["POST"])
def generate_exam():
    data = request.json or {}
    topic = data.get("topic", "All Domains (Full Exam)")
    count = data.get("count", 1)

    prompt = f"""Generate exactly {count} realistic Cisco CCNA (200-301) exam questions for topic: "{topic}".
    Return STRICTLY a JSON array of objects with no surrounding markdown text or codeblocks.
    Each object in the array MUST strictly follow this JSON schema:
    [
      {{
        "question": "The question text",
        "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
        "answer": "A",
        "explanation": "Detailed CCNA level explanation"
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
            if "503" in err_msg or "UNAVAILABLE" in err_msg:
                if attempt < max_retries - 1:
                    print(f"[503 Spike] Server busy. Retrying in {delay}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2
                    continue
            
            return jsonify({"error": f"Gemini API Error: {err_msg}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)