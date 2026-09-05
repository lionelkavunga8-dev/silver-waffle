import os
import sqlite3
import json
import uuid
import secrets
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from google import genai

# 1. Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file!")

app = Flask(__name__)
# Needed for per-visitor sessions (keeps each user's exam history separate).
# Set FLASK_SECRET_KEY in your .env for stable sessions across server restarts.
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

client = genai.Client(api_key=api_key)


@app.errorhandler(Exception)
def handle_any_error(e):
    # Guarantees every response is JSON, even for errors this code doesn't
    # anticipate (proxy timeouts aside) - the frontend always calls
    # res.json() and should never receive an HTML error page.
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": f"Server error: {e}"}), 500


DB_NAME = "exam_progress.db"
MODEL_NAME = "gemini-3.6-flash"
MAX_QUESTIONS_PER_REQUEST = 100

CCNA_DOMAINS = [
    "1.0 Network Fundamentals",
    "2.0 Network Access",
    "3.0 IP Connectivity",
    "4.0 IP Services",
    "5.0 Security Fundamentals",
    "6.0 Automation & Programmability",
]


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            topic TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            status TEXT,
            missed_questions TEXT,
            domain_stats TEXT
        )
    ''')
    # Lightweight migration in case an older DB file already exists on disk
    # (e.g. a previous deploy) without the newer columns.
    cursor.execute("PRAGMA table_info(exam_sessions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, coltype in [("session_id", "TEXT"), ("domain_stats", "TEXT")]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE exam_sessions ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()


init_db()


@app.before_request
def ensure_session_id():
    # Every visitor gets a stable anonymous id (via secure cookie) so exam
    # history / progress reports never mix between different users.
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
        session.permanent = True


@app.route("/")
def home():
    return render_template("index.html")


def _extract_json_array(text):
    """Gemini is asked for raw JSON, but strip markdown fences defensively
    in case it ever wraps the response anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _validate_question(q):
    if not isinstance(q, dict):
        return False
    if not isinstance(q.get("question"), str) or not q["question"].strip():
        return False
    options = q.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return False
    if not all(isinstance(o, str) and o.strip() for o in options):
        return False
    answer = q.get("answer")
    if not isinstance(answer, str) or answer.strip()[:1].upper() not in ("A", "B", "C", "D"):
        return False
    if not isinstance(q.get("explanation"), str) or not q["explanation"].strip():
        return False
    if q.get("domain") not in CCNA_DOMAINS:
        # Don't hard-fail on a missing/odd domain label; fall back gracefully
        # so a single formatting slip doesn't discard an otherwise-good question.
        q["domain"] = q.get("domain") if q.get("domain") in CCNA_DOMAINS else "Unspecified"
    return True


class QuotaExceededError(Exception):
    """Raised when Gemini returns 429/RESOURCE_EXHAUSTED. Never worth
    retrying automatically - free-tier daily caps don't clear in seconds."""
    pass


def _is_quota_error(err_msg):
    lowered = err_msg.lower()
    return "429" in err_msg or "resource_exhausted" in lowered or "quota" in lowered


def _friendly_quota_message(err_msg):
    return (
        "You've hit the Gemini API free-tier daily request limit for this "
        "project (the account is capped, independent of how many questions "
        "you asked for). Wait for the quota to reset, or add billing to "
        "your Google AI Studio project to raise the limit. "
        "See https://ai.google.dev/gemini-api/docs/rate-limits for details."
    )


def _call_gemini_for_questions(topic, n):
    """A single Gemini call requesting up to `n` questions. Not chunked -
    gemini-3.6-flash's context window comfortably fits 100 questions in one
    request, and every extra call eats into the (very small) free-tier
    daily quota."""
    domain_instruction = (
        f'Every question must belong to the domain "{topic}".'
        if topic in CCNA_DOMAINS
        else "Distribute questions realistically across the 6 CCNA domains "
             "based on the official exam weighting."
    )

    prompt = f"""Generate exactly {n} realistic, non-repeating Cisco CCNA (200-301) exam questions for topic: "{topic}".
{domain_instruction}

Return STRICTLY a JSON array (no markdown, no commentary, no code fences) matching this exact format:
[
  {{
    "question": "Question text",
    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    "answer": "A",
    "explanation": "Clear CCNA technical explanation",
    "domain": "One of: {', '.join(CCNA_DOMAINS)}"
  }}
]

Vary which letter (A/B/C/D) holds the correct answer across questions - do not always place it first."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        err_msg = str(e)
        if _is_quota_error(err_msg):
            raise QuotaExceededError(_friendly_quota_message(err_msg))
        raise RuntimeError(f"Gemini API Error: {err_msg}")

    try:
        parsed = _extract_json_array(response.text)
        if not isinstance(parsed, list):
            raise ValueError("Model did not return a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Gemini returned malformed data: {e}")

    return [q for q in parsed if _validate_question(q)]


@app.route("/generate-exam", methods=["POST"])
def generate_exam():
    data = request.json or {}
    topic = data.get("topic", "All Domains (Full Exam)")
    try:
        count = int(data.get("count", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid question count"}), 400

    count = max(1, min(count, MAX_QUESTIONS_PER_REQUEST))

    try:
        questions = _call_gemini_for_questions(topic, count)
        # Only make a second call if the first one came back meaningfully
        # short (e.g. malformed/truncated items got filtered out) - one
        # follow-up request max, to bound quota usage at 2 calls/exam.
        shortfall = count - len(questions)
        if shortfall > 0 and len(questions) > 0:
            try:
                extra = _call_gemini_for_questions(topic, shortfall)
                questions.extend(extra)
            except (QuotaExceededError, RuntimeError):
                pass  # return what we already have rather than fail the exam
    except QuotaExceededError as e:
        return jsonify({"error": str(e)}), 429
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not questions:
        return jsonify({"error": "Gemini did not return any usable questions. Please try again."}), 500

    return jsonify(questions), 200


@app.route("/save-session", methods=["POST"])
def save_session():
    data = request.json or {}
    topic = data.get("topic", "Unknown")
    score = data.get("score", 0)
    total = data.get("total", 0)
    percentage = round((score / total) * 100, 2) if total > 0 else 0
    status = "PASSED" if percentage >= 85.0 else "FAILED"
    missed_questions = json.dumps(data.get("missed_questions", []))
    domain_stats = json.dumps(data.get("domain_stats", {}))
    sid = session.get("sid")

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO exam_sessions
               (session_id, topic, score, total, percentage, status, missed_questions, domain_stats)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, topic, score, total, percentage, status, missed_questions, domain_stats),
        )
        conn.commit()
        conn.close()
        return jsonify({"status": status, "percentage": percentage}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    sid = session.get("sid")
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, topic, score, total, percentage, status
               FROM exam_sessions WHERE session_id = ? ORDER BY id DESC LIMIT 15""",
            (sid,),
        )
        rows = cursor.fetchall()
        conn.close()
        return jsonify([
            {"timestamp": r[0], "topic": r[1], "score": r[2], "total": r[3],
             "percentage": r[4], "status": r[5]}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai-progress-agent", methods=["GET"])
def ai_progress_agent():
    sid = session.get("sid")
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, topic, score, total, percentage, status, missed_questions, domain_stats
               FROM exam_sessions WHERE session_id = ? ORDER BY id DESC LIMIT 10""",
            (sid,),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return jsonify({"report": "No exam history found yet for this browser session. Take a practice exam to initialize your AI progress agent!"})

        # Aggregate real domain accuracy in Python rather than asking the
        # model to eyeball/guess it from raw history - more trustworthy.
        domain_totals = {d: {"correct": 0, "total": 0} for d in CCNA_DOMAINS}
        history_summary = []
        for r in rows:
            history_summary.append({
                "timestamp": r[0], "topic": r[1], "score": r[2],
                "total": r[3], "percentage": r[4], "status": r[5],
            })
            try:
                ds = json.loads(r[7]) if r[7] else {}
            except json.JSONDecodeError:
                ds = {}
            for domain, stats in ds.items():
                if domain in domain_totals:
                    domain_totals[domain]["correct"] += stats.get("correct", 0)
                    domain_totals[domain]["total"] += stats.get("total", 0)

        domain_lines = []
        for d, s in domain_totals.items():
            if s["total"] > 0:
                pct = round(s["correct"] / s["total"] * 100, 1)
                domain_lines.append(f"- {d}: {s['correct']}/{s['total']} correct ({pct}%)")
            else:
                domain_lines.append(f"- {d}: no data yet")

        agent_prompt = f"""You are an expert CCNA (200-301) AI Mentor Agent. A student has the following measured performance (already computed - use these exact numbers, do not recalculate):

Recent exam sessions:
{json.dumps(history_summary, indent=2)}

Measured accuracy per domain (from actual answers):
{chr(10).join(domain_lines)}

Provide, using clean markdown-style headings:
1. Overall readiness assessment against the 85% passing standard.
2. Domain-by-domain commentary using the measured percentages above (do not invent different numbers).
3. Specific, actionable study advice for the weakest domain with data.

Keep it concise and encouraging but honest.
"""
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=agent_prompt)
        except Exception as e:
            err_msg = str(e)
            if _is_quota_error(err_msg):
                return jsonify({"error": _friendly_quota_message(err_msg)}), 429
            raise

        return jsonify({"report": response.text, "domain_totals": domain_totals})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
