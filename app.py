import os
import sqlite3
import json
import uuid
import secrets
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
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

# Official Cisco exam blueprint weighting (%) - used to prioritize which
# weak domain is actually worth the most study time.
CCNA_DOMAIN_WEIGHTS = {
    "1.0 Network Fundamentals": 20,
    "2.0 Network Access": 20,
    "3.0 IP Connectivity": 25,
    "4.0 IP Services": 10,
    "5.0 Security Fundamentals": 15,
    "6.0 Automation & Programmability": 10,
}


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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tutor_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            cumulative_total_at_session INTEGER,
            report_text TEXT
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


# Optional single-password gate. Set SITE_PASSWORD in your .env (and in
# Render's environment variables) to make the whole app private - anyone
# without the password is redirected to /login. Leave it unset to keep the
# app fully public (e.g. for local development).
SITE_PASSWORD = os.getenv("SITE_PASSWORD")


@app.before_request
def require_auth():
    if not SITE_PASSWORD:
        return  # no password configured -> app stays open
    if request.path == "/login" or request.path.startswith("/static"):
        return
    if session.get("authenticated"):
        return
    if request.method == "GET":
        return redirect(url_for("login"))
    return jsonify({"error": "Unauthorized - please log in."}), 401


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        submitted = request.form.get("password", "")
        if SITE_PASSWORD and secrets.compare_digest(submitted, SITE_PASSWORD):
            session["authenticated"] = True
            session.permanent = True
            return redirect(url_for("home"))
        error = "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("login"))


@app.route("/")
def home():
    return render_template("index.html")


def _domain_short_id(domain):
    # "3.0 IP Connectivity" -> "D3"
    num = domain.split(".")[0]
    return f"D{num}"


def _build_journey(sid, limit=30):
    """Computes readiness, per-domain trend, and a priority-ranked list of
    which weak domain is most worth studying next - purely from data already
    in the DB. No Gemini call involved, so this can be refreshed as often as
    the user likes without touching the API quota."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT timestamp, topic, score, total, percentage, domain_stats
           FROM exam_sessions WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
        (sid, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    def parse_ds(raw):
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    sessions = [
        {"timestamp": r[0], "topic": r[1], "score": r[2], "total": r[3],
         "percentage": r[4], "domain_stats": parse_ds(r[5])}
        for r in rows
    ]

    lifetime_total = sum(s["total"] for s in sessions)
    lifetime_correct = sum(s["score"] for s in sessions)
    readiness = round(lifetime_correct / lifetime_total * 100, 1) if lifetime_total else 0

    cumulative = {d: {"correct": 0, "total": 0} for d in CCNA_DOMAINS}
    for s in sessions:
        for domain, stats in s["domain_stats"].items():
            if domain in cumulative:
                cumulative[domain]["correct"] += stats.get("correct", 0)
                cumulative[domain]["total"] += stats.get("total", 0)

    # Trend = latest session's domain accuracy vs. the cumulative average of
    # every session BEFORE it - "did the most recent attempt move the needle?"
    latest_ds = sessions[0]["domain_stats"]
    prior_cumulative = {d: {"correct": 0, "total": 0} for d in CCNA_DOMAINS}
    for s in sessions[1:]:
        for domain, stats in s["domain_stats"].items():
            if domain in prior_cumulative:
                prior_cumulative[domain]["correct"] += stats.get("correct", 0)
                prior_cumulative[domain]["total"] += stats.get("total", 0)

    domain_rows = []
    gap_ranking = []
    for d in CCNA_DOMAINS:
        cum = cumulative[d]
        if cum["total"] == 0:
            continue
        pct = round(cum["correct"] / cum["total"] * 100)
        status = "READY" if pct >= 85 else ("BUILDING" if pct >= 60 else "FOCUS")

        trend = None
        if d in latest_ds and latest_ds[d].get("total", 0) > 0:
            prior = prior_cumulative[d]
            if prior["total"] > 0:
                latest_pct = latest_ds[d]["correct"] / latest_ds[d]["total"] * 100
                prior_pct = prior["correct"] / prior["total"] * 100
                trend = round(latest_pct - prior_pct)

        domain_rows.append({
            "id": _domain_short_id(d), "domain": d, "pct": pct,
            "trend": trend, "status": status,
        })

        gap = 100 - pct
        if gap > 0:
            weight = CCNA_DOMAIN_WEIGHTS.get(d, 0)
            gap_ranking.append({
                "domain": d, "id": _domain_short_id(d), "gap_points": gap,
                "weight": weight, "priority_score": round(gap * weight / 100, 1),
            })

    gap_ranking.sort(key=lambda x: x["priority_score"], reverse=True)
    gap_ranking = gap_ranking[:3]

    log = []
    for i in range(min(len(sessions), 5)):
        cur = sessions[i]
        entry = f"{cur['topic']} — {cur['percentage']}% overall"
        if i + 1 < len(sessions):
            prev = sessions[i + 1]
            best_domain, best_delta = None, 0
            for d in CCNA_DOMAINS:
                cur_s, prev_s = cur["domain_stats"].get(d), prev["domain_stats"].get(d)
                if cur_s and prev_s and cur_s.get("total") and prev_s.get("total"):
                    delta = (cur_s["correct"] / cur_s["total"] * 100) - (prev_s["correct"] / prev_s["total"] * 100)
                    if abs(delta) > abs(best_delta):
                        best_delta, best_domain = delta, d
            if best_domain and abs(best_delta) >= 5:
                sign = "+" if best_delta > 0 else ""
                entry += f" · {_domain_short_id(best_domain)} {sign}{round(best_delta)}pt vs previous attempt"
        log.append({"when": cur["timestamp"], "text": entry})

    return {
        "readiness": readiness,
        "domain_rows": domain_rows,
        "gap_ranking": gap_ranking,
        "log": log,
        "suggested_focus_domains": [g["domain"] for g in gap_ranking[:2]],
        "sessions_counted": len(sessions),
    }


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


def _call_gemini_for_questions(topic, n, focus_domains=None):
    """A single Gemini call requesting up to `n` questions. Not chunked -
    gemini-3.6-flash's context window comfortably fits 100 questions in one
    request, and every extra call eats into the (very small) free-tier
    daily quota."""
    focus_domains = [d for d in (focus_domains or []) if d in CCNA_DOMAINS]

    if focus_domains:
        domain_instruction = (
            f"Weight the question distribution toward these domains, which "
            f"the student is currently weakest in: {', '.join(focus_domains)}. "
            f"Aim for roughly 60% of questions from these domains combined, "
            f"and spread the remaining 40% realistically across all 6 domains."
        )
    elif topic in CCNA_DOMAINS:
        domain_instruction = f'Every question must belong to the domain "{topic}".'
    else:
        domain_instruction = (
            "Distribute questions realistically across the 6 CCNA domains "
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
    focus_domains = data.get("focus_domains", [])
    try:
        count = int(data.get("count", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid question count"}), 400

    count = max(1, min(count, MAX_QUESTIONS_PER_REQUEST))

    try:
        questions = _call_gemini_for_questions(topic, count, focus_domains)
        # Only make a second call if the first one came back meaningfully
        # short (e.g. malformed/truncated items got filtered out) - one
        # follow-up request max, to bound quota usage at 2 calls/exam.
        shortfall = count - len(questions)
        if shortfall > 0 and len(questions) > 0:
            try:
                extra = _call_gemini_for_questions(topic, shortfall, focus_domains)
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
        journey = _build_journey(sid)
        return jsonify({"status": status, "percentage": percentage, "journey": journey}), 200
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
        journey = _build_journey(sid)
        if journey is None:
            return jsonify({
                "report": "No exam history yet. Take a practice exam to initialize your journey.",
                "journey": None,
            })

        agent_prompt = f"""You are an expert CCNA (200-301) AI Mentor Agent. A student has the following measured performance (already computed - use these exact numbers, do not recalculate):

Overall readiness: {journey['readiness']}% (passing standard: 85%)

Per-domain status:
{json.dumps(journey['domain_rows'], indent=2)}

Domains ranked by study priority (gap size x official exam weight):
{json.dumps(journey['gap_ranking'], indent=2)}

Write 3-4 sentences, encouraging but honest: what's working, and specifically why the #1 priority domain above is the fastest path to a higher score. Do not invent different numbers than the ones given.
"""
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=agent_prompt)
            report_text = response.text
        except Exception as e:
            err_msg = str(e)
            if _is_quota_error(err_msg):
                # Structured journey data costs nothing to compute - only the
                # written narrative needs the API, so degrade gracefully
                # instead of losing the whole report.
                report_text = ("(Coach note unavailable - Gemini daily quota reached. "
                                "The stats below are still accurate; try again after the quota resets.)")
            else:
                raise

        return jsonify({"report": report_text, "journey": journey})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


TUTOR_UNLOCK_THRESHOLD = 100


def _tutor_progress(sid):
    """Pure Python, zero API calls - safe to check as often as the UI wants."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(total), 0) FROM exam_sessions WHERE session_id = ?", (sid,))
    lifetime_total = cursor.fetchone()[0]
    cursor.execute(
        "SELECT cumulative_total_at_session FROM tutor_sessions WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (sid,),
    )
    row = cursor.fetchone()
    conn.close()

    last_tutor_total = row[0] if row else 0
    since_unlock = lifetime_total - last_tutor_total
    return {
        "lifetime_total": lifetime_total,
        "questions_since_last_tutor": since_unlock,
        "threshold": TUTOR_UNLOCK_THRESHOLD,
        "unlocked": since_unlock >= TUTOR_UNLOCK_THRESHOLD,
    }


@app.route("/tutor-status", methods=["GET"])
def tutor_status():
    sid = session.get("sid")
    try:
        return jsonify(_tutor_progress(sid))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tutor-session", methods=["POST"])
def tutor_session():
    sid = session.get("sid")
    try:
        progress = _tutor_progress(sid)
        if not progress["unlocked"]:
            remaining = progress["threshold"] - progress["questions_since_last_tutor"]
            return jsonify({
                "error": f"Tutor session locked - answer {remaining} more question(s) to unlock it.",
                "progress": progress,
            }), 403

        journey = _build_journey(sid)

        # Pull a handful of recent missed questions across ALL history so the
        # tutor can ground its explanation in concrete examples, not just
        # abstract percentages.
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT missed_questions FROM exam_sessions
               WHERE session_id = ? ORDER BY id DESC LIMIT 10""",
            (sid,),
        )
        rows = cursor.fetchall()
        conn.close()

        missed_examples = []
        for r in rows:
            try:
                missed_examples.extend(json.loads(r[0]) if r[0] else [])
            except json.JSONDecodeError:
                pass
        missed_examples = missed_examples[:15]

        tutor_prompt = f"""You are a patient, encouraging CCNA (200-301) tutor sitting down with a student who has just answered {progress['lifetime_total']} practice questions total. This is a milestone check-in, not a quick status update - take your time and actually teach.

Overall readiness: {journey['readiness']}% (passing standard: 85%)

Per-domain status:
{json.dumps(journey['domain_rows'], indent=2)}

Domains ranked by study priority (gap size x official exam weight):
{json.dumps(journey['gap_ranking'], indent=2)}

Specific questions the student recently got wrong:
{json.dumps(missed_examples, indent=2)}

Write a genuine tutoring explanation, not a report:
1. Pick the 2-3 concepts (grounded in the specific missed questions above, not just the domain name) that are most worth understanding right now.
2. Explain each one in plain language, using a concrete analogy or real-world comparison a beginner would grasp - avoid just repeating textbook definitions.
3. Directly reference at least one of the missed questions above to show *why* the underlying concept matters, not just that they got it wrong.
4. Close with honest, specific encouragement about what's genuinely improving.

Keep it warm and conversational, like a good teacher explaining something in office hours - not a bulleted corporate report.
"""
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=tutor_prompt)
        except Exception as e:
            err_msg = str(e)
            if _is_quota_error(err_msg):
                # Don't consume the unlock if we couldn't actually deliver it -
                # the student keeps their milestone and can retry later.
                return jsonify({"error": _friendly_quota_message(err_msg), "progress": progress}), 429
            raise

        report_text = response.text
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO tutor_sessions (session_id, cumulative_total_at_session, report_text)
               VALUES (?, ?, ?)""",
            (sid, progress["lifetime_total"], report_text),
        )
        conn.commit()
        conn.close()

        return jsonify({"report": report_text, "progress": _tutor_progress(sid)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tutor-history", methods=["GET"])
def tutor_history():
    sid = session.get("sid")
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, cumulative_total_at_session, report_text
               FROM tutor_sessions WHERE session_id = ? ORDER BY id DESC LIMIT 10""",
            (sid,),
        )
        rows = cursor.fetchall()
        conn.close()
        return jsonify([
            {"timestamp": r[0], "questions_covered": r[1], "report": r[2]}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
