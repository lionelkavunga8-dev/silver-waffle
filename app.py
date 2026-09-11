import os
import json
import uuid
import secrets
import hashlib
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from google import genai
import psycopg2

# 1. Load environment variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file!")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is missing. Set it to your Postgres connection string "
        "(e.g. from Neon or Supabase) - exam history won't survive restarts without it."
    )
# Some providers (e.g. Heroku-style URLs) use the old 'postgres://' scheme,
# which psycopg2 rejects - normalize it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
if not FLASK_SECRET_KEY:
    raise ValueError(
        "FLASK_SECRET_KEY is missing. Without a fixed value here, every server "
        "restart invalidates everyone's login/session cookies, which looks like "
        "your progress 'resetting'. Generate one with: "
        "python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and set it as an env var."
    )

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

client = genai.Client(api_key=api_key)


@app.errorhandler(Exception)
def handle_any_error(e):
    # Guarantees every response is JSON, even for errors this code doesn't
    # anticipate (proxy timeouts aside) - the frontend always calls
    # res.json() and should never receive an HTML error page.
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": f"Server error: {e}"}), 500


def get_db():
    """New connection per call - simplest safe pattern for a low-traffic app
    with a hosted Postgres (Neon/Supabase) sitting behind a pooler anyway."""
    return psycopg2.connect(DATABASE_URL)


MODEL_NAME = "gemini-3.6-flash"  # primary
FALLBACK_MODEL_NAME = "gemini-3.5-flash-lite"  # separate quota bucket - Google
# scopes free-tier limits per-model, so a 429 on the primary doesn't mean the
# fallback is exhausted too.
GEMINI_DAILY_LIMIT = int(os.getenv("GEMINI_DAILY_LIMIT", "20"))
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


# Curated CCNA subtopics - MUST stay in sync with CCNA_TOPICS in
# templates/index.html (used for the search box). Kept as a fixed, bounded
# vocabulary (not freeform LLM labels) so per-concept accuracy actually
# accumulates cleanly instead of fragmenting across near-duplicate phrasings.
CCNA_SUBTOPICS = [
    "OSI and TCP/IP models", "IPv4 addressing and subnetting", "IPv6 addressing",
    "Network topology architectures", "Physical interfaces and cabling", "Wireless principles",
    "Virtualization fundamentals (VMs/containers)", "Switching concepts (MAC table, frame forwarding)",
    "VLANs and inter-VLAN routing", "Trunking (802.1Q)", "EtherChannel (LACP/PAgP)",
    "Spanning Tree Protocol (STP/RSTP)", "Wireless LAN architectures", "WLC and AP management",
    "Routing table components", "Router forwarding decisions", "Static routing (IPv4/IPv6)",
    "OSPFv2 single area", "First hop redundancy (HSRP)",
    "NAT/PAT", "DHCP and DNS", "NTP", "SNMP and Syslog", "QoS concepts", "TFTP/FTP file transfer",
    "Security concepts (threats, vulnerabilities)", "Access Control Lists (standard/extended)",
    "Layer 2 security (port security, DHCP snooping)", "Wireless security (WPA2/WPA3)",
    "Remote access VPN", "AAA concepts", "Device hardening and passwords/MFA",
    "Impact of automation on networking", "Controller-based vs traditional networking",
    "REST API characteristics", "JSON data encoding",
    "Configuration management tools (Puppet/Chef/Ansible)",
    "CLI navigation and modes (user/privileged/global config)",
    "Basic device configuration (hostname, passwords, banners)",
    "Interpreting show command output",
]
CCNA_SUBTOPIC_SET = set(CCNA_SUBTOPICS)
MIN_CONCEPT_SAMPLE = 2  # don't surface a "weak concept" off a single lucky/unlucky guess
MIN_DOMAIN_SAMPLE_FOR_PREDICTION = 5  # domains with less data than this are excluded from the predicted score


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cumulative_total_at_session INTEGER,
            report_text TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            usage_date DATE NOT NULL,
            model_name TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (usage_date, model_name)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS question_history (
            session_id TEXT NOT NULL,
            question_hash TEXT NOT NULL,
            question_text TEXT NOT NULL,
            domain TEXT,
            concept TEXT,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            times_seen INTEGER DEFAULT 1,
            PRIMARY KEY (session_id, question_hash)
        )
    ''')
    # Safe no-op migration for any pre-existing table missing newer columns.
    cursor.execute("ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS session_id TEXT")
    cursor.execute("ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS domain_stats TEXT")
    cursor.execute("ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS concept_stats TEXT")
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
    """Computes readiness (three ways), per-domain trend, and priority-ranked
    lists of what's most worth studying next - both at the broad domain
    level and the specific concept level - purely from data already in the
    DB. No Gemini call involved, so this can be refreshed as often as the
    user likes without touching the API quota."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT timestamp, topic, score, total, percentage, domain_stats, concept_stats
           FROM exam_sessions WHERE session_id = %s ORDER BY id DESC LIMIT %s""",
        (sid, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    def parse_json(raw):
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    sessions = [
        {"timestamp": r[0], "topic": r[1], "score": r[2], "total": r[3],
         "percentage": r[4], "domain_stats": parse_json(r[5]), "concept_stats": parse_json(r[6])}
        for r in rows
    ]

    lifetime_total = sum(s["total"] for s in sessions)
    lifetime_correct = sum(s["score"] for s in sessions)
    readiness_lifetime = round(lifetime_correct / lifetime_total * 100, 1) if lifetime_total else 0

    # Recent-window readiness: pooled accuracy over just the last 5 SESSIONS.
    # Simple and matches how most people intuitively think about "am I
    # improving lately" - but note this counts sessions, not questions, so
    # five 1-question sessions is a much noisier signal than five 50-question
    # sessions. Shown alongside the lifetime number, not as a replacement.
    recent_sessions = sessions[:5]
    recent_total = sum(s["total"] for s in recent_sessions)
    recent_correct = sum(s["score"] for s in recent_sessions)
    readiness_recent = round(recent_correct / recent_total * 100, 1) if recent_total else None

    cumulative = {d: {"correct": 0, "total": 0} for d in CCNA_DOMAINS}
    for s in sessions:
        for domain, stats in s["domain_stats"].items():
            if domain in cumulative:
                cumulative[domain]["correct"] += stats.get("correct", 0)
                cumulative[domain]["total"] += stats.get("total", 0)

    # Predicted exam score: weights each domain's accuracy by its REAL exam
    # weight, instead of pooling everything equally (which silently overweights
    # whichever domain you happen to have practiced the most). Domains with too
    # little data are excluded and the remaining weights are renormalized -
    # this is an honest "based on what we actually know" estimate, not a guess
    # about domains you haven't meaningfully touched yet.
    weighted_sum, weight_covered, domains_covered = 0.0, 0, 0
    for d in CCNA_DOMAINS:
        cum = cumulative[d]
        if cum["total"] >= MIN_DOMAIN_SAMPLE_FOR_PREDICTION:
            pct = cum["correct"] / cum["total"] * 100
            w = CCNA_DOMAIN_WEIGHTS.get(d, 0)
            weighted_sum += pct * w
            weight_covered += w
            domains_covered += 1
    readiness_predicted = round(weighted_sum / weight_covered, 1) if weight_covered else None
    predicted_coverage = f"{domains_covered}/6"

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

    # Concept-level gaps: the same idea as domain gap_ranking, but at the
    # specific-subtopic level, so "build focus exam" can target exactly what
    # you're missing (e.g. "OSPFv2 single area") instead of just a broad
    # domain. Requires MIN_CONCEPT_SAMPLE attempts before a concept is
    # eligible, so one unlucky guess doesn't get flagged as a weak spot.
    concept_cumulative = {}
    concept_domain_map = {}
    for s in sessions:
        for concept, stats in s["concept_stats"].items():
            if concept not in CCNA_SUBTOPIC_SET:
                continue
            bucket = concept_cumulative.setdefault(concept, {"correct": 0, "total": 0})
            bucket["correct"] += stats.get("correct", 0)
            bucket["total"] += stats.get("total", 0)
            if "domain" in stats:
                concept_domain_map[concept] = stats["domain"]

    concept_gaps = []
    for concept, cum in concept_cumulative.items():
        if cum["total"] < MIN_CONCEPT_SAMPLE:
            continue
        pct = round(cum["correct"] / cum["total"] * 100)
        gap = 100 - pct
        if gap <= 0:
            continue
        parent_domain = concept_domain_map.get(concept)
        weight = CCNA_DOMAIN_WEIGHTS.get(parent_domain, 15)  # mild default if unknown
        concept_gaps.append({
            "concept": concept, "domain": parent_domain, "pct": pct,
            "gap_points": gap, "attempts": cum["total"],
            "priority_score": round(gap * weight / 100, 1),
        })
    concept_gaps.sort(key=lambda x: x["priority_score"], reverse=True)
    concept_gaps = concept_gaps[:5]

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
        "readiness": readiness_lifetime,
        "readiness_recent": readiness_recent,
        "readiness_recent_sessions": len(recent_sessions),
        "readiness_predicted": readiness_predicted,
        "readiness_predicted_coverage": predicted_coverage,
        "domain_rows": domain_rows,
        "gap_ranking": gap_ranking,
        "concept_gaps": concept_gaps,
        "log": log,
        "suggested_focus_domains": [g["domain"] for g in gap_ranking[:2]],
        "suggested_focus_concepts": [c["concept"] for c in concept_gaps[:3]],
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
    # Same softness for concept - if the model didn't copy a subtopic label
    # verbatim, don't discard the question, just leave it untracked at the
    # concept level (domain-level stats still work fine either way).
    if q.get("concept") not in CCNA_SUBTOPIC_SET:
        q["concept"] = None
    return True


class QuotaExceededError(Exception):
    """Raised when Gemini returns 429/RESOURCE_EXHAUSTED on BOTH the primary
    and fallback model. Never worth retrying automatically - free-tier daily
    caps don't clear in seconds."""
    pass


def _is_quota_error(err_msg):
    lowered = err_msg.lower()
    return "429" in err_msg or "resource_exhausted" in lowered or "quota" in lowered


def _is_fallback_worthy(err_msg):
    # Covers both "you personally are out of quota" (429) and "the model is
    # overloaded right now for everyone" (503/UNAVAILABLE) - both are good
    # reasons to try a different model rather than fail outright.
    lowered = err_msg.lower()
    return (
        _is_quota_error(err_msg)
        or "503" in err_msg or "unavailable" in lowered or "overloaded" in lowered
    )


def _friendly_quota_message(err_msg):
    return (
        "Both the primary and fallback Gemini models are unavailable right now "
        "(daily quota reached or the API is under heavy load). Wait a bit and "
        "try again, or add billing to your Google AI Studio project to raise "
        "the limit. See https://ai.google.dev/gemini-api/docs/rate-limits for details."
    )


def _record_model_usage(model_name):
    # Best-effort - usage tracking should never break the actual feature.
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO api_usage (usage_date, model_name, count)
               VALUES (CURRENT_DATE, %s, 1)
               ON CONFLICT (usage_date, model_name)
               DO UPDATE SET count = api_usage.count + 1""",
            (model_name,),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _generate_with_fallback(prompt, config=None):
    """Tries the primary model first. If it's rate-limited or overloaded,
    automatically retries once against a model with its own separate quota
    bucket, instead of failing outright. Returns (response, model_used)."""
    kwargs = {"contents": prompt}
    if config is not None:
        kwargs["config"] = config

    try:
        response = client.models.generate_content(model=MODEL_NAME, **kwargs)
        _record_model_usage(MODEL_NAME)
        return response, MODEL_NAME
    except Exception as e:
        primary_err = str(e)
        _record_model_usage(MODEL_NAME)  # the attempt still counted against quota
        if not _is_fallback_worthy(primary_err):
            raise RuntimeError(f"Gemini API Error: {primary_err}")

    try:
        response = client.models.generate_content(model=FALLBACK_MODEL_NAME, **kwargs)
        _record_model_usage(FALLBACK_MODEL_NAME)
        return response, FALLBACK_MODEL_NAME
    except Exception as e2:
        fallback_err = str(e2)
        _record_model_usage(FALLBACK_MODEL_NAME)
        if _is_fallback_worthy(fallback_err):
            raise QuotaExceededError(_friendly_quota_message(fallback_err))
        raise RuntimeError(f"Gemini API Error: {fallback_err}")


@app.route("/quota-status", methods=["GET"])
def quota_status():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT count FROM api_usage WHERE usage_date = CURRENT_DATE AND model_name = %s",
            (MODEL_NAME,),
        )
        row = cursor.fetchone()
        conn.close()
        used = row[0] if row else 0
        remaining = max(0, GEMINI_DAILY_LIMIT - used)
        remaining_pct = round(remaining / GEMINI_DAILY_LIMIT * 100) if GEMINI_DAILY_LIMIT else 0
        return jsonify({
            "used": used, "limit": GEMINI_DAILY_LIMIT, "remaining": remaining,
            "remaining_pct": remaining_pct, "model": MODEL_NAME,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _question_hash(text):
    normalized = " ".join((text or "").strip().lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


def _record_question_history(sid, questions):
    """Records that these questions were shown to this session - powers both
    duplicate-avoidance (recent question texts) and topic coverage tracking
    (which concepts have been shown at all). Best-effort: never blocks the
    actual exam if it fails."""
    if not questions:
        return
    try:
        conn = get_db()
        cursor = conn.cursor()
        for q in questions:
            h = _question_hash(q.get("question", ""))
            cursor.execute(
                """INSERT INTO question_history (session_id, question_hash, question_text, domain, concept)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id, question_hash)
                   DO UPDATE SET times_seen = question_history.times_seen + 1, last_seen = CURRENT_TIMESTAMP""",
                (sid, h, (q.get("question") or "")[:500], q.get("domain"), q.get("concept")),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _get_recent_question_texts(sid, limit=30):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT question_text FROM question_history
               WHERE session_id = %s ORDER BY last_seen DESC LIMIT %s""",
            (sid, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _get_coverage(sid):
    """Which of the fixed, bounded set of official CCNA subtopics has this
    session been shown at least one question about? This is the achievable
    version of 'have I seen everything' - not literally every possible
    question (unbounded), but every named item on the real exam blueprint."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT concept FROM question_history WHERE session_id = %s AND concept IS NOT NULL",
            (sid,),
        )
        covered = {r[0] for r in cursor.fetchall() if r[0] in CCNA_SUBTOPIC_SET}
        conn.close()
    except Exception:
        covered = set()
    uncovered = [t for t in CCNA_SUBTOPICS if t not in covered]
    return {
        "covered_count": len(covered),
        "total": len(CCNA_SUBTOPICS),
        "uncovered": uncovered,
    }


@app.route("/coverage-status", methods=["GET"])
def coverage_status():
    sid = session.get("sid")
    try:
        return jsonify(_get_coverage(sid))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _call_gemini_for_questions(topic, n, focus_domains=None, focus_topics=None, avoid_questions=None):
    """A single Gemini call requesting up to `n` questions. Not chunked -
    gemini-3.6-flash's context window comfortably fits 100 questions in one
    request, and every extra call eats into the (very small) free-tier
    daily quota."""
    focus_domains = [d for d in (focus_domains or []) if d in CCNA_DOMAINS]
    # Cap and sanitize: these come from free-text-adjacent client input, so
    # bound both the count and length of any single topic string.
    focus_topics = [str(t)[:80] for t in (focus_topics or [])][:8]
    avoid_questions = [str(q)[:200] for q in (avoid_questions or [])][:30]

    if focus_topics:
        domain_instruction = (
            f"Generate every question specifically about these exact CCNA subtopics, "
            f"and nothing else: {', '.join(focus_topics)}. Distribute questions "
            f"roughly evenly across the listed subtopics. Still tag each question's "
            f"\"domain\" field with whichever of the 6 official domains it actually belongs to."
        )
    elif focus_domains:
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

    avoid_instruction = ""
    if avoid_questions:
        avoid_list = "\n".join(f'- "{q}"' for q in avoid_questions)
        avoid_instruction = f"""

The student has already been asked these questions recently - do NOT repeat them or generate close rephrasings of them. Ask about different specific facts, commands, or scenarios instead:
{avoid_list}"""

    prompt = f"""Generate exactly {n} realistic, non-repeating Cisco CCNA (200-301) exam questions for topic: "{topic}".
{domain_instruction}{avoid_instruction}

Return STRICTLY a JSON array (no markdown, no commentary, no code fences) matching this exact format:
[
  {{
    "question": "Question text",
    "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
    "answer": "A",
    "explanation": "Clear CCNA technical explanation",
    "domain": "One of: {', '.join(CCNA_DOMAINS)}",
    "concept": "The single closest match from this exact list (copy it verbatim, do not paraphrase): {', '.join(CCNA_SUBTOPICS)}"
  }}
]

Vary which letter (A/B/C/D) holds the correct answer across questions - do not always place it first."""

    try:
        response, used_model = _generate_with_fallback(
            prompt, config={"response_mime_type": "application/json"}
        )
    except QuotaExceededError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {e}")

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
    focus_topics = data.get("focus_topics", [])
    sid = session.get("sid")
    try:
        count = int(data.get("count", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid question count"}), 400

    count = max(1, min(count, MAX_QUESTIONS_PER_REQUEST))
    avoid_questions = _get_recent_question_texts(sid)

    try:
        questions = _call_gemini_for_questions(topic, count, focus_domains, focus_topics, avoid_questions)
        # Only make a second call if the first one came back meaningfully
        # short (e.g. malformed/truncated items got filtered out) - one
        # follow-up request max, to bound quota usage at 2 calls/exam.
        shortfall = count - len(questions)
        if shortfall > 0 and len(questions) > 0:
            try:
                # Also avoid repeating whatever this same batch just generated.
                extra_avoid = avoid_questions + [q["question"] for q in questions]
                extra = _call_gemini_for_questions(topic, shortfall, focus_domains, focus_topics, extra_avoid)
                questions.extend(extra)
            except (QuotaExceededError, RuntimeError):
                pass  # return what we already have rather than fail the exam
    except QuotaExceededError as e:
        return jsonify({"error": str(e)}), 429
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if not questions:
        return jsonify({"error": "Gemini did not return any usable questions. Please try again."}), 500

    _record_question_history(sid, questions)

    return jsonify(questions), 200


def _extract_json_object(text):
    """Same defensive markdown-fence stripping as _extract_json_array, for
    single-object JSON responses (lab scenarios, lab grading)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


@app.route("/generate-lab", methods=["POST"])
def generate_lab():
    data = request.json or {}
    topic = data.get("topic")  # optional - a specific subtopic string
    topic_instruction = (
        f'The scenario must be about this specific CCNA subtopic: "{topic}".'
        if topic in CCNA_SUBTOPIC_SET
        else "Pick any realistic CLI configuration task from across the CCNA 200-301 blueprint."
    )

    prompt = f"""Generate one realistic Cisco IOS CLI configuration lab exercise for CCNA 200-301 practice.
{topic_instruction}

Return STRICTLY a JSON object (no markdown, no commentary) matching this exact format:
{{
  "scenario": "A short, specific task description, e.g. 'Configure the hostname of this router to R1 and set an enable secret of cisco123.'",
  "expected_answer": "The canonical command(s) that accomplish the task, newline-separated if more than one.",
  "acceptable_variations": ["one or two other valid ways to phrase/abbreviate the same commands, e.g. using 'conf t' instead of 'configure terminal'"],
  "domain": "One of: {', '.join(CCNA_DOMAINS)}",
  "concept": "The single closest match from this exact list (copy it verbatim): {', '.join(CCNA_SUBTOPICS)}"
}}

Keep the scenario to one or two sentences. Prefer common, exam-realistic tasks (interface config, routing, ACLs, VLANs, NAT, basic device setup) over obscure edge cases."""

    try:
        response, used_model = _generate_with_fallback(prompt, config={"response_mime_type": "application/json"})
    except QuotaExceededError as e:
        return jsonify({"error": str(e)}), 429
    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {e}"}), 500

    try:
        lab = _extract_json_object(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Gemini returned malformed data: {e}"}), 500

    if not isinstance(lab, dict) or not lab.get("scenario") or not lab.get("expected_answer"):
        return jsonify({"error": "Gemini did not return a usable lab scenario. Please try again."}), 500
    if lab.get("concept") not in CCNA_SUBTOPIC_SET:
        lab["concept"] = None
    if lab.get("domain") not in CCNA_DOMAINS:
        lab["domain"] = "Unspecified"

    return jsonify(lab), 200


@app.route("/check-lab-answer", methods=["POST"])
def check_lab_answer():
    data = request.json or {}
    scenario = str(data.get("scenario", ""))[:500]
    expected_answer = str(data.get("expected_answer", ""))[:500]
    acceptable_variations = [str(v)[:300] for v in data.get("acceptable_variations", [])][:5]
    user_answer = str(data.get("user_answer", ""))[:500]

    if not scenario or not expected_answer:
        return jsonify({"error": "Missing scenario or expected answer."}), 400
    if not user_answer.strip():
        return jsonify({"error": "Type a command before checking."}), 400

    prompt = f"""You are grading a Cisco IOS CLI lab exercise. Be lenient about command abbreviations, ordering
of independent commands, and equivalent valid syntax (e.g. "conf t" = "configure terminal", "int gi0/1" = "interface gigabitethernet0/1").
Be strict about actually accomplishing the task correctly.

Scenario: {scenario}
Canonical correct answer: {expected_answer}
Other acceptable variations: {json.dumps(acceptable_variations)}
Student's answer: {user_answer}

Return STRICTLY a JSON object (no markdown, no commentary):
{{
  "correct": true or false,
  "feedback": "1-2 sentences: if correct, briefly confirm why. If incorrect, explain specifically what's wrong or missing, without just repeating the canonical answer verbatim."
}}"""

    try:
        response, used_model = _generate_with_fallback(prompt, config={"response_mime_type": "application/json"})
    except QuotaExceededError as e:
        return jsonify({"error": str(e)}), 429
    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {e}"}), 500

    try:
        result = _extract_json_object(response.text)
    except (json.JSONDecodeError, ValueError):
        return jsonify({"error": "Could not grade that answer, please try again."}), 500

    if not isinstance(result, dict) or "correct" not in result:
        return jsonify({"error": "Could not grade that answer, please try again."}), 500

    return jsonify({"correct": bool(result.get("correct")), "feedback": result.get("feedback", "")}), 200


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
    concept_stats = json.dumps(data.get("concept_stats", {}))
    sid = session.get("sid")

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO exam_sessions
               (session_id, topic, score, total, percentage, status, missed_questions, domain_stats, concept_stats)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (sid, topic, score, total, percentage, status, missed_questions, domain_stats, concept_stats),
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
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, topic, score, total, percentage, status
               FROM exam_sessions WHERE session_id = %s ORDER BY id DESC LIMIT 15""",
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
            response, used_model = _generate_with_fallback(agent_prompt)
            report_text = response.text
        except Exception as e:
            err_msg = str(e)
            if isinstance(e, QuotaExceededError) or _is_fallback_worthy(err_msg):
                # Structured journey data costs nothing to compute - only the
                # written narrative needs the API, so degrade gracefully
                # instead of losing the whole report.
                report_text = ("(Coach note unavailable - both Gemini models are rate-limited "
                                "or overloaded right now. The stats below are still accurate; try again shortly.)")
            else:
                raise

        return jsonify({"report": report_text, "journey": journey})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


TUTOR_UNLOCK_THRESHOLD = 100


def _tutor_progress(sid):
    """Pure Python, zero API calls - safe to check as often as the UI wants."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(total), 0) FROM exam_sessions WHERE session_id = %s", (sid,))
    lifetime_total = cursor.fetchone()[0]
    cursor.execute(
        "SELECT cumulative_total_at_session FROM tutor_sessions WHERE session_id = %s ORDER BY id DESC LIMIT 1",
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
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT missed_questions FROM exam_sessions
               WHERE session_id = %s ORDER BY id DESC LIMIT 10""",
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
            response, used_model = _generate_with_fallback(tutor_prompt)
        except Exception as e:
            err_msg = str(e)
            if isinstance(e, QuotaExceededError) or _is_fallback_worthy(err_msg):
                # Don't consume the unlock if we couldn't actually deliver it -
                # the student keeps their milestone and can retry later.
                return jsonify({"error": _friendly_quota_message(err_msg), "progress": progress}), 429
            raise

        report_text = response.text
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO tutor_sessions (session_id, cumulative_total_at_session, report_text)
               VALUES (%s, %s, %s)""",
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
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, cumulative_total_at_session, report_text
               FROM tutor_sessions WHERE session_id = %s ORDER BY id DESC LIMIT 10""",
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
