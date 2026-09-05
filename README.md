# CCNA 200-301 Exam Question Generator

An AI-powered web application built with Python Flask and Google's Gemini 3.6 Flash API. It generates realistic Cisco CCNA exam questions dynamically, tracks live user scores, and stores exam history in an SQLite database.

## Features
- **6 CCNA Domains:** Practice across specific topics like Network Fundamentals, IP Connectivity, and Security.
- **Custom Question Sets:** Generate 1, 5, 20, or 50 questions per session.
- **Instant Feedback:** Get immediate right/wrong scoring with detailed technical explanations.
- **Score Tracking:** Test results are automatically saved to an SQLite database (`exam_scores.db`).

## Tech Stack
- **Backend:** Python 3, Flask, SQLite3
- **AI Engine:** Google GenAI SDK (`gemini-3.6-flash`)
- **Frontend:** HTML5, Modern CSS, Asynchronous JavaScript (Fetch API)

## Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/ccna-exam-app.git](https://github.com/your-username/ccna-exam-app.git)
   cd ccna-exam-app
