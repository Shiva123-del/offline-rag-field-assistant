"""Local web UI for the offline field assistant.

Run:  venv/bin/python src/app.py      then open http://127.0.0.1:8000
Uses only the Python standard library for the server, so it works offline
and needs no new dependencies. It listens on localhost only.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rag

HOST, PORT = "127.0.0.1", 8000

# Shown in the page header and as one-click example questions.
MANUAL_NAME = "HMMWV M998 series · Operator's Manual (TM 9-2320-280-10)"
EXAMPLES = [
    "Why does the engine crank but not start?",
    "Why is the engine overheating?",
    "What should I do if the brakes are dragging?",
    "What if the vehicle is not charging according to the voltmeter?",
    "What are the warnings about exhaust gases?",
]
PAGE = (
    (Path(__file__).parent / "ui.html")
    .read_text(encoding="utf-8")
    .replace("__MANUAL__", MANUAL_NAME)
    .replace("__EXAMPLES__", json.dumps(EXAMPLES))
    .encode("utf-8")
)

# The embedding model, index and Ollama handle one question at a time.
_lock = threading.Lock()


def answer_question(question):
    with _lock:
        context = rag.retrieve_context(question)
        raw = rag.generate_answer(question, context)
        answer, notes = rag.postprocess_answer(raw, context)

    problems = rag.check_answer(answer, context)
    sources = [
        {
            "page": item["page"],
            "table": bool(item.get("is_table")),
            "text": " ".join(item["text"].split())[:400],
        }
        for item in context
    ]
    return {
        "answer": answer,
        "notes": notes,
        "warnings": problems,
        "sources": sources,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, b"Not found", "text/plain")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("question"), str):
                raise ValueError("body must be a JSON object with a text 'question'")
            question = payload["question"].strip()
            if not question:
                raise ValueError("empty question")
            result = answer_question(question[:500])
            status = 200
        except (ValueError, KeyError, json.JSONDecodeError):
            result, status = {"error": "Please type a question."}, 400
        except Exception as error:  # e.g. Ollama not running
            result = {
                "error": "Could not generate an answer. Is Ollama running? "
                         f"({type(error).__name__})"
            }
            status = 500

        self._send(status, json.dumps(result).encode(), "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    rag.get_retriever()  # load models before the first question
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Field Assistant running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
