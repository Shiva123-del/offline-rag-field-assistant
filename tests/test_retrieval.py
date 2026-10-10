"""Offline regression checks for troubleshooting retrieval (HMMWV manual).

No Ollama needed: a deliberately incomplete model answer is fed to
postprocess_answer(), as the small model often forgets rows.
Run:  venv/bin/python tests/test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag import postprocess_answer, retrieve_context


def table_bullets(question, model_bullets):
    context = retrieve_context(question)
    answer = "**Troubleshooting table**\n" + "\n".join(model_bullets)
    answer += "\n**Other relevant information**\nNothing relevant.\n**Limitations**\nx"
    text, _ = postprocess_answer(answer, context)
    section = text.split("**Other relevant information**")[0]
    return context, [l for l in section.splitlines() if l.startswith("•")]


# (question, incomplete model bullets, problem, expected rows, must contain)
CASES = [
    ("Why does the engine crank but not start?",
     ["• Cause: Check to see if fuel gauge indicates E (empty). -> Action: Fill fuel tank, and start engine."],
     "3. ENGINE CRANKS BUT DOES NOT START", 4, "WAIT-TO-START"),
    ("Why is the engine overheating?",
     ["• Cause: Check for debris blocking radiator fins. -> Action: Remove debris."],
     "7. ENGINE OVERHEATS", 5, "fan belts"),
    ("Engine fails to crank",
     ["• Cause: Other causes. -> Action: Notify unit maintenance."],
     "1. ENGINE FAILS TO CRANK", 4, "slave-start"),
    ("What should I do if the vehicle wanders to one side?",
     ["• Cause: Check tire air pressure. -> Action: Inflate or deflate tires to correct air pressure."],
     "21. VEHICLE WANDERS", 2, "tire air pressure"),
    ("What if the transmission dipstick is loose?",
     ["• Cause: Check to see if transmission dipstick is loose. -> Action: Secure dipstick."],
     "12. FLUID THROWN", 1, "Secure dipstick"),
]

failures = 0
for question, model, problem, expected, needle in CASES:
    context, bullets = table_bullets(question, model)
    ok = (
        len(bullets) == expected
        and any(needle in b for b in bullets)
        and all(problem in b for b in bullets)
        and sum(not c["is_table"] for c in context) <= 2
    )
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}: {question} -> {len(bullets)} rows "
          f"(expected {expected}), {len(context)} chunks")

# General (non-table) questions must still find the right manual text.
for question, needle in [
    ("How do I slave start the vehicle?", "SLAVE STARTING"),
    ("What are the warnings about exhaust gases?", "EXHAUST"),
]:
    ok = any(needle in c["text"].upper() for c in retrieve_context(question))
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}: {question} -> passage found")

sys.exit(1 if failures else 0)
