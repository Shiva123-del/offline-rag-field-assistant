import os
import re
import sys

# Offline assistant: use the embedding model stored on this computer and do
# not contact Hugging Face. For the very first download, run once with:
#   HF_HUB_OFFLINE=0 python src/ingest.py
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pymupdf
from sentence_transformers import SentenceTransformer
import chromadb

# Which manual to ingest: python src/ingest.py [path/to/manual.pdf]
PDF_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/manuals/TM 9-2320-280-10 HMMWV Operators Manual.pdf"
)

VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = "field_manuals"

# Smaller chunks keep related lines together, instead of burying them
# inside a long, mixed passage.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def clean_cell(cell, line_separator=" "):
    """Collapse whitespace in a table cell; None becomes an empty string."""
    if not cell:
        return ""

    lines = [" ".join(line.split()) for line in cell.split("\n")]
    return line_separator.join(line for line in lines if line)


def extract_troubleshooting_rows(page):
    """Read cause/action tables as structured rows.

    Flattened PDF text loses which cause belongs to which action (and to
    which problem, e.g. "GENSET Not Started" vs "GENSET Not Generates").
    PyMuPDF can read the real table, so each row keeps its pairing.

    Returns (rows, table_areas). table_areas are the page regions covered
    by these tables, so their garbled flat text can be left out.
    """
    rows = []
    table_areas = []

    for table in page.find_tables().tables:
        data = table.extract()
        if not data:
            continue

        header = " ".join(clean_cell(cell) for cell in data[0]).lower()
        if "cause" not in header or "action" not in header:
            continue

        table_areas.append(pymupdf.Rect(table.bbox))
        group = ""

        for row in data[1:]:
            cells = list(row) + [None] * (3 - len(row))

            # The first column names the problem and is only filled on the
            # first row of its group, so remember it for the rows below.
            if clean_cell(cells[0]):
                group = clean_cell(cells[0])

            # " / " keeps two causes that share one cell visibly separate.
            cause = clean_cell(cells[1], line_separator=" / ")
            action = clean_cell(cells[2])

            if cause and action:
                rows.append({
                    "group": group,
                    "cause": cause,
                    "action": action
                })

    return rows, table_areas


_SKIP_LINE = re.compile(
    r"^(TM \d|Change \d|MALFUNCTION$|TEST OR INSPECTION$|CORRECTIVE ACTION$|"
    r"Table \d+-\d+\.|\d+\s*-\s*\d+$)"
)


def fix_ocr(text):
    """Repair scan errors seen in this manual's headings ("UGHT" = "LIGHT")."""
    return text.replace("UGHT", "LIGHT").replace("CEIUNG", "CEILING")


def is_stepwise_troubleshooting_page(text):
    """Pages laid out as MALFUNCTION / TEST OR INSPECTION / CORRECTIVE ACTION."""
    return "MALFUNCTION" in text and "CORRECTIVE ACTION" in text


def page_lines(page):
    """Text lines of a page as (x, y, text), top to bottom, without margins."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text and 90 < line["bbox"][1] < 715:
                lines.append((line["bbox"][0], line["bbox"][1], text))
    return sorted(lines, key=lambda item: (round(item[1]), item[0]))


def extract_stepwise_troubleshooting(page, page_number, state, rows):
    """Read military-style troubleshooting lists into cause/action rows.

    Layout: a numbered MALFUNCTION, then "Step N." tests/inspections, each
    followed by its corrective action. The indentation differs from page to
    page, so rows are rebuilt from structure instead: a test or inspection
    is a sentence ending in a full stop, and what follows it (up to the next
    step) is its corrective action. `state` carries the open malfunction and
    step across pages, because a list can continue on the next page.
    """
    def close_step():
        step = state.get("step")
        state["step"] = None
        if not step or not state.get("group"):
            return
        cause = " ".join(step["test"]).strip()
        action = " ".join(step["action"]).strip()
        rows.append({
            "page": step["page"],
            "group": state["group"],
            "system": state.get("system", ""),
            "cause": cause,
            "action": action or "No separate corrective action is listed; "
                                "follow the step as written.",
        })

    lines = [
        line for line in page_lines(page) if not _SKIP_LINE.match(line[2])
    ]
    if not lines:
        return
    left = min(x for x, _, _ in lines)

    index = 0
    while index < len(lines):
        x, _, text = lines[index]
        index += 1

        numbered = re.match(r"^(\d+)\.\s*(.*)$", text)
        # Indentation varies even within a page, so a malfunction is
        # recognised by its number plus an all-capitals title.
        if numbered and (not numbered.group(2) or numbered.group(2).isupper()):
            title = numbered.group(2)
            if not title and index < len(lines):
                title = lines[index][2]
                index += 1
            close_step()
            state["number"] = numbered.group(1)
            state["title"] = fix_ocr(title.strip())
            state["group"] = f"{state['number']}. {state['title']}"
            state["in_title"] = True
            continue

        # Centered all-capitals line = system heading (ENGINE, BRAKES, ...).
        if x > left + 60 and text.isupper():
            close_step()
            state["system"] = text.title()
            continue

        if state.get("in_title") and text.isupper() and not text.startswith("STEP"):
            state["title"] += " " + fix_ocr(text)
            state["group"] = f"{state['number']}. {state['title']}"
            continue
        state["in_title"] = False

        step = state.get("step")
        if re.match(r"^Step\s*\d+\.", text):
            close_step()
            text = re.sub(r"^Step\s*\d+\.\s*", "", text)
            state["step"] = {"page": page_number, "test": [text], "action": []}
        elif step is None:
            state["step"] = {"page": page_number, "test": [text], "action": []}
        elif not step["action"] and not step["test"][-1].endswith("."):
            step["test"].append(text)
        else:
            step["action"].append(text)

    state["close"] = close_step


def extract_maintenance_rows(page):
    """Read the maintenance schedule (item x period) as structured rows.

    Each row becomes one sentence-like entry, e.g. "Replace Engine Oil:
    First Month: Replace; Every 6 Months: Replace", so "how often" questions
    can be answered from the real grid instead of garbled flat text.

    Returns (rows, table_areas) like extract_troubleshooting_rows().
    """
    rows = []
    table_areas = []

    for table in page.find_tables().tables:
        data = table.extract()
        if len(data) < 3:
            continue

        header = [clean_cell(cell) for cell in data[1]]
        if not header or header[0].lower() != "item" or "daily" not in (
            cell.lower() for cell in header
        ):
            continue

        table_areas.append(pymupdf.Rect(table.bbox))
        periods = header[3:]

        for row in data[2:]:
            item = clean_cell(row[0])
            if not item:
                continue

            cells = [clean_cell(cell, line_separator=" ") for cell in row[3:]]

            # A cell spanning every period (e.g. Fuel Tank) comes back as one
            # value followed by None, so it belongs to no single period.
            if cells and cells[0] and all(cell is None for cell in row[4:]):
                rows.append({"item": item, "schedule": cells[0]})
                continue

            schedule = [
                f"{period}: {cell}"
                for period, cell in zip(periods, cells)
                if cell
            ]
            if schedule:
                rows.append({"item": item, "schedule": "; ".join(schedule)})

    return rows, table_areas


def text_outside_areas(page, areas):
    """Page text, leaving out blocks that sit inside the given areas."""
    if not areas:
        return page.get_text().strip()

    blocks = sorted(
        page.get_text("blocks"),
        key=lambda block: (round(block[1]), block[0])
    )

    kept = []
    for block in blocks:
        is_text = block[6] == 0
        inside_table = any(
            pymupdf.Rect(block[:4]).intersects(area) for area in areas
        )

        if is_text and not inside_table:
            kept.append(block[4].strip())

    return "\n".join(text for text in kept if text).strip()


def extract_pages(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = []
    table_rows = []
    maintenance_rows = []
    step_state = {}

    for page_number, page in enumerate(doc, start=1):
        rows, table_areas = extract_troubleshooting_rows(page)
        schedule, schedule_areas = extract_maintenance_rows(page)
        text = text_outside_areas(page, table_areas + schedule_areas)

        # Troubleshooting written as numbered steps, not as a ruled table.
        if not rows and is_stepwise_troubleshooting_page(page.get_text()):
            extract_stepwise_troubleshooting(
                page, page_number, step_state, table_rows
            )
            # The rows replace the page's flat text (same words, no pairing).
            text = ""

        for row in schedule:
            maintenance_rows.append({"page": page_number, **row})

        for row in rows:
            table_rows.append({
                "page": page_number,
                **row
            })

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    if step_state.get("close"):
        step_state["close"]()

    doc.close()
    return pages, table_rows, maintenance_rows


def create_chunks(pages):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            # Prefer to end the chunk at a line break so table rows and
            # sentences are not cut in half.
            if end < len(text):
                line_break = text.rfind("\n", start + CHUNK_SIZE // 2, end)
                if line_break != -1:
                    end = line_break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page": page_number,
                    "text": chunk_text,
                    "kind": "text",
                    "group": ""
                })

            if end >= len(text):
                break

            # Always move forward, even if the overlap would step backwards.
            start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def create_table_chunks(table_rows):
    """One chunk per troubleshooting row, with cause and action together."""
    return [
        {
            "page": row["page"],
            "text": (
                "TROUBLESHOOTING TABLE ROW | "
                f"Problem: {row['group']} | "
                + (f"System: {row['system']} | " if row.get("system") else "")
                + f"Cause: {row['cause']} | "
                f"Action: {row['action']}"
            ),
            "kind": "table_row",
            "group": row["group"]
        }
        for row in table_rows
    ]


def create_maintenance_chunks(maintenance_rows):
    """One chunk per maintenance-schedule item."""
    return [
        {
            "page": row["page"],
            "text": (
                "MAINTENANCE SCHEDULE ROW (how often to service) | "
                f"Item: {row['item']} | {row['schedule']}"
            ),
            "kind": "maintenance_row",
            "group": "Maintenance schedule"
        }
        for row in maintenance_rows
    ]


def store_in_chromadb(chunks, embeddings):
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Replace the old collection to avoid duplicate or stale chunks.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "page": chunk["page"],
                "source": PDF_PATH.split("/")[-1],
                "kind": chunk["kind"],
                "group": chunk["group"]
            }
            for chunk in chunks
        ]
    )

    return collection


if __name__ == "__main__":
    print("Extracting PDF pages and troubleshooting tables...")
    pages, table_rows, maintenance_rows = extract_pages(PDF_PATH)
    print(f"Extracted {len(pages)} pages.")
    print(f"Extracted {len(table_rows)} troubleshooting table rows.")
    print(f"Extracted {len(maintenance_rows)} maintenance schedule rows.")

    print("Creating chunks...")
    chunks = (
        create_chunks(pages)
        + create_table_chunks(table_rows)
        + create_maintenance_chunks(maintenance_rows)
    )
    print(f"Created {len(chunks)} chunks.")

    print("Loading local embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("Saving chunks to ChromaDB...")
    collection = store_in_chromadb(chunks, embeddings)

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print("Ingestion completed successfully.")
