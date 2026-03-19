"""
Local financial report reader tool.
Supports CSV, Excel, and PDF files placed in the user's data directory:
  ~/.config/fin-agent/reports/
or a custom path specified at call time.
"""

import os
import json
import re
import pandas as pd
from fin_agent.config import Config

# Max characters returned to LLM from a PDF to avoid context overflow
_PDF_MAX_CHARS = 12000


def _get_reports_dir():
    return os.path.join(Config.get_config_dir(), "reports")


def _resolve_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(_get_reports_dir(), filename)


# ── PDF helpers ────────────────────────────────────────────────────────────────

def _extract_pdf_tables(filepath):
    """
    Extract tables from a PDF using pdfplumber.
    Returns a list of dicts: {page, table_index, dataframe}.
    """
    import pdfplumber
    results = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                # Use first row as header if it looks like one
                header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[0])]
                rows = table[1:]
                df = pd.DataFrame(rows, columns=header)
                results.append({"page": page_num, "table_index": t_idx, "df": df})
    return results


def _extract_pdf_text(filepath, page_start=None, page_end=None):
    """
    Extract plain text from a PDF, optionally limited to a page range.
    """
    import pdfplumber
    texts = []
    with pdfplumber.open(filepath) as pdf:
        total = len(pdf.pages)
        p_start = max(1, page_start or 1)
        p_end = min(total, page_end or total)
        for page in pdf.pages[p_start - 1: p_end]:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)


def _pdf_page_count(filepath):
    import pdfplumber
    with pdfplumber.open(filepath) as pdf:
        return len(pdf.pages)


# ── Public functions ───────────────────────────────────────────────────────────

def list_local_reports():
    """
    List all available local financial report files (CSV, Excel, PDF).
    Returns a JSON list of file names found in the reports directory.
    """
    reports_dir = _get_reports_dir()
    if not os.path.exists(reports_dir):
        return json.dumps({
            "reports_dir": reports_dir,
            "files": [],
            "message": "Reports directory does not exist. Create it and place CSV/Excel/PDF files there."
        })

    files = [
        f for f in os.listdir(reports_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls", ".pdf"))
    ]
    return json.dumps({"reports_dir": reports_dir, "files": sorted(files)}, ensure_ascii=False)


def get_report_columns(filename, sheet_name=None):
    """
    Get the column names and a sample of a local report file.
    For PDF files, returns page count and a text preview instead.

    :param filename: File name or absolute path.
    :param sheet_name: Sheet name for Excel files.
    :return: JSON with structure info.
    """
    filepath = _resolve_path(filename)
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}"

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".pdf":
            page_count = _pdf_page_count(filepath)
            preview = _extract_pdf_text(filepath, page_start=1, page_end=2)[:1000]
            tables = _extract_pdf_tables(filepath)
            table_summary = [
                {"page": t["page"], "table_index": t["table_index"], "columns": list(t["df"].columns)}
                for t in tables[:10]  # show first 10 tables
            ]
            return json.dumps({
                "filename": filename,
                "type": "pdf",
                "total_pages": page_count,
                "tables_found": len(tables),
                "table_summary": table_summary,
                "text_preview": preview
            }, ensure_ascii=False)

        elif ext == ".csv":
            df = pd.read_csv(filepath, dtype=str, nrows=5)
            total = sum(1 for _ in open(filepath)) - 1
        elif ext in (".xlsx", ".xls"):
            kwargs = {"sheet_name": sheet_name, "nrows": 5} if sheet_name else {"nrows": 5}
            df = pd.read_excel(filepath, dtype=str, **kwargs)
            full = pd.read_excel(filepath, dtype=str)
            total = len(full)
        else:
            return f"Error: Unsupported format '{ext}'."
    except Exception as e:
        return f"Error reading file: {e}"

    df.columns = [c.strip() for c in df.columns]
    return json.dumps({
        "filename": filename,
        "columns": list(df.columns),
        "sample": df.head(3).to_dict(orient="records"),
        "total_rows": total
    }, ensure_ascii=False)


def read_local_report(filename, sheet_name=None, ts_code=None, period=None):
    """
    Read a local financial report file (CSV, Excel, or PDF).

    For CSV/Excel: supports filtering by ts_code and period columns.
    For PDF: extracts all tables and text. Use page_start/page_end via
             read_pdf_pages for targeted extraction.

    :param filename: File name or absolute path.
    :param sheet_name: Sheet name for Excel files.
    :param ts_code: Filter by stock code column. Optional.
    :param period: Filter by period/date column. Optional.
    :return: JSON string of the data.
    """
    filepath = _resolve_path(filename)
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}"

    ext = os.path.splitext(filepath)[1].lower()

    # ── PDF branch ─────────────────────────────────────────────────────────────
    if ext == ".pdf":
        try:
            tables = _extract_pdf_tables(filepath)
            text = _extract_pdf_text(filepath)
        except ImportError:
            return "Error: pdfplumber is not installed. Run: pip install pdfplumber"
        except Exception as e:
            return f"Error reading PDF: {e}"

        result = {"filename": filename, "tables": [], "text": ""}

        # Serialize tables
        for t in tables:
            df = t["df"]
            df.columns = [str(c).strip() for c in df.columns]
            result["tables"].append({
                "page": t["page"],
                "table_index": t["table_index"],
                "data": df.to_dict(orient="records")
            })

        # Truncate text to avoid overwhelming the LLM context
        result["text"] = text[:_PDF_MAX_CHARS]
        if len(text) > _PDF_MAX_CHARS:
            result["text_truncated"] = True
            result["total_chars"] = len(text)
            result["hint"] = "Use read_pdf_pages to read specific page ranges."

        return json.dumps(result, ensure_ascii=False)

    # ── CSV / Excel branch ─────────────────────────────────────────────────────
    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, dtype=str)
        elif ext in (".xlsx", ".xls"):
            kwargs = {"sheet_name": sheet_name} if sheet_name else {}
            df = pd.read_excel(filepath, dtype=str, **kwargs)
        else:
            return f"Error: Unsupported format '{ext}'. Use CSV, Excel, or PDF."
    except Exception as e:
        return f"Error reading file: {e}"

    df.columns = [c.strip() for c in df.columns]

    if ts_code:
        code_cols = [c for c in df.columns if c.lower() in ("ts_code", "股票代码", "code", "symbol")]
        if code_cols:
            df = df[df[code_cols[0]].str.strip() == ts_code.strip()]
        else:
            return f"Error: No stock code column found. Available columns: {list(df.columns)}"

    if period:
        period_cols = [c for c in df.columns if c.lower() in ("end_date", "period", "报告期", "date", "ann_date", "f_ann_date")]
        if period_cols:
            df = df[df[period_cols[0]].str.strip() == period.strip()]
        else:
            return f"Error: No period column found. Available columns: {list(df.columns)}"

    if df.empty:
        return f"No data found after filtering (ts_code={ts_code}, period={period})."

    return df.to_json(orient="records", force_ascii=False)


def read_pdf_pages(filename, page_start=1, page_end=None, extract_tables=True):
    """
    Read specific pages from a PDF report.
    Useful for large PDFs where read_local_report truncates the text.

    :param filename: File name or absolute path.
    :param page_start: First page to read (1-indexed).
    :param page_end: Last page to read (inclusive). Defaults to page_start + 9.
    :param extract_tables: If True, also extract tables from the page range.
    :return: JSON with text and optionally tables.
    """
    filepath = _resolve_path(filename)
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}"

    if not filepath.lower().endswith(".pdf"):
        return "Error: This function only supports PDF files."

    try:
        import pdfplumber
    except ImportError:
        return "Error: pdfplumber is not installed. Run: pip install pdfplumber"

    # Default window: 10 pages
    if page_end is None:
        page_end = page_start + 9

    try:
        total_pages = _pdf_page_count(filepath)
        page_end = min(page_end, total_pages)

        text = _extract_pdf_text(filepath, page_start=page_start, page_end=page_end)
        result = {
            "filename": filename,
            "page_start": page_start,
            "page_end": page_end,
            "total_pages": total_pages,
            "text": text[:_PDF_MAX_CHARS]
        }
        if len(text) > _PDF_MAX_CHARS:
            result["text_truncated"] = True

        if extract_tables:
            import pdfplumber
            tables = []
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages[page_start - 1: page_end], start=page_start):
                    for t_idx, table in enumerate(page.extract_tables()):
                        if not table:
                            continue
                        header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[0])]
                        rows = table[1:]
                        df = pd.DataFrame(rows, columns=header)
                        tables.append({
                            "page": page_num,
                            "table_index": t_idx,
                            "data": df.to_dict(orient="records")
                        })
            result["tables"] = tables

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return f"Error reading PDF pages: {e}"


# ── Tool Schema ────────────────────────────────────────────────────────────────

LOCAL_REPORT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_local_reports",
            "description": (
                "List all local financial report files (CSV, Excel, PDF) in the user's reports directory. "
                "Call this first to discover available files."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_report_columns",
            "description": (
                "Inspect the structure of a local report file. "
                "For CSV/Excel: returns column names and a data sample. "
                "For PDF: returns page count, detected tables summary, and a text preview. "
                "Always call this before read_local_report to understand the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name or absolute path."},
                    "sheet_name": {"type": "string", "description": "Sheet name for Excel files. Optional."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_report",
            "description": (
                "Read a local financial report file (CSV, Excel, or PDF). "
                "For PDF files: extracts all tables and text (truncated to 12000 chars if large). "
                "For CSV/Excel: supports filtering by stock code and reporting period. "
                "If the PDF text is truncated, use read_pdf_pages to read specific page ranges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name or absolute path."},
                    "sheet_name": {"type": "string", "description": "Sheet name for Excel files. Optional."},
                    "ts_code": {"type": "string", "description": "Filter by stock code (CSV/Excel only). Optional."},
                    "period": {"type": "string", "description": "Filter by reporting period e.g. '20231231' (CSV/Excel only). Optional."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf_pages",
            "description": (
                "Read specific pages from a PDF financial report. "
                "Use this when read_local_report truncates the content, or when you need to focus on "
                "a specific section (e.g., pages 30-50 for the financial statements). "
                "Returns text and tables from the specified page range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "PDF file name or absolute path."},
                    "page_start": {"type": "integer", "description": "First page to read (1-indexed). Default 1."},
                    "page_end": {"type": "integer", "description": "Last page to read (inclusive). Defaults to page_start + 9."},
                    "extract_tables": {"type": "boolean", "description": "Whether to extract tables. Default true."}
                },
                "required": ["filename"]
            }
        }
    }
]
