"""
RAG tools exposed to the LLM agent.

Tools:
  - index_document_to_rag: Index a local report file into the RAG knowledge base
  - search_knowledge: Semantic search over indexed documents
  - list_rag_sources: List all indexed sources
  - delete_rag_source: Remove a source from the index
"""

import json
import os
from fin_agent.config import Config


# ── Tool implementations ───────────────────────────────────────────────────────

def index_document_to_rag(filename: str) -> str:
    """
    Read a local report file and index it into the RAG knowledge base.
    Supports PDF, TXT. CSV/Excel are better queried via read_local_report.

    :param filename: File name in the reports directory (or absolute path)
    """
    from fin_agent.rag import index_document
    from fin_agent.tools.local_report_tools import _resolve_path, _extract_pdf_text, _pdf_page_count

    filepath = _resolve_path(filename)
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}"

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            total_pages = _pdf_page_count(filepath)
            text = _extract_pdf_text(filepath, 1, total_pages)
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            return f"Error: Unsupported file type '{ext}' for RAG indexing. Supported: .pdf, .txt"

        if not text.strip():
            return f"Error: No text content extracted from {filename}."

        added, total = index_document(text, source=filename)
        return json.dumps({
            "status": "ok",
            "file": filename,
            "new_chunks_added": added,
            "total_chunks_in_index": total
        }, ensure_ascii=False)

    except ImportError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error indexing {filename}: {e}"


def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    Semantic search over the RAG knowledge base.

    :param query: Natural language question or keywords
    :param top_k: Number of relevant chunks to return (default 5)
    """
    from fin_agent.rag import search

    try:
        results = search(query, top_k=top_k)
        if not results:
            return "Knowledge base is empty or no relevant content found. Use index_document_to_rag to add documents first."
        return json.dumps(results, ensure_ascii=False, indent=2)
    except ImportError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error searching knowledge base: {e}"


def list_rag_sources() -> str:
    """List all documents currently indexed in the RAG knowledge base."""
    from fin_agent.rag import list_indexed_sources

    try:
        sources = list_indexed_sources()
        if not sources:
            return "Knowledge base is empty. Use index_document_to_rag to add documents."
        return json.dumps(sources, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {e}"


def delete_rag_source(source: str) -> str:
    """Remove a document source from the RAG knowledge base."""
    from fin_agent.rag import delete_source

    try:
        removed = delete_source(source)
        return json.dumps({"status": "ok", "source": source, "chunks_removed": removed}, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


# ── Tool schema (OpenAI function-calling format) ───────────────────────────────

RAG_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "index_document_to_rag",
            "description": (
                "Index a local financial report file (PDF or TXT) into the RAG knowledge base "
                "so it can be semantically searched. Call this before using search_knowledge on a new file. "
                "Files must be placed in the reports directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name in the reports directory, e.g. 'ningde-2025.pdf'"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Semantic search over the RAG knowledge base. Use this to answer questions about "
                "indexed financial reports, annual reports, research notes, etc. "
                "Returns the most relevant text chunks with their source and relevance score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or keywords to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to return. Default 5."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_rag_sources",
            "description": "List all documents currently indexed in the RAG knowledge base, with chunk counts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_rag_source",
            "description": "Remove a document source from the RAG knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source name to remove (as shown by list_rag_sources)."
                    }
                },
                "required": ["source"]
            }
        }
    }
]
