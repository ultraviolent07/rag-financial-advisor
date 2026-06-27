"""
Document parsing utilities for PDFs, TXT, and DOCX files
Updated: streaming page processing for long docs, batch chunking, no double-parse
"""
from typing import List, Tuple, Generator
from pathlib import Path
import os
import re
from datetime import datetime
import pdfplumber
from docx import Document as DocxDocument

MAX_PAGES = 200          # hard cap — beyond this splits into multiple ingestions
BATCH_SIZE = 20          # process N pages at a time to avoid memory spike

# ============= PDF Parser =============

class PDFParser:
    """Parse PDF files — streaming for long docs"""

    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[str, List[str]]:
        """
        Extract text from PDF, streaming in batches of BATCH_SIZE pages.
        Caps at MAX_PAGES to prevent timeout on huge files.
        Returns: (full_text, page_texts)
        """
        try:
            full_text = ""
            page_texts = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(total_pages, MAX_PAGES)

                if total_pages > MAX_PAGES:
                    print(f"⚠️  PDF has {total_pages} pages — processing first {MAX_PAGES}")

                # Process in batches to avoid memory spike
                for batch_start in range(0, pages_to_process, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, pages_to_process)

                    for page_num in range(batch_start, batch_end):
                        page = pdf.pages[page_num]
                        text = page.extract_text() or ""
                        labeled = f"\n[Page {page_num + 1}]\n{text}\n"
                        full_text += labeled
                        page_texts.append(text)

            return full_text, page_texts

        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return "", []

    @staticmethod
    def extract_text_and_tables(pdf_path: str) -> dict:
        """
        Extract text AND tables, streaming in batches.
        Tables extracted only from first 50 pages (financial tables usually upfront).
        """
        try:
            full_text = ""
            all_tables = []
            pages_data = []
            TABLE_PAGE_LIMIT = 50  # don't extract tables from every page of a 139-pager

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(total_pages, MAX_PAGES)

                for batch_start in range(0, pages_to_process, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, pages_to_process)

                    for page_num in range(batch_start, batch_end):
                        page = pdf.pages[page_num]
                        page_text = page.extract_text() or ""

                        # Only extract tables from early pages
                        if page_num < TABLE_PAGE_LIMIT:
                            page_tables = page.extract_tables() or []
                            all_tables.extend(page_tables)
                        else:
                            page_tables = []

                        full_text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                        pages_data.append({
                            "page_num": page_num + 1,
                            "text": page_text,
                            "tables": page_tables
                        })

            return {
                "text": full_text,
                "tables": all_tables,
                "pages": pages_data,
                "num_pages": len(pages_data),
                "total_pages_in_file": total_pages
            }

        except Exception as e:
            print(f"Error extracting PDF with tables: {e}")
            return {"text": "", "tables": [], "pages": [], "num_pages": 0, "total_pages_in_file": 0}

    @staticmethod
    def stream_pages(pdf_path: str) -> Generator[Tuple[int, str], None, None]:
        """
        Generator — yields (page_num, page_text) one page at a time.
        Use this for very large docs to avoid loading everything into memory.
        """
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages[:MAX_PAGES]):
                text = page.extract_text() or ""
                yield page_num + 1, text


# ============= Text File Parser =============

class TextParser:
    @staticmethod
    def extract_text(txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""


# ============= DOCX Parser =============

class DocxParser:
    @staticmethod
    def extract_text(docx_path: str) -> str:
        try:
            doc = DocxDocument(docx_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return ""


# ============= Universal Document Parser =============

class DocumentParser:
    SUPPORTED_FORMATS = {".pdf", ".txt", ".docx"}

    @staticmethod
    def parse(file_path: str) -> Tuple[str, dict]:
        """
        Parse any supported document.
        Returns: (text_content, metadata)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in DocumentParser.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {extension}")

        metadata = {
            "filename": file_path.name,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "created_date": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "format": extension[1:].upper()
        }

        if extension == ".pdf":
            # Single pass — no double parse
            result = PDFParser.extract_text_and_tables(str(file_path))
            text = result["text"]
            metadata["pages"] = result["num_pages"]
            metadata["total_pages_in_file"] = result.get("total_pages_in_file", result["num_pages"])
            metadata["tables_found"] = len(result["tables"])

        elif extension == ".txt":
            text = TextParser.extract_text(str(file_path))

        elif extension == ".docx":
            text = DocxParser.extract_text(str(file_path))

        else:
            text = ""

        metadata["char_count"] = len(text)
        return text, metadata

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into word-aligned chunks (not character-aligned).
        Prevents cutting mid-word/mid-sentence.
        """
        words = text.split()
        # Convert char sizes to approximate word counts
        words_per_chunk = chunk_size // 5      # ~5 chars per word
        words_overlap = overlap // 5

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(" ".join(chunk_words))
            i += words_per_chunk - words_overlap

        return [c for c in chunks if c.strip()]

    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """Split text into chunks by sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def chunk_for_rag(text: str) -> List[str]:
        """
        Recommended chunking for RAG — sentence-based with reasonable size.
        Works well with the parent-child chunker in rag_pipeline.py.
        """
        return DocumentParser.chunk_by_sentences(text, sentences_per_chunk=6)


# ============= Document Loader =============

class DocumentLoader:
    """Load documents and prepare for RAG indexing"""

    def __init__(self, chunk_strategy: str = "sentences"):
        self.chunk_strategy = chunk_strategy
        self.parser = DocumentParser()

    def load_and_chunk(self, file_path: str, chunk_size: int = 500) -> dict:
        """
        Load document and chunk for RAG.
        Returns: {filename, chunks, metadata, chunk_count}
        """
        text, metadata = self.parser.parse(file_path)

        if self.chunk_strategy == "overlap":
            chunks = self.parser.chunk_text(text, chunk_size=chunk_size)
        elif self.chunk_strategy == "rag":
            chunks = self.parser.chunk_for_rag(text)
        else:
            chunks = self.parser.chunk_by_sentences(text)

        print(f"📄 {metadata['filename']}: {metadata.get('pages', '?')} pages → {len(chunks)} chunks")

        return {
            "filename": metadata["filename"],
            "chunks": chunks,
            "chunk_count": len(chunks),
            "metadata": metadata,
            "total_characters": len(text)
        }

    def load_directory(self, directory: str) -> List[dict]:
        """Load all supported documents from a directory"""
        documents = []
        dir_path = Path(directory)

        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() in DocumentParser.SUPPORTED_FORMATS:
                try:
                    doc = self.load_and_chunk(str(file_path))
                    documents.append(doc)
                    print(f"✅ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")

        return documents


# ============= Example Usage =============

if __name__ == "__main__":
    print("Testing document parsing...\n")
    pdf_path = "sample.pdf"

    if os.path.exists(pdf_path):
        parser = DocumentParser()
        text, metadata = parser.parse(pdf_path)

        print(f"File:        {metadata['filename']}")
        print(f"Pages:       {metadata.get('pages', 'N/A')}")
        print(f"Total pages: {metadata.get('total_pages_in_file', 'N/A')}")
        print(f"Chars:       {metadata['char_count']}")
        print(f"Tables:      {metadata.get('tables_found', 0)}")
        print(f"First 200:   {text[:200]}...")

        chunks = parser.chunk_for_rag(text)
        print(f"\nRAG chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0][:150]}...")
    else:
        print("No sample.pdf found.")

    loader = DocumentLoader(chunk_strategy="rag")
    docs = loader.load_directory("./data/uploads")
    print(f"\n✅ Loaded {len(docs)} documents")