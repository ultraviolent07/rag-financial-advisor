"""
RAG Pipeline — Tier 2 Upgrade
Implements: parent-child chunking, hybrid BM25+ChromaDB retrieval,
RRF fusion, query expansion, improved prompt with fallback to general knowledge
"""
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import os
import re

# ============= Embeddings =============

class EmbeddingsManager:
    """Manages Hugging Face embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# ============= Parent-Child Chunker =============

class ParentChildChunker:
    """
    Tier 2: Parent-child chunking
    Index small child chunks (150 tokens) for precise retrieval.
    Return large parent chunks (600 tokens) to LLM for full context.
    Fixes: LLM hallucinating context around a retrieved chunk.
    """

    def __init__(self, child_size: int = 150, parent_size: int = 600, overlap: int = 30):
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def _split_words(self, text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def chunk_document(self, text: str, doc_id: str) -> Tuple[List[dict], List[dict]]:
        """
        Returns (child_chunks, parent_chunks)
        Each chunk: {"id": str, "text": str, "parent_id": str (for children)}
        """
        parents = self._split_words(text, self.parent_size, self.overlap)
        parent_chunks = []
        child_chunks = []

        for p_idx, parent_text in enumerate(parents):
            parent_id = f"{doc_id}_p{p_idx}"
            parent_chunks.append({"id": parent_id, "text": parent_text})

            # Split parent into children
            children = self._split_words(parent_text, self.child_size, self.overlap)
            for c_idx, child_text in enumerate(children):
                child_id = f"{parent_id}_c{c_idx}"
                child_chunks.append({
                    "id": child_id,
                    "text": child_text,
                    "parent_id": parent_id
                })

        return child_chunks, parent_chunks


# ============= Vector Store =============

class ChromaVectorStore:
    """Chroma vector database — stores child chunks for retrieval"""

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = None

    def create_collection(self, name: str = "financial_docs"):
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return self.collection

    def add_documents(self,
                      documents: List[str],
                      ids: List[str],
                      metadatas: List[dict] = None):
        if metadatas is None:
            metadatas = [{} for _ in documents]
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, query_text: str, n_results: int = 10) -> Tuple[List[str], List[float], List[dict]]:
        """Returns (docs, distances, metadatas)"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )
        docs = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        return docs, distances, metadatas


# ============= BM25 Index =============

class BM25Index:
    """
    Tier 2: Keyword-based retrieval
    Catches exact matches (ticker names, dates, "Q3 2024") that semantic search misses.
    """

    def __init__(self):
        self.index = None
        self.documents = []
        self.doc_ids = []

    def build(self, documents: List[str], doc_ids: List[str]):
        self.documents = documents
        self.doc_ids = doc_ids
        tokenized = [doc.lower().split() for doc in documents]
        self.index = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Returns list of (doc_text, score, doc_id)"""
        if self.index is None or not self.documents:
            return []
        tokens = query_text.lower().split()
        scores = self.index.get_scores(tokens)
        ranked = sorted(zip(self.documents, scores, self.doc_ids),
                        key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ============= RRF Fusion =============

def reciprocal_rank_fusion(
    semantic_results: List[Tuple[str, str]],   # (doc_text, doc_id)
    bm25_results: List[Tuple[str, str]],        # (doc_text, doc_id)
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Tier 2: Reciprocal Rank Fusion
    Combines semantic + keyword ranked lists without needing a learned model.
    Formula: score = sum(1 / (k + rank)) across all lists.
    Returns sorted list of (doc_text, fused_score).
    """
    scores: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    for rank, (text, doc_id) in enumerate(semantic_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    for rank, (text, doc_id) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(texts[doc_id], scores[doc_id]) for doc_id in sorted_ids]


# ============= RAG Retriever =============

class RAGRetriever:
    """
    Tier 2 RAG Retriever
    - Query expansion: 3 reformulations before retrieval
    - Hybrid retrieval: BM25 + ChromaDB semantic search
    - RRF fusion: combines both ranked lists
    - Parent-child: returns parent chunks (600 tokens) to LLM
    """

    def __init__(self,
                 embeddings_manager: EmbeddingsManager,
                 vector_store: ChromaVectorStore,
                 groq_client=None):
        self.embeddings = embeddings_manager
        self.vector_store = vector_store
        self.bm25_index = BM25Index()
        self.chunker = ParentChildChunker()
        self.groq_client = groq_client  # for query expansion

        # Parent chunk store: parent_id -> parent_text
        self._parent_store: Dict[str, str] = {}
        # child_id -> parent_id mapping
        self._child_to_parent: Dict[str, str] = {}

    def ingest_document(self, text: str, doc_id: str, metadata: dict = None):
        """
        Chunk document into parent-child, store children in ChromaDB,
        keep parents in memory for context retrieval.
        """
        if metadata is None:
            metadata = {}

        child_chunks, parent_chunks = self.chunker.chunk_document(text, doc_id)

        # Store parents
        for p in parent_chunks:
            self._parent_store[p["id"]] = p["text"]

        # Store child->parent mapping
        for c in child_chunks:
            self._child_to_parent[c["id"]] = c["parent_id"]

        # Add children to ChromaDB
        child_texts = [c["text"] for c in child_chunks]
        child_ids = [c["id"] for c in child_chunks]
        child_metas = [{**metadata, "parent_id": c["parent_id"]} for c in child_chunks]

        if child_texts:
            self.vector_store.add_documents(child_texts, child_ids, child_metas)

        # Rebuild BM25 on all children
        all_docs, all_ids = self._get_all_children_for_bm25(child_texts, child_ids)
        self.bm25_index.build(all_docs, all_ids)

    def _get_all_children_for_bm25(self, new_texts, new_ids):
        """Merge new children with existing BM25 docs"""
        existing = list(zip(self.bm25_index.documents, self.bm25_index.doc_ids)) \
            if self.bm25_index.documents else []
        combined = existing + list(zip(new_texts, new_ids))
        if not combined:
            return [], []
        texts, ids = zip(*combined)
        return list(texts), list(ids)

    def _expand_query(self, query: str) -> List[str]:
        """
        Tier 2: Query expansion via LLM
        Generates 3 reformulations to catch paraphrased content in docs.
        Falls back to original query if Groq not available.
        """
        if self.groq_client is None:
            return [query]

        try:
            prompt = f"""Generate 3 different reformulations of this financial query.
Return ONLY the 3 queries, one per line, no numbering or extra text.

Original query: {query}"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            expanded = response.choices[0].message.content.strip().split("\n")
            expanded = [q.strip() for q in expanded if q.strip()]
            return [query] + expanded[:3]  # original + up to 3 expansions
        except Exception:
            return [query]

    def _get_parent_text(self, child_id: str, child_text: str) -> Tuple[str, str]:
        """Return parent text if available, else child text"""
        parent_id = self._child_to_parent.get(child_id)
        if parent_id and parent_id in self._parent_store:
            return self._parent_store[parent_id], parent_id
        return child_text, child_id

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Full Tier 2 retrieval pipeline:
        1. Expand query
        2. Semantic search (ChromaDB) for each expansion
        3. BM25 keyword search for each expansion
        4. RRF fusion across all results
        5. Return parent chunks for top results
        """
        expanded_queries = self._expand_query(query)

        all_semantic: List[Tuple[str, str]] = []  # (text, id)
        all_bm25: List[Tuple[str, str]] = []

        for q in expanded_queries:
            # Semantic search
            try:
                docs, distances, metas = self.vector_store.query(q, n_results=10)
                for doc, dist, meta in zip(docs, distances, metas):
                    if dist < 0.8:  # filter low relevance
                        child_id = meta.get("id", doc[:20])
                        all_semantic.append((doc, child_id))
            except Exception:
                pass

            # BM25 keyword search
            bm25_results = self.bm25_index.query(q, top_k=10)
            for doc, score, doc_id in bm25_results:
                if score > 0:
                    all_bm25.append((doc, doc_id))

        # Deduplicate
        seen_sem = {}
        for text, doc_id in all_semantic:
            if doc_id not in seen_sem:
                seen_sem[doc_id] = text
        seen_bm25 = {}
        for text, doc_id in all_bm25:
            if doc_id not in seen_bm25:
                seen_bm25[doc_id] = text

        semantic_deduped = [(v, k) for k, v in seen_sem.items()]
        bm25_deduped = [(v, k) for k, v in seen_bm25.items()]

        # RRF fusion
        if semantic_deduped or bm25_deduped:
            fused = reciprocal_rank_fusion(semantic_deduped, bm25_deduped)
        else:
            # fallback: direct semantic query on original
            try:
                docs, _, _ = self.vector_store.query(query, n_results=top_k)
                return docs[:top_k]
            except Exception:
                return []

        # Get top-k, upgrade to parent chunks
        top_results = fused[:top_k]
        parent_texts = []
        seen_parents = set()

        for child_text, _ in top_results:
            # Try to find child_id to get parent
            child_id = None
            for cid, pid in self._child_to_parent.items():
                if self._parent_store.get(pid, "")[:100] in child_text or child_text[:100] in self._parent_store.get(pid, ""):
                    child_id = cid
                    break

            parent_text, parent_id = self._get_parent_text(child_id or "", child_text)
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_texts.append(parent_text)

        return parent_texts if parent_texts else [r[0] for r in top_results]

    def get_context_string(self, query: str, top_k: int = 5) -> str:
        """Get formatted context string for LLM prompt"""
        docs = self.retrieve_context(query, top_k)
        context = "\n\n---\n\n".join([f"Document {i+1}:\n{doc}"
                                      for i, doc in enumerate(docs)])
        return context


# ============= LLM Chain =============

class LLMChain:
    """LLM-powered chain with RAG context"""

    def __init__(self, retriever: RAGRetriever, llm_client=None):
        self.retriever = retriever
        self.llm = llm_client

    def build_prompt(self, query: str, context: str, history: List[dict] = None) -> str:
        if history is None:
            history = []

        history_str = ""
        if history:
            history_str = "Previous conversation:\n"
            for msg in history[-4:]:
                history_str += f"- {msg['role'].upper()}: {msg['content'][:200]}...\n"

        prompt = f"""You are a financial analyst AI assistant with deep expertise in markets, stocks, and investments.

{history_str}

CONTEXT FROM DOCUMENTS:
{context}

QUESTION:
{query}

INSTRUCTIONS:
1. First answer using the provided documents — cite which document when relevant
2. If specific data is not in documents, STILL answer using your general financial knowledge
3. Clearly separate: "From documents: ..." vs "From general knowledge: ..."
4. Be specific with numbers, dates, and formulas where applicable
5. Explain your reasoning step by step
"""
        return prompt

    def generate_response(self, query: str, history: List[dict] = None) -> Tuple[str, List[str]]:
        context = self.retriever.get_context_string(query, top_k=5)
        prompt = self.build_prompt(query, context, history)
        response = "LLM response pending"
        source_docs = self.retriever.retrieve_context(query, top_k=3)
        return response, source_docs


# ============= Initialization =============

def initialize_rag_pipeline(groq_client=None) -> RAGRetriever:
    """Initialize complete Tier 2 RAG pipeline"""
    embeddings = EmbeddingsManager(model_name="all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(persist_dir="./data/chroma_db")
    vector_store.create_collection("financial_docs")
    retriever = RAGRetriever(embeddings, vector_store, groq_client=groq_client)
    return retriever


# ============= Example Usage =============

if __name__ == "__main__":
    embeddings = EmbeddingsManager()
    vector_store = ChromaVectorStore()
    vector_store.create_collection()
    retriever = RAGRetriever(embeddings, vector_store)

    sample_text = """Tesla reported Q3 earnings of $25 billion with gross margin of 25.9%.
    Apple's iPhone sales decreased by 5% YoY in 2023.
    Microsoft's cloud revenue grew 30% driven by Azure adoption.
    NVDA reported record revenue of $18.1 billion, up 122% YoY driven by data center demand."""

    retriever.ingest_document(sample_text, "sample_doc_1",
                              {"source": "sample.pdf", "date": "2024-01-01"})

    query = "What was Tesla's gross margin?"
    context = retriever.get_context_string(query)
    print("Retrieved context:")
    print(context)