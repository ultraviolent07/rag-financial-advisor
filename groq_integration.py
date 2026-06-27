"""
Groq Integration — Tier 3 upgrade
Adds 3-call decomposed chain on top of existing GroqLLMChain:
  Call 1 (fast model): classify regime + sentiment alignment
  Call 2 (balanced):   extract key claims from RAG docs
  Call 3 (balanced):   synthesise signal + chain-of-thought reasoning
Conflict flag → forced uncertainty hedge in output.
"""

from groq import Groq
from typing import List, Dict, Optional
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()


# ============= Groq Client Wrapper (unchanged) =============

class GroqLLMChain:
    """Drop-in Groq LLM wrapper — free, fast inference"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Get free key from https://console.groq.com")

        self.client = Groq(api_key=self.api_key)
        self.models = {
            "fast":     "llama-3.1-8b-instant",
            "balanced": "llama-3.3-70b-versatile",
            "quality":  "llama-3.3-70b-versatile",
        }
        self.model = self.models["balanced"]
        self.rate_limit_calls = 0
        self.rate_limit_start = time.time()

    def set_model(self, model_type: str = "balanced"):
        if model_type in self.models:
            self.model = self.models[model_type]

    def _check_rate_limit(self):
        current_time = time.time()
        if current_time - self.rate_limit_start >= 60:
            self.rate_limit_calls = 0
            self.rate_limit_start = current_time
        self.rate_limit_calls += 1
        if self.rate_limit_calls > 490:
            wait = 60 - (current_time - self.rate_limit_start)
            print(f"Rate limit approaching. Wait {wait:.1f}s")
            return False
        return True

    def generate(self,
                 prompt: str,
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 system_prompt: Optional[str] = None,
                 model_type: str = "balanced") -> str:
        if not self._check_rate_limit():
            return "Rate limit exceeded. Please try again."
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.models.get(model_type, self.model),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return f"Error: {str(e)}"

    def generate_with_history(self,
                              query: str,
                              conversation_history: List[Dict],
                              system_prompt: Optional[str] = None,
                              max_tokens: int = 1000) -> str:
        if not self._check_rate_limit():
            return "Rate limit exceeded."
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in conversation_history[-10:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            messages.append({"role": "user", "content": query})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_streaming(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000):
        if not self._check_rate_limit():
            yield "Rate limit exceeded."
            return
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"


# ============= Tier 3: 3-Call Decomposed Chain =============

class DecomposedAnalysisChain:
    """
    3-call Groq chain for deep research queries.
    Call 1 (fast):     classify regime alignment + sentiment
    Call 2 (balanced): extract key claims from RAG docs
    Call 3 (balanced): synthesise into signal + chain-of-thought
    Conflict detection → uncertainty hedge injected into final output.
    """

    def __init__(self, llm: GroqLLMChain):
        self.llm = llm

    # ---- Call 1: Classify ----
    def call1_classify(
        self,
        query: str,
        regime_context: str,
        sentiment_summary: str
    ) -> Dict:
        """
        Fast model — classify whether regime + sentiment align.
        Returns JSON: {alignment, signal, conflict, reasoning}
        """
        prompt = f"""You are a financial signal classifier. Respond ONLY in valid JSON.

Query: {query}
Market regime: {regime_context}
Sentiment summary: {sentiment_summary}

Classify and return exactly this JSON structure:
{{
  "alignment": "aligned" | "conflicted" | "neutral",
  "signal": "bullish" | "bearish" | "neutral",
  "conflict": true | false,
  "conflict_reason": "brief reason if conflict else null",
  "reasoning": "1-2 sentence explanation"
}}"""

        raw = self.llm.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2,
            model_type="fast"
        )

        try:
            clean = raw.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean.strip())
        except Exception:
            return {
                "alignment": "neutral",
                "signal": "neutral",
                "conflict": False,
                "conflict_reason": None,
                "reasoning": raw[:200]
            }

    # ---- Call 2: Extract Claims ----
    def call2_extract_claims(
        self,
        query: str,
        rag_context: str
    ) -> Dict:
        """
        Balanced model — extract key factual claims from RAG docs.
        Returns JSON: {claims, data_points, gaps}
        """
        prompt = f"""You are a financial document analyst. Extract key facts from the provided context.
Respond ONLY in valid JSON.

Query: {query}

Document context:
{rag_context[:3000]}

Return exactly this JSON structure:
{{
  "claims": ["fact 1 with specific number/date", "fact 2", "fact 3"],
  "data_points": {{
    "revenues": "if found, else null",
    "margins": "if found, else null",
    "growth": "if found, else null",
    "eps": "if found, else null"
  }},
  "gaps": ["what data is missing that would help answer the query"],
  "source_quality": "strong" | "partial" | "weak"
}}"""

        raw = self.llm.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.1,
            model_type="balanced"
        )

        try:
            clean = raw.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean.strip())
        except Exception:
            return {
                "claims": [],
                "data_points": {},
                "gaps": ["Could not parse document claims"],
                "source_quality": "weak"
            }

    # ---- Call 3: Synthesise ----
    def call3_synthesise(
        self,
        query: str,
        classification: Dict,
        claims: Dict,
        technicals_summary: str,
        regime_context: str,
        conflict: bool
    ) -> str:
        """
        Balanced model — final synthesis with chain-of-thought.
        Injects uncertainty hedge if conflict detected.
        """
        conflict_instruction = ""
        if conflict:
            conflict_instruction = (
                f"\n⚠️  CONFLICT DETECTED: {classification.get('conflict_reason', 'signals disagree')}. "
                "You MUST include an uncertainty hedge in your response. "
                "Explicitly state the conflicting signals and why the outlook is uncertain."
            )

        claims_str = "\n".join(f"- {c}" for c in claims.get("claims", []))
        gaps_str   = "\n".join(f"- {g}" for g in claims.get("gaps", []))

        prompt = f"""You are a senior financial analyst. Provide a thorough investment analysis with chain-of-thought reasoning.
{conflict_instruction}

QUERY: {query}

MARKET REGIME: {regime_context}

SIGNAL CLASSIFICATION:
- Alignment: {classification.get('alignment')}
- Signal: {classification.get('signal')}
- Reasoning: {classification.get('reasoning')}

KEY FACTS FROM DOCUMENTS:
{claims_str if claims_str else "No document facts extracted."}

DATA GAPS:
{gaps_str if gaps_str else "None identified."}

TECHNICAL SUMMARY:
{technicals_summary}

INSTRUCTIONS:
1. Walk through your reasoning step by step (chain-of-thought)
2. Cite specific numbers from the document facts above
3. Factor in the market regime when assessing risk
4. If conflict detected, explicitly address the uncertainty
5. End with a clear signal: BULLISH / BEARISH / NEUTRAL with confidence level
6. Keep response to 4-5 paragraphs

ANALYSIS:"""

        return self.llm.generate(
            prompt=prompt,
            max_tokens=1200,
            temperature=0.4,
            model_type="balanced"
        )

    def run(
        self,
        query: str,
        rag_context: str,
        regime_context: str = "",
        sentiment_summary: str = "",
        technicals_summary: str = "",
        simple_query: bool = False
    ) -> Dict:
        """
        Adaptive depth routing:
        - simple_query=True  → skip Call 2, use fast path
        - simple_query=False → all 3 calls + full trace
        """
        if simple_query:
            # Fast path — single call
            response = self.llm.generate(
                prompt=f"""Answer this financial question using the context below.
Context: {rag_context[:2000]}
Question: {query}
Answer concisely in 2-3 sentences.""",
                max_tokens=400,
                temperature=0.5,
                model_type="fast"
            )
            return {
                "response": response,
                "signal": "neutral",
                "conflict": False,
                "chain": "fast_path",
                "calls_made": 1
            }

        # Full 3-call chain
        classification = self.call1_classify(query, regime_context, sentiment_summary)
        claims         = self.call2_extract_claims(query, rag_context)
        synthesis      = self.call3_synthesise(
            query, classification, claims,
            technicals_summary, regime_context,
            conflict=classification.get("conflict", False)
        )

        return {
            "response":       synthesis,
            "signal":         classification.get("signal", "neutral"),
            "alignment":      classification.get("alignment", "neutral"),
            "conflict":       classification.get("conflict", False),
            "conflict_reason":classification.get("conflict_reason"),
            "claims":         claims.get("claims", []),
            "data_points":    claims.get("data_points", {}),
            "source_quality": claims.get("source_quality", "unknown"),
            "chain":          "full_3_call",
            "calls_made":     3
        }


# ============= RAG + Groq Chain (updated) =============

class GroqRAGChain:
    """RAG pipeline using Groq — supports both simple and deep research modes"""

    def __init__(self, retriever, groq_llm: Optional[GroqLLMChain] = None):
        self.retriever = retriever
        self.llm = groq_llm or GroqLLMChain()
        self.decomposed = DecomposedAnalysisChain(self.llm)

    def _is_simple_query(self, query: str) -> bool:
        """Heuristic: short factual queries go fast path"""
        simple_keywords = ["what is", "define", "how much", "when did", "price of", "what was"]
        q = query.lower()
        return any(kw in q for kw in simple_keywords) and len(query.split()) < 10

    def build_rag_prompt(self, query: str, context: str, history: List[Dict] = None) -> str:
        history_str = ""
        if history:
            history_str = "PREVIOUS CONVERSATION:\n"
            for msg in history[-4:]:
                role    = msg.get("role", "user").upper()
                content = msg.get("content", "")[:150]
                history_str += f"{role}: {content}\n"

        return f"""You are a financial analyst AI. Use the provided context to answer questions accurately.

{history_str}

CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based on the provided context first
2. If not in documents, answer from general financial knowledge and say so
3. Be specific with numbers, dates, percentages
4. Explain your reasoning step by step
5. Keep response concise but complete (2-3 paragraphs)

ANSWER:"""

    def answer_question(
        self,
        query: str,
        history: List[Dict] = None,
        regime_context: str = "",
        sentiment_summary: str = "",
        technicals_summary: str = "",
        deep_research: bool = False
    ) -> Dict:
        """
        Unified answer method.
        deep_research=True  → 3-call decomposed chain
        deep_research=False → adaptive (simple fast-path, complex single-call)
        """
        context = self.retriever.get_context_string(query, top_k=5)
        sources = self.retriever.retrieve_context(query, top_k=3)

        if deep_research:
            result = self.decomposed.run(
                query=query,
                rag_context=context,
                regime_context=regime_context,
                sentiment_summary=sentiment_summary,
                technicals_summary=technicals_summary,
                simple_query=False
            )
            return {
                "answer":         result["response"],
                "sources":        sources,
                "signal":         result.get("signal", "neutral"),
                "conflict":       result.get("conflict", False),
                "conflict_reason":result.get("conflict_reason"),
                "claims":         result.get("claims", []),
                "calls_made":     result.get("calls_made", 3),
                "confidence":     0.90 if not result.get("conflict") else 0.60
            }

        # Standard path
        simple = self._is_simple_query(query)
        if simple:
            result = self.decomposed.run(
                query=query, rag_context=context,
                simple_query=True
            )
            return {
                "answer": result["response"],
                "sources": sources,
                "signal": "neutral",
                "confidence": 0.80,
                "calls_made": 1
            }

        # Single balanced call with full context
        prompt = self.build_rag_prompt(query, context, history)
        answer = self.llm.generate(prompt=prompt, max_tokens=1000, temperature=0.5)
        return {
            "answer": answer,
            "sources": sources,
            "signal": "neutral",
            "confidence": 0.85,
            "calls_made": 1
        }

    def answer_with_streaming(self, query: str, history: List[Dict] = None):
        context = self.retriever.get_context_string(query, top_k=5)
        prompt  = self.build_rag_prompt(query, context, history)
        for token in self.llm.generate_streaming(prompt, max_tokens=1000):
            yield token


# ============= FastAPI Integration =============

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[dict] = []
    deep_research: bool = False
    ticker: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float
    signal: Optional[str] = None
    conflict: Optional[bool] = None
    calls_made: Optional[int] = None

router = APIRouter()
groq_llm   = None
rag_chain  = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not groq_llm or not rag_chain:
            raise HTTPException(status_code=500, detail="RAG system not initialized")

        result = rag_chain.answer_question(
            query=request.query,
            history=request.conversation_history,
            deep_research=request.deep_research
        )

        return ChatResponse(
            response=result["answer"],
            sources=result["sources"][:3],
            confidence=result.get("confidence", 0.85),
            signal=result.get("signal"),
            conflict=result.get("conflict"),
            calls_made=result.get("calls_made", 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            for token in rag_chain.answer_with_streaming(
                query=request.query,
                history=request.conversation_history
            ):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============= Example Usage =============

if __name__ == "__main__":
    llm = GroqLLMChain()

    print("Testing 3-call decomposed chain...\n")
    chain = DecomposedAnalysisChain(llm)

    sample_context = """Apple Q2 2025: Total net sales $95.4B (up 5% YoY).
    iPhone revenue $46.8B. Services $26.6B (up 11.6% YoY). 
    Gross margin 47.1%. EPS $1.65."""

    result = chain.run(
        query="Should I buy Apple stock?",
        rag_context=sample_context,
        regime_context="MARKET REGIME [AAPL]: TREND — directional momentum, RSI: 58",
        sentiment_summary="Positive: analyst consensus bullish (buy_pct: 0.72), news sentiment positive",
        technicals_summary="Price above MA50, RSI 58, positive momentum",
        simple_query=False
    )

    print(f"Signal:     {result['signal']}")
    print(f"Conflict:   {result['conflict']}")
    print(f"Calls made: {result['calls_made']}")
    print(f"Claims:     {result['claims']}")
    print(f"\nResponse:\n{result['response'][:500]}...")
