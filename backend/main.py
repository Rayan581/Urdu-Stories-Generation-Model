"""
Urdu Story Generator API - FastAPI Backend
Phase IV: Microservice & Containerization

Endpoints:
    - GET  /           Health check
    - POST /generate   Generate story from prefix (standard response)
    - POST /generate/stream  Generate story with Server-Sent Events (streaming)
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model import TrigramModel
from tokenizer import EOS, EOP, EOT


# ─────────────────────────────────────────────────────────────────────────────
# Special Token Conversion
# ─────────────────────────────────────────────────────────────────────────────
def format_special_tokens(text: str) -> str:
    """
    Convert special tokens to readable format:
    - EOS (\u0003) → period + space (sentence end)
    - EOP (\u0004) → double newline (paragraph break)
    - EOT (\u0005) → [کہانی ختم] (end of story marker)
    """
    # Replace EOP first (paragraph break)
    text = text.replace(EOP, "\n\n")
    # Replace EOS (sentence end) - add period if not already there
    text = text.replace(EOS, "۔ ")
    # Replace EOT (end of story)
    text = text.replace(EOT, "\n\n[کہانی ختم]")
    # Clean up multiple spaces
    import re
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Paths configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.json")

# Global model instance
model: Optional[TrigramModel] = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan - Load model on startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the app starts."""
    global model
    print("[API] Loading Trigram model...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Please run evaluate.py first.")
    if not os.path.exists(TOKENIZER_PATH):
        raise RuntimeError(f"Tokenizer file not found: {TOKENIZER_PATH}")

    model = TrigramModel(tokenizer_path=TOKENIZER_PATH)
    model.load(MODEL_PATH)
    print(f"[API] Model loaded successfully: {model}")

    yield  # App runs here

    # Cleanup (if needed)
    print("[API] Shutting down...")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Urdu Story Generator API",
    description="A trigram language model API for generating Urdu stories",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    """Input schema for text generation."""
    prefix: str = Field(..., description="Starting phrase in Urdu")
    max_length: int = Field(default=100, ge=1, le=500,
                            description="Maximum tokens to generate")
    temperature: float = Field(
        default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=40, ge=0, le=100,
                       description="Top-k sampling (0 to disable)")
    top_p: float = Field(default=0.92, ge=0.0, le=1.0,
                         description="Nucleus sampling threshold")
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility")


class GenerateResponse(BaseModel):
    """Output schema for text generation."""
    input_prefix: str
    max_length: int
    generated_text: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "Urdu Story Generator API is running",
        "status": "healthy",
        "model_loaded": model is not None and model.trained,
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_trained": model.trained if model else False,
        "vocab_size": model.vocab_size if model else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_story(request: GenerateRequest):
    """
    Generate Urdu story text from a prefix.

    This endpoint returns the complete generated text in a single response.
    For streaming generation, use POST /generate/stream.
    """
    if model is None or not model.trained:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Custom generation loop: stop at EOT or max_length
        _BOS_ID = -1
        if request.seed is not None:
            import random
            random.seed(request.seed)

        prompt_ids = [t for t in model.tokenizer.encode(
            request.prefix) if t >= 0] if request.prefix else []
        if len(prompt_ids) >= 2:
            t_2, t_1 = prompt_ids[-2], prompt_ids[-1]
        elif len(prompt_ids) == 1:
            t_2, t_1 = _BOS_ID, prompt_ids[-1]
        else:
            t_2, t_1 = _BOS_ID, _BOS_ID

        generated = list(prompt_ids)
        for _ in range(request.max_length):
            candidates = model._candidate_tokens(t_2, t_1)
            tokens, probs = model._build_distribution(
                candidates, t_2, t_1,
                request.temperature,
                request.top_k,
                request.top_p,
            )
            next_id = model._weighted_sample(tokens, probs)
            generated.append(next_id)
            if next_id == model._eot_id:
                break
            t_2, t_1 = t_1, next_id

        generated_text = model.tokenizer.decode(generated)
        formatted_text = format_special_tokens(generated_text)
        return GenerateResponse(
            input_prefix=request.prefix,
            max_length=request.max_length,
            generated_text=formatted_text,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/stream")
async def generate_story_stream(request: GenerateRequest):
    """
    Generate Urdu story text with Server-Sent Events (SSE) streaming.

    Returns tokens one-by-one for a ChatGPT-like streaming experience.
    Each SSE message contains a token in the format: data: {"token": "..."}
    Final message: data: {"done": true}
    """
    if model is None or not model.trained:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def token_generator():
        """Generate tokens one at a time for streaming."""
        import random
        import json

        _BOS_ID = -1

        if request.seed is not None:
            random.seed(request.seed)

        # Encode prompt
        prompt_ids = [t for t in model.tokenizer.encode(
            request.prefix) if t >= 0] if request.prefix else []

        # Initialize context
        if len(prompt_ids) >= 2:
            t_2, t_1 = prompt_ids[-2], prompt_ids[-1]
        elif len(prompt_ids) == 1:
            t_2, t_1 = _BOS_ID, prompt_ids[-1]
        else:
            t_2, t_1 = _BOS_ID, _BOS_ID

        generated = list(prompt_ids)

        # Yield the prompt first
        if request.prefix:
            yield f"data: {json.dumps({'token': request.prefix}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming

        # Generate tokens one by one
        for _ in range(request.max_length):
            candidates = model._candidate_tokens(t_2, t_1)
            tokens, probs = model._build_distribution(
                candidates, t_2, t_1,
                request.temperature,
                request.top_k,
                request.top_p,
            )
            next_id = model._weighted_sample(tokens, probs)
            generated.append(next_id)

            # Decode just the new token and format special tokens
            token_text = model.tokenizer.decode([next_id])
            token_text = format_special_tokens(token_text)

            # Check for end of text
            if next_id == model._eot_id:
                yield f"data: {json.dumps({'token': token_text, 'done': True}, ensure_ascii=False)}\n\n"
                break

            # Yield the token
            yield f"data: {json.dumps({'token': token_text}, ensure_ascii=False)}\n\n"

            # Update context
            t_2, t_1 = t_1, next_id

            # Small delay for visual streaming effect
            await asyncio.sleep(0.02)

        # Final done signal
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
