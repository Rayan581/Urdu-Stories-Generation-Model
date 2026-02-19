from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI application
app = FastAPI(title="Urdu Story Generator API")

# Define the input structure required by the assignment
class GenerateRequest(BaseModel):
    prefix: str
    max_length: int = 50  # Default to 50 if the frontend doesn't provide a length

# A simple health check route (Good practice for deployments)
@app.get("/")
def read_root():
    return {"message": "Urdu Story API is running. Visit /docs to test it."}

# The main endpoint for Phase IV
@app.post("/generate")
def generate_story(request: GenerateRequest):
    # TODO: In Phase II & III, you will load your BPE Tokenizer and Trigram model here.
    # For now, we return a mock response to prove the API and frontend can talk to each other.
    
    mock_story = f"{request.prefix} ... [Model will generate the rest of the story here until <EOT>]"
    
    return {
        "input_prefix": request.prefix,
        "max_length": request.max_length,
        "generated_text": mock_story
    }