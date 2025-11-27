from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Recipe(BaseModel):
    title: str
    ingredients_used: list
    missing_ingredients: list
    steps: list
    time: str
    difficulty: str
    waste_score: int

@app.post("/analyze-fridge")
async def analyze_fridge(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    # Convert to base64 for Gemini
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Detectar ingredientes
    prompt_ingredients = """Eres un chef experto. Analiza esta foto del interior de un refrigerador.
    Devuélveme SOLO un JSON con esta estructura exacta:
    {"ingredients": ["3 tomates", "huevos", "leche", "pollo", ...]} 
    Incluye cantidad aproximada si es obvia. No inventes nada."""
    
    response = model.generate_content(
        [prompt_ingredients, {"mime_type": "image/jpeg", "data": img_str}]
    )
    
    ingredients_json = response.text.strip('```json\n').strip('\n```')
    
    # Generar recetas
    prompt_recetas = f"""Eres un chef creativo que odia desperdiciar comida.
    Tengo exactamente estos ingredientes: {ingredients_json}
    Genera 3 recetas deliciosas que usen el mayor número posible de estos ingredientes.
    Prioriza usar TODO. 
    Devuélveme JSON con esta estructura: [{{"title": "Título", "ingredients_used": [], "missing_ingredients": [], "steps": [], "time": "30 min", "difficulty": "fácil", "waste_score": 95}}]"""

    recipe_response = model.generate_content(prompt_recetas)
    
    return recipe_response.text.strip('```json\n').strip('\n```')
