# app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import openai  # o google.generativeai si prefieres Gemini
import os

app = FastAPI()

class Recipe(BaseModel):
    title: str
    ingredients_used: list
    missing_ingredients: list
    steps: list
    time: str
    difficulty: str
    waste_score: int  # 0-100

@app.post("/analyze-fridge")
async def analyze_fridge(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Usamos GPT-4o con visión (o Gemini)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                     "text": """Eres un chef experto. Analiza esta foto del interior de un refrigerador.
                     Devuélveme SOLO un JSON con esta estructura exacta:
                     {"ingredients": ["tomate", "huevos", "leche", "pollo", ...]} 
                     Incluye cantidad aproximada si es obvia (ej: "3 tomates", "media cebolla"). 
                     No inventes nada que no veas claramente."""},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{contents.base64()}"}
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    # Aquí obtienes la lista de ingredientes detectados
    ingredients = response.choices[0].message.content  # parsear JSON
    
    # Generar recetas
    recipe_response = openai.chat.completions.create(
        model="gpt-4o",  # aquí puedes poner "claude-3-5-sonnet" si usas Anthropic
        messages=[
            {"role": "system", "content": "Eres un chef creativo que odia desperdiciar comida."},
            {"role": "user", "content": f"""
            Tengo exactamente estos ingredientes en la nevera: {ingredients}
            Genera 3 recetas deliciosas que usen el mayor número posible de estos ingredientes.
            Prioriza usar TODO. 
            Devuélveme JSON con título, pasos, ingredientes usados y faltantes.
            """}
        ]
    )
    
    return recipe_response.choices[0].message.content
