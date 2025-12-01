from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
import base64
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.0-pro')
    print("Gemini configurado correctamente")
except Exception as e:
    print(f"Error configurando Gemini: {e}")

@app.get("/")
async def root():
    return {"message": "FridgeChef API funcionando"}

@app.post("/analyze-fridge")
async def analyze_fridge(file: UploadFile = File(...)):
    try:
        # Verificar que se subió un archivo
        if not file:
            raise HTTPException(status_code=400, detail="No se subió ningún archivo")
        
        # Leer el archivo
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        # Verificar formato de imagen
        try:
            image = Image.open(BytesIO(contents))
            image.verify()
            image = Image.open(BytesIO(contents))  # Re-abrir después de verify
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")
        
        # Convertir a base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prompt para detectar ingredientes
        prompt_ingredients = """Analiza esta foto del interior de un refrigerador.
        Devuelve SOLO un JSON con esta estructura exacta:
        {"ingredients": ["3 tomates", "huevos", "leche", "pollo", "cebolla", ...]}
        Incluye cantidad aproximada si es visible. No inventes ingredientes que no veas."""
        
        # Generar respuesta de ingredientes
        try:
            response = model.generate_content([prompt_ingredients, {"mime_type": "image/jpeg", "data": img_str}])
            ingredients_text = response.text.strip()
            
            # Extraer JSON de la respuesta
            if "```json" in ingredients_text:
                ingredients_json = ingredients_text.split("```json")[1].split("```")[0].strip()
            else:
                ingredients_json = ingredients_text
            
            ingredients = json.loads(ingredients_json)
            
        except Exception as e:
            print(f"Error con Gemini ingredientes: {e}")
            ingredients = {"ingredients": ["error en detección"]}
        
        # Prompt para generar recetas
        ingredients_str = ", ".join(ingredients["ingredients"])
        prompt_recipes = f"""Eres un chef experto que odia desperdiciar comida.
        Tengo estos ingredientes: {ingredients_str}
        
        Genera 3 recetas deliciosas que usen el MÁXIMO de estos ingredientes.
        Devuelve SOLO un JSON con esta estructura exacta:
        [
          {{
            "title": "Título de la receta",
            "ingredients_used": ["ingredientes que usa de la lista"],
            "missing_ingredients": ["pocos que falten"],
            "steps": ["Paso 1", "Paso 2", "Paso 3"],
            "time": "15 minutos",
            "difficulty": "fácil",
            "waste_score": 95
          }},
          // 2 más igual
        ]
        
        Prioriza usar TODOS los ingredientes disponibles."""
        
        try:
            recipe_response = model.generate_content(prompt_recipes)
            recipes_text = recipe_response.text.strip()
            
            if "```json" in recipes_text:
                recipes_json = recipes_text.split("```json")[1].split("```")[0].strip()
            else:
                recipes_json = recipes_text
            
            recipes = json.loads(recipes_json)
            
        except Exception as e:
            print(f"Error con Gemini recetas: {e}")
            recipes = [{"title": "Error temporal", "error": str(e)}]
        
        return {
            "ingredients_detected": ingredients,
            "recipes": recipes,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error general: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gemini_configured": genai.is_configured()}
