import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
from io import BytesIO
import os

# Définir le répertoire de cache pour Transformers
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"


# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration du modèle Stable Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "digiplay/majicMIX_realistic_v7", torch_dtype=dtype, low_cpu_mem_usage=True
)
pipe = pipe.to(device)
pipe.safety_checker = None
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Prompt par défaut
DEFAULT_PROMPT = (
    "Realistic greenery to the ground, including grass, small shrubs, trees, "
    "and flower beds along the sidewalks"
)
NEGATIVE_PROMPT = "ugly, deformed, disfigured, poor details"

# Endpoint de génération d'image
@app.post("/api/generate")
async def generate_image(file: UploadFile):
    # Charger l'image de départ
    init_image = Image.open(BytesIO(await file.read())).convert("RGB")

    # Paramètres de génération
    denoising_strength = 0.4
    steps = 25
    guidance = 7

    # Générer l'image
    images = pipe(
        prompt=DEFAULT_PROMPT,
        image=init_image,
        strength=denoising_strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=800,
        width=800,
        negative_prompt=NEGATIVE_PROMPT,
    ).images

    # Convertir l'image en flux binaire (sans la sauvegarder)
    buffer = BytesIO()
    images[0].save(buffer, format="PNG")
    buffer.seek(0)

    # Retourner l'image sous forme de réponse HTTP
    return StreamingResponse(buffer, media_type="image/png")

# Ajoute la gestion dynamique du port avec la variable d'environnement `PORT`
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
