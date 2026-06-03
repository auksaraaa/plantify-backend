import os
import json
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

app = FastAPI(title="Hierarchical Plant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO_ID = os.getenv("HF_REPO_ID", "Teerakorn/Efficientnetv2splantclassification")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "my_models") # เปลี่ยนชื่อแฟ้มแคชเป็น my_models 
CONFIDENCE_THRESHOLD = 0.5

part_model = None
species_models = {}
part_idx2name = {}
species_idx2name = {}


def load_models():
    global part_model, species_models, part_idx2name, species_idx2name
    if part_model is not None: 
        return

    print(f"Downloading/Syncing models from Hugging Face... ⌛")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    snapshot_download(repo_id=HF_REPO_ID, local_dir=MODEL_CACHE_DIR)
    
    mapping_path = os.path.join(MODEL_CACHE_DIR, "class_mappings.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        class_info = json.load(f)
        
    part_idx2name = {str(k): str(v) for k, v in class_info["part_classes"].items()}
    species_idx2name = class_info["species_classes"]
    parts_list = class_info.get("parts", ["bark", "flower", "fruit", "leaf"])

    print("Loading Keras Models... 🧠")
    
    # +++ ส่วนที่แก้ใหม่ เพื่อให้ Keras เก่าข้ามพารามิเตอร์ของ Keras 3 +++
    class SafeDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None) # โยน parameter เจ้าปัญหาทิ้ง
            super().__init__(*args, **kwargs)
            
    custom_objects = {"Dense": SafeDense}
    
    part_model = tf.keras.models.load_model(
        os.path.join(MODEL_CACHE_DIR, "part_model.keras"),
        custom_objects=custom_objects
    )
    
    for part in parts_list:
        sp_path = os.path.join(MODEL_CACHE_DIR, f"species_{part}.keras")
        if os.path.exists(sp_path):
            species_models[part] = tf.keras.models.load_model(
                sp_path, 
                custom_objects=custom_objects
            )
    print("✅ All Models loaded successfully!")


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/")
def read_root():
    return {"message": "Plant Classifier API is running", "status": "Ready"}


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")
        
    if part_model is None:
        load_models()
    
    try:
        image_bytes = await file.read()
        # ทำให้อยู่ในขนาด 384x384 ตามแบบ EfficientNetV2S
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((384, 384))
        
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0) # shape (1, 384, 384, 3)

        # --- 1. ทำนายส่วนพืช (Part) ---
        part_pred = part_model.predict(arr, verbose=0)[0]
        part_idx = int(np.argmax(part_pred))
        best_part_name = part_idx2name.get(str(part_idx), str(part_idx))
        part_conf = float(part_pred[part_idx])
        
        part_probs_dict = {part_idx2name.get(str(i), str(i)): float(p) for i, p in enumerate(part_pred)}

        # --- 2. ทำนายสายพันธุ์ (Species) ตาม Part ---
        if best_part_name in species_models:
            sp_model = species_models[best_part_name]
            sp_pred = sp_model.predict(arr, verbose=0)[0]
            sp_idx = int(np.argmax(sp_pred))
            
            best_sp_name = species_idx2name[best_part_name].get(str(sp_idx), str(sp_idx))
            sp_conf = float(sp_pred[sp_idx])
            
            sp_probs_dict = {species_idx2name[best_part_name].get(str(i), str(i)): float(p) for i, p in enumerate(sp_pred)}
        else:
            best_sp_name = "N/A"
            sp_conf = 0.0
            sp_probs_dict = {}

        low_confidence = bool((part_conf < CONFIDENCE_THRESHOLD) or (sp_conf < CONFIDENCE_THRESHOLD))

        return {
            "part": best_part_name,
            "part_conf": part_conf,
            "part_probs": part_probs_dict,
            "species": best_sp_name,
            "species_conf": sp_conf,
            "species_probs": sp_probs_dict,
            "low_confidence": low_confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))