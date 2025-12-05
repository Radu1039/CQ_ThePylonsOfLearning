import os
import cv2
import torch
import numpy as np
import albumentations as A
from facenet_pytorch import MTCNN
from PIL import Image
import shutil

# ═══════════════════════════════════════════════════════════════
# ⚙️ CONFIGURARE
# ═══════════════════════════════════════════════════════════════

# 👇 PUNE AICI FOLDERUL CU POZELE PE CARE VREI SĂ LE VERIFICI
INPUT_FOLDER = "dataset/db/outdoor_persons" 

# Unde salvăm rezultatele vizuale
OUTPUT_ROOT = "debug_output_batch"

# Curățăm folderul de output vechi
if os.path.exists(OUTPUT_ROOT): shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inițializare Detector (Aceleași setări ca la generarea DB)
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    min_face_size=20,
    thresholds=[0.4, 0.5, 0.5],
    factor=0.709,
)

# ═══════════════════════════════════════════════════════════════
# 🎨 LISTA DE AUGMENTĂRI (IDENTICĂ CU CEA DIN DB)
# ═══════════════════════════════════════════════════════════════

aug_mask = A.CoarseDropout(max_holes=1, max_height=80, max_width=160, min_holes=1, min_height=60, min_width=100, fill_value=0, p=1.0)
aug_hat = A.CoarseDropout(max_holes=1, max_height=60, max_width=160, min_holes=1, min_height=40, min_width=100, fill_value=0, p=1.0)
aug_blur = A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0), A.Downscale(scale_min=0.25, scale_max=0.5, p=1.0)])
aug_bright = A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.3, 0.5), contrast_limit=0.2, p=1.0), A.CoarseDropout(max_holes=2, max_height=30, max_width=30, p=0.4)])
aug_low = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=(-0.3, -0.1), p=1.0), A.GaussNoise(var_limit=(30.0, 60.0), p=0.7)])
aug_noise = A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
aug_extreme = A.Compose([
    A.OneOf([A.GaussianBlur(blur_limit=(7, 11), p=1.0), A.MotionBlur(blur_limit=9, p=1.0)], p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.GaussNoise(var_limit=(20.0, 50.0), p=0.5),
    A.CoarseDropout(max_holes=4, max_height=45, max_width=45, p=0.9)
])

# Lista pentru vizualizare
AUGMENTATIONS = [
    ("Original", None),
    ("Masca", aug_mask),
    ("Caciula", aug_hat),
    ("Blur_Distanta", aug_blur),
    ("Lumina_Tare", aug_bright),
    ("Intuneric", aug_low),
    ("Zgomot", aug_noise),
    ("EXTREM", aug_extreme)
]

# ═══════════════════════════════════════════════════════════════
# 🛠️ FUNCȚIA DE CROP CU ZOOM (EXACT CA ÎN SCRIPTUL DE BAZĂ DE DATE)
# ═══════════════════════════════════════════════════════════════

def get_face_debug(img_path):
    img = cv2.imread(img_path)
    if img is None: return None

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Pas 1: Încercăm detecție normală
    boxes, _ = mtcnn.detect(img_pil)

    zoomed_msg = ""
    
    # Pas 2: Logica de Zoom dacă nu găsește sau poza e mică
    if boxes is None and w < 2000:
        zoomed_msg = "(ZOOM 3X APLICAT)"
        scale_factor = 3.0
        
        img_zoomed = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]) # Sharpen
        img_zoomed = cv2.filter2D(img_zoomed, -1, kernel)
        
        img_rgb_z = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2RGB)
        img_pil_z = Image.fromarray(img_rgb_z)
        
        boxes, _ = mtcnn.detect(img_pil_z)
        img_pil = img_pil_z # Înlocuim cu varianta mărită

    if boxes is None: return None

    # Luăm cea mai mare față
    best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
    x1, y1, x2, y2 = [int(b) for b in best_box]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)

    face_img = img_pil.crop((x1, y1, x2, y2))
    face_img = face_img.resize((160, 160)) 
    
    return np.array(face_img), zoomed_msg

# ═══════════════════════════════════════════════════════════════
# 🚀 RULARE BATCH
# ═══════════════════════════════════════════════════════════════

if not os.path.exists(INPUT_FOLDER):
    print(f"❌ Eroare: Folderul {INPUT_FOLDER} nu există!")
    exit()

files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('jpg', 'png', 'jpeg'))])
print(f"🚀 Încep vizualizarea pentru {len(files)} imagini din {INPUT_FOLDER}...")

for filename in files:
    file_path = os.path.join(INPUT_FOLDER, filename)
    base_name = os.path.splitext(filename)[0]
    
    # 1. Obținem fața
    result = get_face_debug(file_path)
    
    if result is None:
        print(f"❌ {filename}: Față nedetectată.")
        continue
        
    face_np, msg = result
    print(f"✅ {filename}: Față găsită {msg}")
    
    # 2. Creăm folder pentru această imagine
    current_out_dir = os.path.join(OUTPUT_ROOT, base_name)
    os.makedirs(current_out_dir, exist_ok=True)
    
    # 3. Generăm și salvăm toate variantele
    for aug_name, aug_fn in AUGMENTATIONS:
        save_name = f"{aug_name}.jpg"
        save_path = os.path.join(current_out_dir, save_name)
        
        if aug_fn is None:
            # Originalul
            img_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
        else:
            # Augmentarea
            res = aug_fn(image=face_np)
            aug_face = res['image']
            img_bgr = cv2.cvtColor(aug_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)

print(f"\n🎉 GATA! Verifică folderul: {OUTPUT_ROOT}")