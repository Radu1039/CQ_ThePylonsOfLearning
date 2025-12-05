import os
import cv2
import torch
import numpy as np
import albumentations as A
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rulez pe: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Foldere
input_dirs = ["dataset/db/indoor_persons", "dataset/db/outdoor_persons"]
output_file = "data.pt"

# 2. AUGMENTĂRI ÎMBUNĂTĂȚITE (Random, nu preset)

# Ocluzie Random (înlocuiește masca + ochelarii)
# Patratte random colorate pe poziții aleatorii
aug_random_occlusion = A.CoarseDropout(
    max_holes=3,          # 1-3 blocuri
    max_height=60,        # Max 37% din înălțime
    max_width=60,
    min_holes=1,
    min_height=15,        # Min 9% din înălțime
    min_width=15,
    fill_value=None,      # CULORI RANDOM!!
    mask_fill_value=None,
    p=1.0,
)

# Ocluzie mare (simulează obiecte mai mari)
aug_large_occlusion = A.CoarseDropout(
    max_holes=2,
    max_height=80,
    max_width=80,
    min_holes=1,
    min_height=40,
    min_width=40,
    fill_value=None,
    mask_fill_value=None,
    p=1.0,
)

# Distanță/Blur (îmbunătățit cu resize)
aug_distance = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(5, 9), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
    ], p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
])

# Iluminare/Culoare (mai agresivă)
aug_lighting = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.ToGray(p=0.2),
])

# Perspective/Rotație (simulează unghiuri diferite)
aug_geometric = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    ),
])

# Compresie JPEG (simulează calitate joasă)
aug_quality = A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0)


def to_tensor(numpy_img):
    img = torch.tensor(numpy_img).float().permute(2, 0, 1) / 255.0
    img = (img - 0.5) / 0.5
    return img.unsqueeze(0).to(device)


embeddings_list = []
names_list = []

print("--- Procesare cu augmentări random îmbunătățite ---")

for folder in input_dirs:
    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):
        if not filename.endswith((".jpg")):
            continue

        person_name = os.path.splitext(filename)[0]
        filepath = os.path.join(folder, filename)

        img_full = cv2.imread(filepath)
        if img_full is None:
            continue
        img_full_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_full_rgb)

        boxes, _ = mtcnn.detect(img_pil)

        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]

            face_img = img_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((160, 160))
            face_np = np.array(face_img)

            # Lista de augmentări
            augmentations = [
                ("original", None),
                ("occlusion_small", aug_random_occlusion),
                ("occlusion_large", aug_large_occlusion),
                ("distance", aug_distance),
                ("lighting", aug_lighting),
                ("geometric", aug_geometric),
                ("quality", aug_quality),
            ]

            for aug_name, aug_fn in augmentations:
                if aug_fn is None:
                    face_augmented = face_np
                else:
                    face_augmented = aug_fn(image=face_np)["image"]
                
                emb = resnet(to_tensor(face_augmented)).detach().cpu()
                embeddings_list.append(emb)
                names_list.append(person_name)

            print(f"Procesat: {person_name} ({len(augmentations)} variante)")

# Salvare
if len(embeddings_list) > 0:
    torch.save([embeddings_list, names_list], output_file)
    print(f"\nSUCCES! Total vectori: {len(embeddings_list)}")
else:
    print("Eroare: Nu am gasit fete.")