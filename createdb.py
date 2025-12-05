import os
import cv2
import torch
import numpy as np
import albumentations as A
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# 🎨 20 AUGMENTĂRI (Import din fișierul anterior)
# ═══════════════════════════════════════════════════════════════

# Ocluzie
aug_1_small_occlusion = A.CoarseDropout(max_holes=2, max_height=25, max_width=25, min_holes=1, min_height=10, min_width=10, fill_value=None, p=1.0)
aug_2_medium_occlusion = A.CoarseDropout(max_holes=2, max_height=50, max_width=50, min_holes=1, min_height=30, min_width=30, fill_value=None, p=1.0)
aug_3_large_occlusion = A.CoarseDropout(max_holes=1, max_height=90, max_width=120, min_holes=1, min_height=60, min_width=80, fill_value=None, p=1.0)
aug_4_multi_occlusion = A.CoarseDropout(max_holes=5, max_height=30, max_width=30, min_holes=3, min_height=15, min_width=15, fill_value=None, p=1.0)
aug_5_horizontal_bar = A.CoarseDropout(max_holes=1, max_height=25, max_width=140, min_holes=1, min_height=15, min_width=100, fill_value=None, p=1.0)
aug_6_vertical_bar = A.CoarseDropout(max_holes=1, max_height=140, max_width=40, min_holes=1, min_height=100, min_width=20, fill_value=None, p=1.0)

# Blur & Distanță
aug_7_light_blur = A.Compose([A.GaussianBlur(blur_limit=(3, 5), p=1.0), A.CoarseDropout(max_holes=1, max_height=20, max_width=20, min_holes=1, min_height=10, min_width=10, fill_value=None, p=0.3)])
aug_8_heavy_blur = A.Compose([A.GaussianBlur(blur_limit=(9, 15), p=1.0), A.CoarseDropout(max_holes=2, max_height=35, max_width=35, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.5)])
aug_9_motion_blur = A.Compose([A.MotionBlur(blur_limit=(7, 15), p=1.0), A.CoarseDropout(max_holes=2, max_height=30, max_width=30, min_holes=1, min_height=15, min_width=15, fill_value=None, p=0.4)])
aug_10_pixelation = A.Compose([A.Downscale(scale_min=0.25, scale_max=0.5, p=1.0), A.CoarseDropout(max_holes=2, max_height=40, max_width=40, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.5)])

# Iluminare
aug_11_bright_light = A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.3, 0.5), contrast_limit=0.2, p=1.0), A.CoarseDropout(max_holes=2, max_height=30, max_width=30, min_holes=1, min_height=15, min_width=15, fill_value=None, p=0.4)])
aug_12_low_light = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=(-0.3, -0.1), p=1.0), A.GaussNoise(var_limit=(30.0, 60.0), p=0.7), A.CoarseDropout(max_holes=2, max_height=35, max_width=35, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.5)])
aug_13_high_contrast = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(0.4, 0.7), p=1.0), A.CoarseDropout(max_holes=1, max_height=50, max_width=50, min_holes=1, min_height=30, min_width=30, fill_value=None, p=0.5)])
aug_14_color_shift = A.Compose([A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=20, p=1.0), A.CoarseDropout(max_holes=2, max_height=30, max_width=30, min_holes=1, min_height=15, min_width=15, fill_value=None, p=0.4)])
aug_15_grayscale = A.Compose([A.ToGray(p=1.0), A.CoarseDropout(max_holes=2, max_height=35, max_width=35, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.5)])

# Geometrie
aug_16_rotation = A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), A.CoarseDropout(max_holes=2, max_height=40, max_width=40, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.6)])
aug_17_perspective = A.Compose([A.Perspective(scale=(0.05, 0.15), p=1.0), A.CoarseDropout(max_holes=2, max_height=35, max_width=35, min_holes=1, min_height=20, min_width=20, fill_value=None, p=0.5)])

# Calitate
aug_18_low_quality = A.Compose([A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0), A.CoarseDropout(max_holes=3, max_height=30, max_width=30, min_holes=1, min_height=15, min_width=15, fill_value=None, p=0.6)])
aug_19_noisy = A.Compose([A.GaussNoise(var_limit=(40.0, 80.0), p=1.0), A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.3, 0.7), p=0.5), A.CoarseDropout(max_holes=3, max_height=35, max_width=35, min_holes=2, min_height=20, min_width=20, fill_value=None, p=0.7)])
aug_20_extreme_combo = A.Compose([A.OneOf([A.GaussianBlur(blur_limit=(7, 11), p=1.0), A.MotionBlur(blur_limit=9, p=1.0)], p=0.8), A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8), A.GaussNoise(var_limit=(20.0, 50.0), p=0.5), A.ImageCompression(quality_lower=25, quality_upper=50, p=0.6), A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5), A.CoarseDropout(max_holes=4, max_height=45, max_width=45, min_holes=2, min_height=20, min_width=20, fill_value=None, p=0.9)])

ALL_AUGMENTATIONS = [
    ("original", None),
    ("small_occlusion", aug_1_small_occlusion),
    ("medium_occlusion", aug_2_medium_occlusion),
    ("large_occlusion", aug_3_large_occlusion),
    ("multi_occlusion", aug_4_multi_occlusion),
    ("horizontal_bar", aug_5_horizontal_bar),
    ("vertical_bar", aug_6_vertical_bar),
    ("light_blur", aug_7_light_blur),
    ("heavy_blur", aug_8_heavy_blur),
    ("motion_blur", aug_9_motion_blur),
    ("pixelation", aug_10_pixelation),
    ("bright_light", aug_11_bright_light),
    ("low_light", aug_12_low_light),
    ("high_contrast", aug_13_high_contrast),
    ("color_shift", aug_14_color_shift),
    ("grayscale", aug_15_grayscale),
    ("rotation", aug_16_rotation),
    ("perspective", aug_17_perspective),
    ("low_quality", aug_18_low_quality),
    ("noisy", aug_19_noisy),
    ("extreme_combo", aug_20_extreme_combo),
]

# ═══════════════════════════════════════════════════════════════
# 🚀 PROCESARE
# ═══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

input_dirs = ["dataset/db/indoor_persons", "dataset/db/outdoor_persons"]
output_file = "data.pt"

def to_tensor(numpy_img):
    img = torch.tensor(numpy_img).float().permute(2, 0, 1) / 255.0
    img = (img - 0.5) / 0.5
    return img.unsqueeze(0).to(device)

embeddings_list = []
names_list = []

print(f"\n🎨 Procesez cu {len(ALL_AUGMENTATIONS)} augmentări")
print("=" * 60)

for folder in input_dirs:
    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):
        if not filename.endswith((".jpg", ".jpeg", ".png")):
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

            # Aplică fiecare augmentare
            for aug_name, aug_fn in ALL_AUGMENTATIONS:
                if aug_fn is None:
                    face_augmented = face_np
                else:
                    face_augmented = aug_fn(image=face_np)["image"]
                
                emb = resnet(to_tensor(face_augmented)).detach().cpu()
                embeddings_list.append(emb)
                names_list.append(person_name)

            print(f"✅ {person_name:20s} → {len(ALL_AUGMENTATIONS)} variante")

# Salvare
if len(embeddings_list) > 0:
    torch.save([embeddings_list, names_list], output_file)
    print("\n" + "=" * 60)
    print(f"🎉 SUCCES! Salvat {len(embeddings_list)} embeddings în {output_file}")
    print(f"📊 {len(set(names_list))} persoane unice")
    print(f"📈 {len(embeddings_list) // len(set(names_list))} variante/persoană")
    print("=" * 60)
else:
    print("❌ Eroare: Nu am găsit fețe")