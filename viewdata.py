import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

# Încarcă date
data = torch.load("data.pt")
embeddings = data[0]
names = data[1]

# Convertește embeddings la NumPy
# embeddings = listă de tensori (1, 512)
embeddings_np = np.array([emb.squeeze().numpy() for emb in embeddings])

print(f"📊 Shape embeddings: {embeddings_np.shape}")
print(f"👤 Procesez {len(names)} embeddings...")

# Aplică t-SNE pentru reducere la 2D
print("🔄 Rulează t-SNE (poate dura 10-30 sec)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(names)-1))
embeddings_2d = tsne.fit_transform(embeddings_np)

# Creează hartă culori per persoană
unique_names = list(set(names))
color_map = {}
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_names)))
for i, name in enumerate(unique_names):
    color_map[name] = colors[i]

# Plot
plt.figure(figsize=(14, 10))

# Grupează punctele per persoană
person_points = defaultdict(list)
for i, name in enumerate(names):
    person_points[name].append(embeddings_2d[i])

# Desenează fiecare persoană
for name, points in person_points.items():
    points = np.array(points)
    plt.scatter(
        points[:, 0], 
        points[:, 1], 
        c=[color_map[name]], 
        label=name,
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

plt.title("Vizualizare Embeddings (t-SNE)", fontsize=16, fontweight='bold')
plt.xlabel("Dimensiune 1", fontsize=12)
plt.ylabel("Dimensiune 2", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvează figura
plt.savefig("embeddings_visualization.png", dpi=300, bbox_inches='tight')
print("✅ Salvat: embeddings_visualization.png")
plt.show()

# Afișează distanțe intra-persoană vs inter-persoană
print("\n📏 Analiza distanțelor:")
from scipy.spatial.distance import cosine

# Calculează distanță medie între embeddings-uri ale aceleiași persoane
intra_distances = []
for name in unique_names:
    indices = [i for i, n in enumerate(names) if n == name]
    if len(indices) > 1:
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                dist = cosine(embeddings_np[indices[i]], embeddings_np[indices[j]])
                intra_distances.append(dist)

# Calculează distanță între persoane diferite (sample)
inter_distances = []
sample_size = min(1000, len(embeddings) * 2)
for _ in range(sample_size):
    i, j = np.random.choice(len(names), 2, replace=False)
    if names[i] != names[j]:
        dist = cosine(embeddings_np[i], embeddings_np[j])
        inter_distances.append(dist)

if intra_distances:
    print(f"   Distanță medie INTRA-persoană: {np.mean(intra_distances):.4f}")
if inter_distances:
    print(f"   Distanță medie INTER-persoană: {np.mean(inter_distances):.4f}")
    print(f"   Ratio (mai mare = mai bine): {np.mean(inter_distances)/np.mean(intra_distances):.2f}x")
