import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

# Distance Poincaré
def poincare_distance(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    diff_norm = np.linalg.norm(u - v)
    
    # Projeter dans la boule unité si besoin
    if norm_u >= 1 or norm_v >= 1:
        u = u / (norm_u + 1e-5)
        v = v / (norm_v + 1e-5)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        diff_norm = np.linalg.norm(u - v)
    
    numerator = 2 * (diff_norm ** 2)
    denominator = (1 - norm_u ** 2) * (1 - norm_v ** 2)
    arg = 1 + numerator / denominator
    
    dist = np.log(arg + np.sqrt(arg**2 - 1))
    return dist

# Charger dataset
df = pd.read_csv("mon_dataset.csv")
print("Extrait dataset :")
print(df.head())

# Charger GloVe (avec no_header=True)
glove_file = "glove.6B.100d.txt"
model = KeyedVectors.load_word2vec_format(glove_file, no_header=True)

embeddings_1 = []
embeddings_2 = []
distances = []

for _, row in df.iterrows():
    w1 = str(row['word1']).lower()
    w2 = str(row['word2']).lower()
    if w1 in model and w2 in model:
        vec1 = model[w1]
        vec2 = model[w2]
        dist = poincare_distance(vec1, vec2)
    else:
        vec1 = np.zeros(100)
        vec2 = np.zeros(100)
        dist = None
    
    embeddings_1.append(vec1.tolist())
    embeddings_2.append(vec2.tolist())
    distances.append(dist)

df["embedding_word1"] = embeddings_1
df["embedding_word2"] = embeddings_2
df["poincare_distance"] = distances

print(df[["word1", "word2", "is_hypernym", "poincare_distance", "embedding_word1", "embedding_word2"]])

df.to_csv("resultats_embeddings_poincare.csv", index=False)
print("Résultats enregistrés dans resultats_embeddings_poincare.csv")
