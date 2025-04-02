import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet as wn


def get_synonyms(word, lang="spa"):
    """Encuentra sinónimos en español para una palabra dada"""
    synonyms = set()
    print("Empezamos")
    for synset in wn.synsets(word, lang=lang):
        for lemma in synset.lemmas(lang):
            synonyms.add(lemma.name())
    return list(synonyms)


# # Buscar sinónimos de 'habitación'
print(get_synonyms("alberca"))

import fasttext.util
import numpy as np

# Cargar modelo en español (o multilingüe)
fasttext.util.download_model("es", if_exists="ignore")

ft = fasttext.load_model(
    "cc.es.300.bin"
)  # Download it from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz
# Buscar sinónimos de "habitaciones"
similar_words = ft.get_nearest_neighbors("habitaciones", k=10)
print(similar_words)


def cosine_similarity(word1, word2):
    vec1 = ft.get_word_vector(word1)
    vec2 = ft.get_word_vector(word2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


print("vamos")
print(cosine_similarity("habitaciones", "abitacion"))
