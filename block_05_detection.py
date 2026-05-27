# ============================================================
# BLOCK 05: Disease and Aspect Detection Utilities
# ============================================================

from block_00_config import re, nlp
from block_02_embedding_models import (
    get_pubmedbert_embedding,
    get_biobert_embedding,
    get_sentence_transformer_embedding,
)
from block_04_kg_bootstrap import (
    diseases,
    disease_embeddings_cache,
    aspect_keywords,
    aspect_embeddings_cache,
    compute_average_similarity,
)

# Disease/Aspect Detection
def identify_diseases(user_query, threshold=0.90):
    """
    1) direct substring match with word boundaries -> 1.0
    2) otherwise semantic match against disease name embeddings, thresholded
    """
    user_query_lower = user_query.lower().strip()
    detected = []

    #  loop is properly indented now
    for disease in diseases:
        if re.search(r"\b" + re.escape(disease) + r"\b", user_query_lower):
            detected.append((disease, 1.0))

    if not detected:
        query_emb = {
            'pubmed': get_pubmedbert_embedding(user_query_lower),
            'biobert': get_biobert_embedding(user_query_lower),
            'st': get_sentence_transformer_embedding(user_query_lower)
        }
        for disease, emb in disease_embeddings_cache.items():
            avg_sim = compute_average_similarity(query_emb, emb)
            if avg_sim >= threshold:
                detected.append((disease, round(avg_sim, 3)))

    return detected


def detect_aspect(text):
    """
    Lemma/keyword pass, then embedding fallback.
    """
    text_lower = text.lower()
    doc = nlp(text_lower)
    query_lemmas = [t.lemma_ for t in doc]

    keyword_matches = {}
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if " " not in keyword:
                k_lemma = nlp(keyword)[0].lemma_
                count = query_lemmas.count(k_lemma)
                if count > 0:
                    keyword_matches[aspect] = keyword_matches.get(aspect, 0) + count
            else:
                if keyword in text_lower:
                    keyword_matches[aspect] = keyword_matches.get(aspect, 0) + text_lower.count(keyword)

    if keyword_matches:
        best_aspect_kw = max(keyword_matches, key=keyword_matches.get)
        return best_aspect_kw, 1.0

    # embedding fallback
    query_emb = {
        'pubmed': get_pubmedbert_embedding(text_lower),
        'biobert': get_biobert_embedding(text_lower),
        'st': get_sentence_transformer_embedding(text_lower)
    }
    scores = {}
    for aspect, emb in aspect_embeddings_cache.items():
        scores[aspect] = compute_average_similarity(query_emb, emb)

    if not scores:
        return None, 0.0

    sorted_aspects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_aspect, best_score = sorted_aspects[0]
    second_best = sorted_aspects[1][1] if len(sorted_aspects) > 1 else 0.0
    if best_score - second_best < 0.05 and "diagnos" in text_lower:
        return "diagnosis", 1.0
    return best_aspect, round(best_score, 3)

def compute_confidence_score(disease_confidence, aspect_confidence):
    return round((0.4 * disease_confidence + 0.6 * aspect_confidence) * 100, 2)
