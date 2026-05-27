# ============================================================
# BLOCK 04: Disease and Aspect Bootstrapping from Neo4j KG
# ============================================================

from block_00_config import cosine_similarity
from block_01_neo4j_setup import neo4j_driver
from block_02_embedding_models import (
    get_pubmedbert_embedding,
    get_biobert_embedding,
    get_sentence_transformer_embedding,
)

#  Disease & Aspect Bootstrapping (from Neo4j)
def get_all_diseases():
    query = "MATCH (d:Disease) RETURN toLower(d.name) AS disease"
    with neo4j_driver.session() as session:
        result = session.run(query)
        diseases = [record["disease"] for record in result]
    return diseases

def get_disease_overviews():
    """
    Pull per-disease overview text. Fallback to disease name if missing.
    """
    query = """
    MATCH (d:Disease)
    OPTIONAL MATCH (d)-[:HAS_OVERVIEW]->(o)
    RETURN toLower(d.name) AS disease, coalesce(o.content, "") AS overview
    """
    over = {}
    with neo4j_driver.session() as session:
        for rec in session.run(query):
            d = rec["disease"]
            txt = rec["overview"] or ""
            over[d] = (f"{d}: {txt.strip()}" if txt.strip() else d)
    return over

# diseases list from KG
diseases = get_all_diseases()

# cache (name-level) embeddings for detection
disease_embeddings_cache = {
    disease: {
        'pubmed': get_pubmedbert_embedding(disease),
        'biobert': get_biobert_embedding(disease),
        'st': get_sentence_transformer_embedding(disease)
    } for disease in diseases
}

# aspect keywords and embeddings (for fallback)
aspect_keywords = {
    "overview": ["overview", "summary"],
    "symptoms": ["symptoms", "signs"],
    "when to see a doctor": [
        "when to see a doctor", "should i see a doctor", "when should i see a doctor",
        "see a doctor", "seek medical attention", "emergency", "call 911", "when to seek help",
        "when to seek medical attention", "when to see a gp", "should i go to the er",
        "when to go to the hospital"
    ],
    "causes": ["causes", "reason"],
    "complications": ["complications", "risks"],
    "prevention": ["prevention", "prevent", "preventing", "avoid", "reduce", "lower"],
    "diagnosis": ["diagnosis", "diagnose", "diagnosed", "tests", "test", "examination", "exam", "detection", "detected"],
    "treatment": ["treatment", "treat", "cure", "treated"]
}


aspect_embeddings_cache = {
    aspect: {
        'pubmed': get_pubmedbert_embedding(aspect),
        'biobert': get_biobert_embedding(aspect),
        'st': get_sentence_transformer_embedding(aspect)
    } for aspect in aspect_keywords.keys()
}

def compute_average_similarity(query_emb, target_emb, weights=(0.5, 0.25, 0.25)):
    sim_pubmed = cosine_similarity([query_emb['pubmed']], [target_emb['pubmed']])[0][0]
    sim_biobert = cosine_similarity([query_emb['biobert']], [target_emb['biobert']])[0][0]
    sim_st = cosine_similarity([query_emb['st']], [target_emb['st']])[0][0]
    return weights[0]*sim_pubmed + weights[1]*sim_biobert + weights[2]*sim_st
