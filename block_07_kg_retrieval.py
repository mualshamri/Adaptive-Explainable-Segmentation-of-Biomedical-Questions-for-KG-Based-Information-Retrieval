# ============================================================
# BLOCK 07: Knowledge Graph Retrieval and Text Cleaning
# ============================================================

from block_00_config import re, nlp
from block_01_neo4j_setup import neo4j_driver

# KG Retrieval + Cleaning
def retrieve_medical_info(disease, aspect):
    aspect_mapping = {
        "overview": "HAS_OVERVIEW",
        "symptoms": "HAS_SYMPTOMS",
        "when to see a doctor": "HAS_WHENTOSEEDOCTOR",
        "causes": "HAS_CAUSES",
        "complications": "HAS_COMPLICATIONS",
        "prevention": "HAS_PREVENTION",
        "diagnosis": "HAS_DIAGNOSIS",
        "treatment": "HAS_TREATMENT"
    }
    relationship = aspect_mapping.get(aspect.lower())
    if not relationship:
        return {"error": f"Invalid aspect requested: {aspect}"}
    query = f"""
    MATCH (d:Disease) WHERE toLower(d.name) = toLower($disease)
    MATCH (d)-[:{relationship}]->(a)
    OPTIONAL MATCH (a)-[:HAS_SOURCE]->(s)
    OPTIONAL MATCH (d)-[:HAS_REFERENCE]->(r)
    OPTIONAL MATCH (d)-[:HAS_IMAGE]->(img)
    RETURN 
        a.content AS content, 
        s.name AS source, 
        s.rating AS rating,
        COLLECT(DISTINCT r.name) AS references,  
        COLLECT(DISTINCT img.url) AS images
    """
    with neo4j_driver.session() as session:
        record = session.run(query, disease=disease.lower()).single()
    return record if record else None

def clean_retrieved_content(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    unique_sentences = list(dict.fromkeys(sentences))
    filtered_sentences = []
    for sent in unique_sentences:
        if not any((sent != other and sent in other) for other in unique_sentences):
            filtered_sentences.append(sent)
    cleaned_text = " ".join(filtered_sentences)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text