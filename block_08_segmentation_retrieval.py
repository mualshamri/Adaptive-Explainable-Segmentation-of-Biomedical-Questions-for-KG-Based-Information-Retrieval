# ============================================================
# BLOCK 08: Segmentation-to-Retrieval Integration
# ============================================================

from block_00_config import re
from block_05_detection import identify_diseases, detect_aspect, compute_confidence_score
from block_06_shap_segmentation import DEFAULT_LAMBDA, segment_question
from block_07_kg_retrieval import retrieve_medical_info, clean_retrieved_content

#  Segmentation → Retrieval Integration 
def parse_canonical_segment(canon: str):
    """
    Parse canonical strings like:
      - "What are the symptoms of heart attack?"
      - "How is heart failure treated?"
      - "How is myocardial infarction diagnosed?"
      - "Provide an overview of obesity."
      - "When should someone seek medical attention for asthma?"
    Return (aspect, disease) in lower-case.
    """
    t = canon.strip().lower()

    # overview
    m = re.match(r"provide an overview of (.+?)\.*$", t)
    if m:
        return "overview", m.group(1).strip(" ?.")

    # symptoms
    m = re.match(r"what are the symptoms of (.+?)\??$", t)
    if m:
        return "symptoms", m.group(1).strip(" ?.")

    # causes
    m = re.match(r"what are the causes of (.+?)\??$", t)
    if m:
        return "causes", m.group(1).strip(" ?.")

    # diagnosis
    m = re.match(r"how is (.+?) diagnosed\??$", t)
    if m:
        return "diagnosis", m.group(1).strip(" ?.")

    # treatment
    m = re.match(r"how is (.+?) treated\??$", t)
    if m:
        return "treatment", m.group(1).strip(" ?.")

    # prevention
    m = re.match(r"how can (.+?) be prevented\??$", t)
    if m:
        return "prevention", m.group(1).strip(" ?.")

    # complications
    m = re.match(r"what are the complications of (.+?)\??$", t)
    if m:
        return "complications", m.group(1).strip(" ?.")

    # when to see a doctor
    m = re.match(r"when should someone seek medical attention for (.+?)\??$", t)
    if m:
        return "when to see a doctor", m.group(1).strip(" ?.")

    # fallback: attempt aspect detection + disease extraction
    a, _ = detect_aspect(t)
    ds = identify_diseases(t)
    d = ds[0][0] if ds else None
    return a, d

def _format_segmentation_report(seg_dict):
    corrected = seg_dict.get("grammar_corrected", "")
    diseases = seg_dict.get("detected_diseases", [])
    segments = seg_dict.get("segments", [])

    # Pretty-print diseases (first 5 only)
    if isinstance(diseases, list) and len(diseases) > 5:
        shown = ", ".join(diseases[:5])
        diseases_str = f"{shown} (+{len(diseases)-5} more)"
    else:
        diseases_str = str(diseases)

    seg_lines = "\n".join(f"- {s}" for s in segments) if segments else "(none)"
    return (
        "=== SEGMENTATION REPORT ===\n"
        f"Grammar-Corrected:\n{corrected}\n\n"
        f"Detected Disease(s): {diseases_str}\n\n"
        "Canonical Segments:\n"
        f"{seg_lines}\n"
        "===========================\n"
    )




def generate_response_segmented(user_query, seg_lambda=DEFAULT_LAMBDA):
    """
    1) Segment with SHAP-guided pipeline (canonical outputs)
    2) For each canonical (aspect,disease), retrieve from KG
    3) Compute confidence (via detect_aspect/identify_diseases over canonical)
    4) RETURN answers + a human-readable segmentation/grammar report
    """
    seg = segment_question(user_query, lam=seg_lambda)
    canons = seg["segments"]
    seg_report = _format_segmentation_report(seg)

    if not canons:
        return {
            "response": "No actionable segments were produced.",
            "confidence_score": "0.00",
            "segmentation_report": seg_report,
            "segments": []
        }

    responses = []
    overall_confidences = []

    for canon in canons:
        aspect, disease = parse_canonical_segment(canon)
        if not disease or not aspect:
            responses.append(f"Could not resolve aspect/disease for: {canon}")
            continue

        data = retrieve_medical_info(disease, aspect)
        if not data:
            responses.append(f"No data for {disease.title()} on {aspect.title()}.")
            continue

        content = clean_retrieved_content(data["content"]) if data["content"] else "No content found."
        references = ""
        if data["references"]:
            try:
                references = "\n".join("- " + r for r in data["references"] if r)
            except Exception:
                references = ""

        # Confidence via detectors over canonical
        asp_name, asp_conf = detect_aspect(canon)
        ds = identify_diseases(canon)
        if ds:
            _, dis_conf = ds[0]
        else:
            dis_conf = 0.9 if disease in canon.lower() else 0.0

        final_conf = compute_confidence_score(dis_conf, asp_conf)
        overall_confidences.append(final_conf)

        responses.append(f"""
### {disease.title()} - {aspect.title()}:
{content}

Source: {data['source']}
Rating: {data['rating']}
References:
{references}
Images: {", ".join(data['images']) if data['images'] else ""}

**Disease Confidence:** {dis_conf * 100:.2f}%
**Aspect Confidence:** {asp_conf * 100:.2f}%
**Overall Confidence Score:** {final_conf:.2f}%
""".strip())

    avg_conf = round(sum(overall_confidences)/len(overall_confidences), 2) if overall_confidences else 0.0

    return {
        "response": "\n\n".join(responses),
        "confidence_score": f"{avg_conf:.2f}",
        "segmentation_report": seg_report,   
        "segments": canons                  
    }
