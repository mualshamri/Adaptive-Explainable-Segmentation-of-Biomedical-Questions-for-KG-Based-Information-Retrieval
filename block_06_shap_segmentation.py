# ============================================================
# BLOCK 06: SHAP-Guided Question Segmentation
# ============================================================

from block_00_config import re, time, np, shap, get_close_matches, nlp, cosine_similarity
from block_02_embedding_models import (
    pubmedbert_model, pubmedbert_tokenizer,
    biobert_model, biobert_tokenizer,
    st_model, bert_embed_mean,
)
from block_03_grammar_corrector import correct_with_bert
from block_04_kg_bootstrap import diseases, get_disease_overviews

overview_texts = get_disease_overviews()

print("Precomputing disease overview embeddings for segmentation...")
d_over_pubmed = {d: bert_embed_mean(overview_texts[d], pubmedbert_tokenizer, pubmedbert_model) for d in diseases}
d_over_biobert = {d: bert_embed_mean(overview_texts[d], biobert_tokenizer, biobert_model) for d in diseases}
d_over_st = {d: st_model.encode(overview_texts[d]) for d in diseases}
print("Done.")

# Reference embeddings for SHAP (pick stable one)
PLACEHOLDER_DISEASE = diseases[0] if diseases else "unknown"
ref_pubmed = d_over_pubmed[PLACEHOLDER_DISEASE]
ref_biobert = d_over_biobert[PLACEHOLDER_DISEASE]
ref_st = d_over_st[PLACEHOLDER_DISEASE]


# SHAP-Guided Segmentation Module (Integrated) ---
DEFAULT_LAMBDA = 0.6
SHOW_SHAP_WEIGHTS = True
TOP_K_TOKENS = 5

# Aspect triggers used during segmentation
ASPECT_TRIGGERS_SEG = {
    "symptoms": {"symptom", "symptoms", "sign", "signs"},
    "treatment": {"treat", "treated", "treatment", "therapy", "manage", "management"},
    "causes": {
        "cause", "causes", "reason", "reasons",
        "factor", "factors",              
        "risk factor", "risk factors",
        "contribute", "contributes", "contributing",  
        "etiology", "aetiology", "pathogenesis"       
    },
    "diagnosis": {"diagnose", "diagnosis", "test", "tests"},
    "prevention": {"prevent", "prevention", "avoid"},
    "complications": {"complication", "complications", "risk", "risks"},
    "when to see a doctor": {"when to see", "seek help", "medical attention", "emergency", "see a doctor"}
}

def track_disease_positions(full_text: str, diseases_list):
    positions = []
    tlow = full_text.lower()
    for disease in diseases_list:
        for match in re.finditer(re.escape(disease), tlow, flags=re.IGNORECASE):
            positions.append((disease, match.start()))
    return sorted(positions, key=lambda x: x[1])

def extract_disease_names_seg(text: str):
    t = text.lower()

    # 1) strict substring matches with word boundaries
    hits = []
    for d in diseases:
        pat = r"\b" + re.escape(d.lower()) + r"\b"
        if re.search(pat, t):
            hits.append(d)

    if hits:
        # prefer more specific (longer) names first, dedup
        hits = sorted(set(hits), key=lambda x: (-len(x), x))
        return hits

    # 2) fallback: spaCy noun-chunks + fuzzy (slightly stricter cutoff)
    doc = nlp(t)
    candidates = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        match = get_close_matches(phrase, diseases, n=1, cutoff=0.86)
        if match:
            candidates.add(match[0])

    if not candidates:
        match = get_close_matches(t, diseases, n=1, cutoff=0.86)
        if match:
            candidates.add(match[0])

    return list(candidates)



def dep_discourse_split(text: str):
    """Split by sentences, then cautiously split coordinated queries."""
    doc = nlp(text)
    seeds = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    refined = []
    for sent in seeds:
        parts = re.split(r"\b(and|or)\b", sent, flags=re.IGNORECASE)
        if len(parts) <= 1:
            refined.append(sent.strip())
            continue
        buffer = parts[0].strip()
        for i in range(1, len(parts), 2):
            conj = parts[i].lower()
            nxt = parts[i+1].strip() if i+1 < len(parts) else ""
            left_has_aspect = any(kw in buffer.lower() for kws in ASPECT_TRIGGERS_SEG.values() for kw in kws)
            right_has_aspect = any(kw in nxt.lower() for kws in ASPECT_TRIGGERS_SEG.values() for kw in kws)
            if left_has_aspect and right_has_aspect:
                refined.append(buffer.strip(" ,"))
                buffer = nxt
            else:
                buffer = (buffer + " " + conj + " " + nxt).strip()
        if buffer:
            refined.append(buffer.strip(" ,"))
    refined = [r for r in (x.strip() for x in refined) if r]
    return refined

def extract_aspects_set(text: str):
    t = text.lower()
    doc = nlp(t)
    lemmas = [tok.lemma_ for tok in doc]

    hits = []
    for aspect, kws in ASPECT_TRIGGERS_SEG.items():
        count = 0
        for kw in kws:
            if " " in kw:
                # multi-word: use word boundaries
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t):
                    count += 1
            else:
                # single-word: compare by lemma (prevents 'treat' matching 'untreated')
                kw_lemma = nlp(kw.lower())[0].lemma_
                count += lemmas.count(kw_lemma)
        if count > 0:
            hits.append(aspect)

    return hits or ["overview"]


def expand_aspects_diseases(segment: str, known_diseases_in_text):
    aspects = extract_aspects_set(segment)
    explicit = extract_disease_names_seg(segment)

    if explicit:
        Ds = explicit
    else:
        Ds = known_diseases_in_text or []

    # safety: keep it small
    if len(Ds) > 3:
        Ds = shortlist_diseases_by_semantics(segment, Ds, k=3)

    if not Ds:
        return []

    return [(a, d) for a in aspects for d in Ds if d and d != "unknown"]



# SHAP explainers (global, cached)
class SimilarityModelHF:
    def __init__(self, ref_emb, tokenizer, model):
        self.ref_emb = ref_emb
        self.tokenizer = tokenizer
        self.model = model
    def __call__(self, texts):
        sims = []
        for text in texts:
            emb = bert_embed_mean(text, self.tokenizer, self.model)
            sims.append(cosine_similarity([emb], [self.ref_emb])[0][0])
        return np.array(sims)

class SimilarityModelST:
    def __init__(self, ref_emb):
        self.ref_emb = ref_emb
    def __call__(self, texts):
        embs = st_model.encode(texts)
        return np.array([cosine_similarity([e], [self.ref_emb])[0][0] for e in embs])

def token_importance_score(shap_values):
    vals = np.abs(shap_values.values)
    total = vals.sum()
    if total == 0 or len(vals) == 0:
        return 0.0
    vals = vals / total
    top_n = 5 if len(vals) <= 10 else 10
    return float(np.mean(np.sort(vals)[-top_n:]))

TEXT_MASKER_PUBMED = shap.maskers.Text(pubmedbert_tokenizer)
TEXT_MASKER_BIOBERT = shap.maskers.Text(biobert_tokenizer)
TEXT_MASKER_ST = shap.maskers.Text()

PUBMED_EXPLAINER = shap.Explainer(SimilarityModelHF(ref_pubmed, pubmedbert_tokenizer, pubmedbert_model), TEXT_MASKER_PUBMED)
BIOBERT_EXPLAINER = shap.Explainer(SimilarityModelHF(ref_biobert, biobert_tokenizer, biobert_model), TEXT_MASKER_BIOBERT)
ST_EXPLAINER = shap.Explainer(SimilarityModelST(ref_st), TEXT_MASKER_ST, algorithm="partition")


# SHAP-Guided Segmentation Module (Integrated) ---
DEFAULT_LAMBDA = 0.6
SHOW_SHAP_WEIGHTS = True
TOP_K_TOKENS = 5

# Global tracker for SHAP runtime
TOTAL_SHAP_TIME = 0.0





def shap_model_weights(segment_text: str, top_k_tokens=TOP_K_TOKENS):
    print(f"   [SHAP] explaining segment: {segment_text!r}")

    global TOTAL_SHAP_TIME
    shap_start = time.perf_counter()  # <<< start timing SHAP for this segment

    def explain_and_print(model_name, explainer):
        start = time.perf_counter()  # <<< per-model timing
        try:
            sv = explainer([segment_text])
            imp_score = token_importance_score(sv[0])
            try:
                tokens = list(sv.data[0])
            except Exception:
                tokens = list(sv.data)
            try:
                values = np.array(sv.values[0])
            except Exception:
                values = np.array(sv.values)
            token_importances = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)
            print(f"     {model_name} top {top_k_tokens} tokens:")
            for tok, val in token_importances[:top_k_tokens]:
                print(f"        {repr(tok)}: {val:.4f}")
            elapsed_model = time.perf_counter() - start
            print(f"     {model_name} SHAP time: {elapsed_model:.2f} seconds")  # <<< per-model time
            return imp_score
        except Exception as e:
            elapsed_model = time.perf_counter() - start
            print(f"     {model_name} SHAP failed after {elapsed_model:.2f} seconds: {e}")
            return 0.0

    imp_pubmed  = explain_and_print("PubMedBERT", PUBMED_EXPLAINER)
    imp_biobert = explain_and_print("BioBERT", BIOBERT_EXPLAINER)
    imp_st      = explain_and_print("ST", ST_EXPLAINER)

    shap_end = time.perf_counter()
    elapsed_segment = shap_end - shap_start
    TOTAL_SHAP_TIME += elapsed_segment  # <<< accumulate total SHAP runtime

    print(f"     [SHAP] Total SHAP runtime for this segment: {elapsed_segment:.2f} seconds")
    print(f"     [SHAP] Cumulative SHAP runtime so far: {TOTAL_SHAP_TIME:.2f} seconds")

    s = imp_pubmed + imp_biobert + imp_st
    weights = {"pubmed": 1/3, "biobert": 1/3, "st": 1/3} if s <= 0 else {
        "pubmed": imp_pubmed/s, "biobert": imp_biobert/s, "st": imp_st/s
    }

    if SHOW_SHAP_WEIGHTS:
        print(f"     Final SHAP Weights → PubMedBERT: {weights['pubmed']:.3f} | BioBERT: {weights['biobert']:.3f} | ST: {weights['st']:.3f}")
    return weights


def shortlist_diseases_by_semantics(segment: str, candidate_diseases, k=3):
    """Return top-k diseases whose overviews best match this segment."""
    if not candidate_diseases:
        return []

    w = shap_model_weights(segment)

    seg_pub = bert_embed_mean(segment, pubmedbert_tokenizer, pubmedbert_model)
    seg_bio = bert_embed_mean(segment, biobert_tokenizer, biobert_model)
    seg_st  = st_model.encode(segment)

    scored = []
    for d in candidate_diseases:
        try:
            sim_pub = cosine_similarity([seg_pub], [d_over_pubmed[d]])[0][0]
            sim_bio = cosine_similarity([seg_bio], [d_over_biobert[d]])[0][0]
            sim_stv = cosine_similarity([seg_st],  [d_over_st[d]])[0][0]
        except KeyError:
            continue
        score = w["pubmed"] * sim_pub + w["biobert"] * sim_bio + w["st"] * sim_stv
        scored.append((d, float(score)))

    if not scored:
        return []

    # small bias for exact token presence in the question (helps here)
    t = segment.lower()
    for i, (d, s) in enumerate(scored):
        if d in t:
            scored[i] = (d, s + 0.05)

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def assign_disease(segment: str, full_text: str, known_diseases, lam=DEFAULT_LAMBDA):
    # sanitize incoming list
    kd = [d for d in (known_diseases or []) if d and d != "unknown"]
    if not kd:
        # fallback: consider all diseases from the KG
        kd = diseases

    seg_start = full_text.lower().find(segment.lower())

    # proximity score
    prox = {}
    span_list = track_disease_positions(full_text, kd)
    for d, pos in span_list:
        dist = max(1, abs(seg_start - pos))
        prox[d] = max(prox.get(d, 0.0), 1.0 / dist)
    if prox:
        maxp = max(prox.values())
        prox = {d: (v / maxp) for d, v in prox.items()}
    else:
        prox = {d: 0.0 for d in kd}

    # SHAP weights for this segment
    w = shap_model_weights(segment)

    # segment embeddings
    seg_pub = bert_embed_mean(segment, pubmedbert_tokenizer, pubmedbert_model)
    seg_bio = bert_embed_mean(segment, biobert_tokenizer, biobert_model)
    seg_st  = st_model.encode(segment)

    # semantic similarity vs each candidate disease overview
    sem_raw = {}
    for d in kd:
        try:
            sim_pub = cosine_similarity([seg_pub], [d_over_pubmed[d]])[0][0]
            sim_bio = cosine_similarity([seg_bio], [d_over_biobert[d]])[0][0]
            sim_stv = cosine_similarity([seg_st],  [d_over_st[d]])[0][0]
        except KeyError:
            # if a disease somehow lacks a precomputed overview embedding, skip it
            continue
        sem_raw[d] = w["pubmed"] * sim_pub + w["biobert"] * sim_bio + w["st"] * sim_stv

    if sem_raw:
        vals = np.array(list(sem_raw.values()))
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            sem = {d: (sem_raw[d] - vmin) / (vmax - vmin) for d in sem_raw}
        else:
            sem = {d: 0.5 for d in sem_raw}
    else:
        sem = {d: 0.0 for d in kd}

    scores = {d: lam * prox.get(d, 0.0) + (1 - lam) * sem.get(d, 0.0) for d in kd}
    chosen = max(scores, key=scores.get) if scores else "unknown"
    print(f"     → Assigned disease: {chosen} (λ={lam})")
    return chosen


def canonicalize(aspect: str, disease: str):
    if not disease or disease == "unknown":
        return None
    mapping = {
        "overview": f"Provide an overview of {disease}.",
        "symptoms": f"What are the symptoms of {disease}?",
        "causes": f"What are the causes of {disease}?",
        "diagnosis": f"How is {disease} diagnosed?",
        "treatment": f"How is {disease} treated?",
        "prevention": f"How can {disease} be prevented?",
        "complications": f"What are the complications of {disease}?",
        "when to see a doctor": f"When should someone seek medical attention for {disease}?"
    }
    return mapping.get(aspect, f"Provide an overview of {disease}.")

def segment_question(text: str, lam=DEFAULT_LAMBDA):
    corrected = correct_with_bert(text)

    # try explicit matches first
    global_diseases = extract_disease_names_seg(corrected)

    if not global_diseases:
        # shortlist semantically instead of dumping ALL diseases
        global_diseases = shortlist_diseases_by_semantics(corrected, diseases, k=3)

    if not global_diseases:
        # ultra-rare last resort — but still cap to 3
        global_diseases = diseases[:3]

    chunks = dep_discourse_split(corrected)

    segments = []
    for ch in chunks:
        print("\nSegment candidate:", ch)
        _ = shap_model_weights(ch)

        pairs = expand_aspects_diseases(ch, global_diseases)
        if not pairs:
            d = assign_disease(ch, corrected, global_diseases, lam=lam)
            if d != "unknown":
                pairs = [("overview", d)]

        for a, d in pairs:
            if not d or d == "unknown":
                d = assign_disease(ch, corrected, global_diseases, lam=lam)
            if d != "unknown":
                canon = canonicalize(a, d)
                if canon:
                    segments.append(canon)

    seen = set(); out = []
    for s in segments:
        key = s.lower().strip()
        if key not in seen and len(s.split()) > 3:
            seen.add(key); out.append(s)

    return {
        "grammar_corrected": corrected,
        "detected_diseases": global_diseases,
        "segments": out
    }
