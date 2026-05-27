# ============================================================
# BLOCK 02: Embedding Models and Embedding Functions
# ============================================================

from block_00_config import AutoModel, AutoTokenizer, SentenceTransformer, torch

#  Embedding Models (PubMedBERT, BioBERT, SentenceT)

pubmedbert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").eval()
pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").eval()
biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# General sentence encoder
st_model = SentenceTransformer('all-mpnet-base-v1')

def get_pubmedbert_embedding(text: str):
    tokens = pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = pubmedbert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_biobert_embedding(text: str):
    tokens = biobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = biobert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_sentence_transformer_embedding(text: str):
    return st_model.encode(text)

def bert_embed_mean(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb
