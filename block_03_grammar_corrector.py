# ============================================================
# BLOCK 03: Grammar Correction Model
# ============================================================

from block_00_config import AutoTokenizer, AutoModelForSeq2SeqLM

g_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
g_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

def correct_with_bert(text: str) -> str:
    input_ids = g_tokenizer.encode(text, return_tensors="pt")
    outputs = g_model.generate(input_ids, max_length=160, num_beams=4, early_stopping=True)
    return g_tokenizer.decode(outputs[0], skip_special_tokens=True)
