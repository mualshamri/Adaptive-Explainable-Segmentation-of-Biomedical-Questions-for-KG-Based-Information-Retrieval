# ============================================================
# BLOCK 09: T5 Paraphrasing Loop for Low-Confidence Questions
# ============================================================

from block_00_config import T5Tokenizer, T5ForConditionalGeneration
from block_08_segmentation_retrieval import generate_response_segmented

# T5 Paraphraser 
t5_model_name = "ramsrigouthamg/t5_paraphraser"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def rephrase_question_candidates(question):
    prompt = f"Rephrase the following question and correct any spelling mistakes: {question} </s>"
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=64, truncation=True)
    outputs = t5_model.generate(
        input_ids,
        max_length=64,
        num_beams=5,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3,
        early_stopping=True
    )
    rephrased_list = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return rephrased_list

def answer_with_paraphrase_loop(query, min_confidence=90.0, max_attempts=3):
    # initial attempt
    response_data = generate_response_segmented(query)
    best_response_data = response_data
    best_confidence = float(response_data["confidence_score"])
    best_rephrased_question = query

    if best_confidence >= min_confidence:
        return best_rephrased_question, best_response_data

    attempt = 0
    while float(response_data["confidence_score"]) < min_confidence and attempt < max_attempts:
        candidates = rephrase_question_candidates(query)

        candidate_results = []
        for candidate in candidates:
            new_response_data = generate_response_segmented(candidate)
            new_confidence = float(new_response_data["confidence_score"])
            candidate_results.append((candidate, new_response_data, new_confidence))

        best_candidate, best_candidate_response, best_candidate_confidence = max(candidate_results, key=lambda x: x[2])

        if best_candidate_confidence > best_confidence:
            best_confidence = best_candidate_confidence
            best_response_data = best_candidate_response
            best_rephrased_question = best_candidate

        response_data = best_candidate_response
        attempt += 1

    return best_rephrased_question, best_response_data
