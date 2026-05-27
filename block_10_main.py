# ============================================================
# BLOCK 10: Main Execution Script
# ============================================================

from block_00_config import time
from block_09_t5_paraphraser import answer_with_paraphrase_loop

if __name__ == "__main__":
    query = "What are the symptoms of anemia?"
    start_time = time.perf_counter()
    best_q, result = answer_with_paraphrase_loop(query, min_confidence=90.0, max_attempts=3)
    end_time = time.perf_counter()
    total_time_sec = end_time - start_time

    print("\n===== FINAL OUTPUT =====")
    print("Best (possibly paraphrased) Query:", best_q)

    print("\n" + result.get("segmentation_report", ""))

    print("Confidence Score:", result["confidence_score"])
    print(result["response"])

    print(f"\nTotal runtime for this question: {total_time_sec:.2f} seconds")

    
    from block_06_shap_segmentation import TOTAL_SHAP_TIME
    print(f"Total SHAP runtime (all segments & models): {TOTAL_SHAP_TIME:.2f} seconds")
