# Adaptive-Explainable-Segmentation-of-Biomedical-Questions-for-KG-Based-Information-Retrieval
# SEG-SHAP Biomedical Question Answering Pipeline

This repository contains the Python implementation of a segmentation-aware biomedical question-answering pipeline for a cardiovascular knowledge graph. The system takes a user medical question, corrects grammar or spelling issues, detects the disease and medical aspect, segments complex questions into simpler sub-questions, retrieves grounded answers from Neo4j, and uses a T5 paraphrasing loop when the confidence score is low.

## Project Overview

The pipeline is designed for knowledge graph-grounded medical question answering. Instead of generating unsupported answers, it retrieves information from a structured Neo4j knowledge graph. The system combines:

- Grammar correction for noisy user questions.
- Disease detection using direct matching and semantic similarity.
- Aspect detection for categories such as symptoms, causes, diagnosis, treatment, prevention, complications, overview, and when to see a doctor.
- SHAP-guided segmentation for complex multi-intent questions.
- PubMedBERT, BioBERT, and SentenceTransformer embeddings for biomedical semantic matching.
- Neo4j retrieval for grounded medical answers.
- T5 paraphrasing for low-confidence questions.

## Files Included

- `block_00_config.py`  
  Contains shared imports, warnings, and the spaCy model setup.

- `block_01_neo4j_setup.py`  
  Sets up the Neo4j connection and LlamaIndex graph store.

- `block_02_embedding_models.py`  
  Loads PubMedBERT, BioBERT, and SentenceTransformer models and defines embedding functions.

- `block_03_grammar_corrector.py`  
  Loads the grammar correction model and defines the grammar correction function.

- `block_04_kg_bootstrap.py`  
  Loads disease names and aspects from the Neo4j knowledge graph and creates embedding caches.

- `block_05_detection.py`  
  Contains disease detection, aspect detection, and confidence score functions.

- `block_06_shap_segmentation.py`  
  Performs SHAP-guided segmentation and creates canonical sub-questions.

- `block_07_kg_retrieval.py`  
  Retrieves medical information from the Neo4j knowledge graph and cleans retrieved text.

- `block_08_segmentation_retrieval.py`  
  Connects segmentation output with knowledge graph retrieval.

- `block_09_t5_paraphraser.py`  
  Uses T5 to rephrase low-confidence questions and retries retrieval.

- `block_10_main.py`  
  Main runnable script with an example question.

- `requirements.txt`  
  Lists the required Python packages.

- `README_RUN_ORDER.txt`  
  Gives a short run order for the block files.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Install the required spaCy model:

```bash
python -m spacy download en_core_web_md
```

You also need:

- Python 3.9 or later.
- Neo4j installed and running.
- A cardiovascular knowledge graph already loaded into Neo4j.
- Internet access the first time you run the code, because Hugging Face models must be downloaded.

## Neo4j Configuration

Open `block_01_neo4j_setup.py` and update these values:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "your_username"
NEO4J_PASSWORD = "your_password"
```

Make sure the Neo4j database contains the expected graph structure.

## Expected Knowledge Graph Structure

The code expects disease nodes connected to aspect nodes using relationships such as:

- `HAS_OVERVIEW`
- `HAS_SYMPTOMS`
- `HAS_WHENTOSEEDOCTOR`
- `HAS_CAUSES`
- `HAS_COMPLICATIONS`
- `HAS_PREVENTION`
- `HAS_DIAGNOSIS`
- `HAS_TREATMENT`

It also expects optional links to:

- `Source` nodes using `HAS_SOURCE`
- `Reference` nodes using `HAS_REFERENCE`
- `Image` nodes using `HAS_IMAGE`

## How to Run

After installing the dependencies and updating the Neo4j credentials, run:

```bash
python block_10_main.py
```

The default example question is inside `block_10_main.py`:

```python
query = "What are the symptoms of anemia?"
```

You can replace this with any question related to the diseases available in your knowledge graph.

## Run Order

The files are imported automatically when you run `block_10_main.py`. Conceptually, the order is:

1. `block_00_config.py`
2. `block_01_neo4j_setup.py`
3. `block_02_embedding_models.py`
4. `block_03_grammar_corrector.py`
5. `block_04_kg_bootstrap.py`
6. `block_05_detection.py`
7. `block_06_shap_segmentation.py`
8. `block_07_kg_retrieval.py`
9. `block_08_segmentation_retrieval.py`
10. `block_09_t5_paraphraser.py`
11. `block_10_main.py`

## Example Output

The system prints:

- The best original or paraphrased query.
- The segmentation report.
- Detected disease and aspect information.
- Confidence score.
- Retrieved answer from the knowledge graph.
- Runtime information.
- SHAP runtime information.

## Main Workflow

The pipeline works as follows:

1. The user enters a biomedical question.
2. The question is corrected using a grammar correction model.
3. The system detects possible disease names.
4. The system detects the medical aspect, such as symptoms or treatment.
5. Complex questions are segmented into simpler sub-questions.
6. SHAP is used to support explainable segmentation and model contribution analysis.
7. Each sub-question is mapped to a disease and aspect.
8. The matching answer is retrieved from Neo4j.
9. If confidence is low, T5 generates paraphrased versions of the question.
10. The best answer is returned with a confidence score.

## Notes

- The system is retrieval-based and depends on the diseases and aspects available in your Neo4j knowledge graph.
- If a disease is not in the graph, the system may return a low-confidence result or fail to retrieve an answer.
- The first run can be slow because the embedding, grammar correction, and T5 models need to be downloaded.
- SHAP can also increase runtime, especially for long or complex questions.

## Suggested Use

This code is suitable for experiments on explainable, knowledge graph-grounded biomedical question answering. It can be used to test disease detection, aspect detection, segmentation accuracy, grounded answer retrieval, and confidence-based query rewriting.
