# ============================================================
# BLOCK 00: Global Imports and Shared Configuration
# ============================================================

import re
import time
import warnings
from difflib import get_close_matches

import numpy as np
import pandas as pd
import torch
import shap

from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from sentence_transformers import SentenceTransformer

from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

warnings.filterwarnings("ignore", message="This sequence already has </s>.")

# Shared spaCy model used by detection, segmentation, and cleaning.
nlp = spacy.load("en_core_web_md")
