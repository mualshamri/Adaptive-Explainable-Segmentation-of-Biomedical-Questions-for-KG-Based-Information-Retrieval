# ============================================================
# BLOCK 01: Neo4j Connection and LlamaIndex Graph Store Setup
# ============================================================

from block_00_config import GraphDatabase, Neo4jPropertyGraphStore, HuggingFaceEmbedding, Settings

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = ""
NEO4J_PASSWORD = ""

try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Connected to Neo4j successfully!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    neo4j_driver = None

graph_store = Neo4jPropertyGraphStore(
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    url=NEO4J_URI,
)

embedding_model_general = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = embedding_model_general
Settings.llm = None
