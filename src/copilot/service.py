"""Build the engine from configuration. Blank endpoints mean the matching feature is off."""
import json

from copilot.agent import Engine
from copilot.config import ROOT, embeddings_configured, llm_configured
from copilot.rag import load_corpus
from copilot.agent.llm import ChatClient, EmbedClient

EMBEDDINGS_FILE = ROOT / "data" / "processed" / "embeddings.json"


def load_vectors() -> dict[str, list[float]] | None:
    """Load pre-computed passage embeddings (data/processed/embeddings.json) if they were built."""
    if EMBEDDINGS_FILE.exists():
        return json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    return None


def build_engine() -> Engine:
    """Build the Engine from the corpus and whichever endpoints are configured; blank settings switch features off."""
    corpus = load_corpus()
    chat = ChatClient() if llm_configured() else None
    embed = EmbedClient() if embeddings_configured() else None
    return Engine(corpus, chat=chat, embed=embed, vectors=load_vectors())
