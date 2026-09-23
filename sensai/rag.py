"""RAG sur les mails — propriétaire : B
STUB : renvoie de fausses données. À remplacer par ChromaDB."""
from .interfaces import MailChunk


def ingest(mails_dir: str) -> int:
    """Indexe les .eml du dossier. Renvoie le nombre de chunks indexés."""
    return 0


def retrieve(query: str, k: int = 5) -> list[MailChunk]:
    return [{
        "id": "stub-1",
        "sender": "julie@example.com",
        "subject": "Réunion jeudi",
        "date": "2026-09-22T10:00:00",
        "content": "Salut, tu es dispo jeudi 14h pour la réunion projet ?",
        "score": 0.9,
    }][:k]