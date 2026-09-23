"""Filtrage PII entrée/sortie — propriétaire : A
STUB : laisse tout passer."""


def filter_input(text: str, config: dict) -> str:
    """Masque les données personnelles avant envoi au modèle."""
    return text


def filter_output(text: str, config: dict) -> str:
    return text