"""Input/output PII and scope filtering. STUB: lets everything through until layer 6."""


def filter_input(text: str, config: dict) -> str:
    """Mask personal data before it reaches the model."""
    return text


def filter_output(text: str, config: dict) -> str:
    return text
