import uuid


def new_id() -> str:
    """Id for turn_id, run_id and session_id."""
    return uuid.uuid4().hex
