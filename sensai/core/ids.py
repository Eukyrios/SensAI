# Uuid ids shared by sessions, turns, runs and their database rows.

import uuid


def new_id() -> str:
    return uuid.uuid4().hex
