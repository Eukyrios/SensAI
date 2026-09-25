"""Ollama transport, owned by C.

The only code allowed to call the model. Everything goes through the Ollama HTTP API.
    ollama.py  chat streaming, embeddings, model listing and capabilities (/api/tags, /api/show)
"""
