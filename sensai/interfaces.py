"""Contrats partagés entre les modules.
Ne modifiez pas ces formats sans prévenir l'équipe."""
from typing import TypedDict


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant" | "tool"
    content: str


class MailChunk(TypedDict):
    """Retour de rag.retrieve() — propriétaire : B"""
    id: str
    sender: str
    subject: str
    date: str       # ISO 8601, ex: "2026-09-22T10:00:00"
    content: str
    score: float    # similarité, 0 à 1


class PendingAction(TypedDict):
    """Action en attente de confirmation (A3) — propriétaire : C"""
    tool: str       # ex: "send_email"
    args: dict
    summary: str    # texte lisible montré à l'utilisateur