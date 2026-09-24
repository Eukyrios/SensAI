# Sensai — Assistant mail local

Assistant IA qui lit, résume et trie tes mails, 100 % en local via Ollama.
Aucune donnée ne quitte la machine.

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et lancé (`ollama serve`)
- Node.js 18+ (pour l'interface web)

## Installation

```bash
git clone <url-du-repo>
cd G-AIA-500-PAR-5-1-sensai-4

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

## Lancement

**CLI**
```bash
python -m sensai.main
python -m sensai.main --model qwen3.5:4b    # autre modèle
```

**Web UI**
```bash
uvicorn api.main:app --reload               # backend : http://127.0.0.1:8000
cd frontend && npm install && npm run dev   # frontend : http://localhost:5173
```

## Configuration

Tout se règle dans `config.yaml` (modèle, RAG, persona, guardrails…), sans modifier le code.

## Features

| Code | Feature | Statut |
|------|---------|--------|
| B1 | CLI chatbot (streaming, historique, erreurs) | 🚧 |
| M1 | Persistance de session & profil utilisateur | ⏳ |
| R1 | RAG sur la boîte mail | ⏳ |
| T5 | Extraction structurée (expéditeur, urgence, action) | ⏳ |
| T1 | Appel d'outils (recherche, envoi de mail) | ⏳ |
| A3 | Confirmation avant action irréversible | ⏳ |
| A5 | Personas switchables en cours de session | ⏳ |
| EV2 | Masquage des données personnelles | ⏳ |
| X1 | Interface web | ⏳ |

✅ fait · 🚧 en cours · ⏳ à faire

## Exemples d'utilisation

```
> Résume mes mails non lus de ce matin
> Quels mails demandent une action de ma part ?
> Réponds à Julie que je suis dispo jeudi
  [confirmation] Envoyer ce mail à julie@... ? (o/n)
> /persona casual
```

## Structure

```
sensai/      code Python (chat, RAG, outils, guardrails)
api/         backend FastAPI
frontend/    interface React
data/mails/  jeu de mails de démo (.eml)
personas/    configurations des personas
```

## User stories

Voir [`user_stories.md`](user_stories.md).

## Équipe

| Membre | Rôle |
|--------|------|
| … | Cœur (B1, M1, A5, EV2) |
| … | RAG (R1) |
| … | Outils & actions (T5, T1, A3) |
| … | Web UI (X1) |