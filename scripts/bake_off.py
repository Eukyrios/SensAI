import time
from tabulate import tabulate  # pip install tabulate
from sensai.llm.ollama import OllamaClient
from sensai.core.messages import Message

# Les 3 modèles exigés par le cahier des charges
MODELS = ["qwen3.5:4b", "qwen3.5:9b", "gemma4:12b-it-qat"]
NUM_CTX = 2048 # Basé sur vos tests actuels

# 20 cas en japonais : grammaire, traduction, kanji, et correction
TEST_CASES = [
    "Traduis en japonais formel : 'Je voudrais réserver une table pour deux personnes à 19h.'",
    "Explique la différence entre les particules 'wa' (は) et 'ga' (が) avec des exemples simples.",
    "Conjugue le verbe 'taberu' (食べる) au passé négatif poli.",
    "Que signifie ce kanji : 鬱 ? Donne ses lectures principales.",
    "Corrige cette phrase : 'Watashi wa eiga ni mimasu.'",
    "Comment dit-on 'C'est la première fois que je viens au Japon' ?",
    "Donne 3 exemples d'utilisation de la forme en '-te' (て).",
    "Quelle est la différence entre 'arimasu' et 'imasu' ?",
    "Traduis : 'Il pleut depuis ce matin, donc je ne veux pas sortir.'",
    "Explique l'usage grammatical de '~te shimau' (〜てしまう).",
    "Donne la lecture et le sens du mot : 飛行機",
    "Formule une demande polie avec 'itadakemasen ka'.",
    "Traduis en français : '明日は忙しいから、明後日にしましょう。'",
    "Comment exprimer le désir (je veux faire...) en japonais ?",
    "Corrige cette phrase : 'Kono hon wa oishii desu.'",
    "Explique le concept de 'Keigo' (langage honorifique) brièvement.",
    "Traduis : 'Même si c'est difficile, je n'abandonnerai pas.'",
    "Quel est le compteur approprié pour compter des feuilles de papier ?",
    "Donne la différence entre 'kiku' (聞く) et 'kikoeru' (聞こえる).",
    "Écris une courte présentation personnelle (Jikoshoukai) classique."
]

def run_evaluation():
    client = OllamaClient("http://localhost:11434/")
    results = []

    print(f"Démarrage du Bake-off sur {len(TEST_CASES)} prompts avec num_ctx={NUM_CTX}...\n")

    for model in MODELS:
        print(f"\n--- Évaluation de {model} ---")
        try:
            client.require_model(model)
        except Exception as e:
            print(f"Erreur : Modèle manquant. Lance 'ollama pull {model}'")
            continue

        total_tokens_sec = 0
        valid_runs = 0

        for i, prompt in enumerate(TEST_CASES, 1):
            # Ajout d'une contrainte stricte pour éviter les dissertations à l'infini
            constrained_prompt = prompt + "\n(Réponds en 2 phrases maximum, sois très concis.)"
            messages = [{"role": "user", "content": constrained_prompt}]
            
            print(f"\n  Prompt {i:02d} : ", end="", flush=True)
            stream = client.chat_stream(model, messages, num_ctx=NUM_CTX, temperature=0.1)
            
            stats = None
            while True:
                try:
                    chunk = next(stream)
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                except StopIteration as stop:
                    stats = stop.value
                    break
            
            if stats:
                duration_s = stats.eval_duration_ns / 1e9 if stats.eval_duration_ns else (stats.total_duration_ns / 1e9)
                if duration_s > 0:
                    tps = stats.eval_count / duration_s
                    total_tokens_sec += tps
                    valid_runs += 1
                    print(f"\n  -> [Vitesse : {tps:.2f} t/s]")
            
        avg_tps = (total_tokens_sec / valid_runs) if valid_runs > 0 else 0
        results.append([model, f"{avg_tps:.2f} t/s", "À évaluer"])

    print("\n" + "="*50)
    print("RÉSULTATS DU BAKE-OFF (Layer 2 - Task C)")
    print("="*50)
    print(tabulate(results, headers=["Modèle", "Vitesse Moyenne", "Précision"], tablefmt="grid"))

if __name__ == "__main__":
    run_evaluation()
if __name__ == "__main__":
    run_evaluation()