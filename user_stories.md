# User stories

Two user stories per feature, written when the feature is specced.

## M1 — Session persistence & profile

- En tant qu'utilisateur, je veux reprendre une conversation après avoir relancé Sensai, afin de ne pas réexpliquer le contexte de mes échanges précédents.
- En tant qu'utilisateur, je veux que chaque tour (ma question, la réponse, le temps de réponse) reste enregistré, afin de pouvoir suivre ma progression et retrouver une explication donnée plus tôt.

## A2 — Multi-agent orchestration

**As a** beginner learner, **I want** my Japanese sentences checked by a dedicated corrector agent while the sensei keeps teaching me, **so that** I get a precise correction without losing the flow of the lesson.

Acceptance criteria:
- Sending a sentence with a mistake produces two agent runs in the trace: `sensei` and `corrector`, the corrector's `parent_run_id` pointing to the sensei's run.
- The handoff between them is a JSON payload (the sentence and the suspected error) shown in the trace, never free text.
- The reply shows the corrected sentence with its romaji and an explanation in English.

**As a** learner, **I want** the tutor to keep answering even when one of its agents fails or loops, **so that** a bug in one specialist never leaves me without a reply.

Acceptance criteria:
- An agent spec that hands work back and forth forever is stopped by the orchestrator after the maximum number of hops, within the turn budget, and the trace shows which guard stopped it.
- When an agent times out or raises an error, the sensei answers instead and the learner sees a normal reply.
- Both cases are covered by automated tests that run without a real model.
