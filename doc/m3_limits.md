# M3 (structured memory): what is not done

- **`facts` is never written.** No agent has the `ledger_*` tools. Giving them to the sensei (a 9B model
  that already has `dictionary`, `conjugate`, `kanji`, `time` and RAG passages) risks worse answers and
  badly formed tool calls. To do it: add the tools to the sensei, then run the benchmark and the
  adversarial cases before keeping it.
- **`errors` are written by code, not by an agent's tool** (`Orchestrator.record_errors`). The M3-1 box
  "the agent writes these records itself through tools" stays unchecked.
- **An error is resolved only by a simple text match**: when a learner's message contains the stored
  correction and not the mistake (`Orchestrator._resolve_errors`). A learner who only quotes the
  correction in a question ("is X right?") resolves it too early, and a rephrased correct sentence does
  not resolve it.
- **Session-start notes are English prompts**, like every other prompt of the project.
