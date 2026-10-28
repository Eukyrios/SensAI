from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from sensai.knowledge.tools import ToolRegistry
from sensai.knowledge.tools import registry as default_registry

REACT_DEFAULT_NUM_CTX = 800
REACT_DEFAULT_MAX_STEPS = 3
REACT_MAX_NUM_CTX = 1024

REACT_SYSTEM_PROMPT = """You are a strict Japanese grammar corrector using a ReAct loop.
Before giving a correction, you MUST verify a verb form, a word or a particle note with a tool.

{tools_instructions}

Step 1, write exactly two lines and stop:
Thought: <your hypothesis about the mistake>
Action: {{"tool": "<tool_name>", "args": {{...}}}}

Step 2, after the Observation, write exactly these lines:
Verification: <compare the Observation with your Thought>
Hypothesis: <wrong if the Observation showed your Thought was mistaken, confirmed if it agreed>
Rule: <the grammar rule behind the correction, in one English sentence>
Correction: <the corrected Japanese sentence only, no romaji>
Final Answer: <corrected sentence> (<romaji>) — <short English explanation citing the rule>
"""
MUST_VERIFY = "You must verify with a tool first. Reply with 'Thought:' and 'Action:' only."
AFTER_OBSERVATION = ("Observation: {observation}\nNow write 'Verification:', 'Hypothesis:', 'Rule:', "
                     "'Correction:' and 'Final Answer:'.")
MAX_OBSERVATION_CHARS = 400

_LABELS = "Thought|Action|Verification|Hypothesis|Revised|Rule|Correction|Final Answer|Observation"
_FIELD_STOP = rf"(?=\n\W*(?:{_LABELS})\W*:|\Z)"


@dataclass
class ReActStep:
    step: str
    content: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ReActResult:
    sentence: str
    correction: str
    rule: str
    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    revised: bool = False
    stopped_by_max_steps: bool = False
    num_ctx_used: int = REACT_DEFAULT_NUM_CTX
    stats: list[Any] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [step.tool for step in self.steps if step.step == "tool_call" and step.tool]

    @property
    def verified(self) -> bool:
        return not self.stopped_by_max_steps and any(
            step.step == "tool_call" and observation_ok(step.observation or "") for step in self.steps)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "correction": self.correction,
            "rule": self.rule,
            "answer": self.answer,
            "revised": self.revised,
            "stopped_by_max_steps": self.stopped_by_max_steps,
            "num_ctx": self.num_ctx_used,
            "steps": [s.to_dict() for s in self.steps],
        }

def observation_ok(observation: str, need_found: bool = True) -> bool:
    if observation.startswith("Error"):
        return False
    try:
        data = json.loads(observation)
    except json.JSONDecodeError:
        return True
    if not isinstance(data, dict):
        return True
    return "error" not in data and not (need_found and data.get("found") is False)

# --- Fonctions de parsing ---
def _clean(text: str) -> str:
    return re.sub(rf"\*+\s*({_LABELS})\s*:\s*\*+", r"\1:", text)

def _extract_field(label: str, text: str) -> str | None:
    match = re.search(rf"{label}:[ \t]*(.*?){_FIELD_STOP}", text, re.DOTALL)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _parse_action_block(text: str, thought: str | None, verification: str | None) -> dict[str, Any]:
    action_match = re.search(r"Action:", text)
    start = text.find("{", action_match.end()) if action_match else -1
    if start < 0:
        return {"type": "error", "message": "Malformed JSON in Action.", "thought": thought}
    try:
        action_json, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return {"type": "error", "message": "Malformed JSON in Action.", "thought": thought}

    if not isinstance(action_json, dict) or "tool" not in action_json:
        return {"type": "error", "message": "Action JSON must contain 'tool'.", "thought": thought}

    return {
        "type": "action",
        "thought": thought,
        "verification": verification,
        "action": {"tool": action_json["tool"], "args": action_json.get("args", {})},
    }


def parse_react_output(text: str) -> dict[str, Any]:
    text = _clean(text)
    thought = _extract_field("Thought", text)
    verification = _extract_field("Verification", text)
    rule = _extract_field("Rule", text)
    revised = _extract_field("Hypothesis", text) or _extract_field("Revised", text)
    correction = _extract_field("Correction", text)

    if "Action:" in text:
        return _parse_action_block(text, thought, verification)

    final_match = re.search(r"Final Answer:\s*(\S.*)", text, re.DOTALL)
    if final_match:
        return {
            "type": "final_answer",
            "thought": thought,
            "verification": verification,
            "rule": rule,
            "revised": revised,
            "correction": correction,
            "content": final_match.group(1).strip(),
        }

    return {"type": "error", "message": "Could not find valid Action or Final Answer format.", "thought": thought}


# --- Helpers d'analyse et de communication ---

def _consume_client_reply(client: Any, model: str, messages: list[dict[str, str]], num_ctx: int,
                          stats: list[Any] | None = None) -> str:
    if hasattr(client, "chat_stream"):
        stream = client.chat_stream(model, messages, num_ctx=num_ctx, temperature=0.1)
        pieces: list[str] = []
        while True:
            try:
                chunk = next(stream)
            except StopIteration as end:
                if stats is not None and end.value is not None:
                    stats.append(end.value)
                return "".join(pieces)
            if isinstance(chunk, str):
                pieces.append(chunk)
    if hasattr(client, "chat"):
        return client.chat(model, messages, num_ctx=num_ctx, temperature=0.1)
    raise TypeError("Client must implement 'chat_stream' or 'chat'.")


def _detect_revision(verification_text: str | None) -> bool:
    if not verification_text:
        return False
    lowered = verification_text.lower()
    markers = ("contradict", "wrong", "incorrect", "revise", "my first idea was", "exception")
    return any(marker in lowered for marker in markers)


def _extract_correction_and_rule(content: str, explicit_rule: str | None) -> tuple[str, str]:
    first_line = content.splitlines()[0].strip()
    correction = first_line.split("—")[0].strip() if "—" in first_line else first_line
    if explicit_rule:
        return correction, explicit_rule

    rule_in_text = re.search(r"(?:rule|grammar rule|because)\s*[:\-]?\s*([^.\n]+)", content, re.IGNORECASE)
    if rule_in_text:
        return correction, rule_in_text.group(1).strip()
    return correction, "Japanese grammar and conjugation verification rule"


# --- Sous-fonctions de la boucle ReAct ---

def _handle_action_step(
    parsed: dict[str, Any],
    raw_output: str,
    tool_registry: ToolRegistry,
    messages: list[dict[str, str]],
    record_step: Callable[[ReActStep], None],
) -> None:
    tool_name = parsed["action"]["tool"]
    tool_args = parsed["action"]["args"]
    observation = tool_registry.execute(tool_name, tool_args)
    shown = observation if len(observation) <= MAX_OBSERVATION_CHARS else observation[:MAX_OBSERVATION_CHARS] + "…"

    record_step(ReActStep(
        step="tool_call",
        content=f"Called {tool_name}",
        tool=tool_name,
        args=tool_args,
        observation=observation,
    ))
    messages.append({"role": "assistant", "content": raw_output})
    messages.append({
        "role": "user",
        "content": AFTER_OBSERVATION.format(observation=shown),
    })


def _build_final_result(
    parsed: dict[str, Any],
    sentence: str,
    steps: list[ReActStep],
    revised: bool,
    num_ctx: int,
    record_step: Callable[[ReActStep], None],
) -> ReActResult:
    has_tool = any(s.step == "tool_call" for s in steps)
    has_verif = any(s.step == "verification" for s in steps)
    if has_tool and not has_verif:
        record_step(ReActStep(step="verification", content="Verified final answer against tool observation."))

    correction, rule = _extract_correction_and_rule(parsed["content"], parsed.get("rule"))
    if parsed.get("correction"):
        correction = _without_romaji(parsed["correction"].splitlines()[0])
    final_text = parsed["content"]
    if rule and rule.lower() not in final_text.lower():
        final_text = f"{final_text} (Grammar rule: {rule})"

    record_step(ReActStep(step="answer", content=final_text))
    return ReActResult(
        sentence=sentence,
        correction=correction,
        rule=rule,
        answer=final_text,
        steps=steps,
        revised=revised,
        stopped_by_max_steps=False,
        num_ctx_used=num_ctx,
    )


def _build_fallback_result(
    sentence: str,
    steps: list[ReActStep],
    revised: bool,
    max_steps: int,
    num_ctx: int,
    record_step: Callable[[ReActStep], None],
) -> ReActResult:
    fallback_rule = "Tool verification limit reached (fallback explanation)"
    fallback_answer = (
        f"Could not complete full verification within {max_steps} steps for '{sentence}'. "
        f"(Grammar rule: {fallback_rule})"
    )
    record_step(ReActStep(step="answer", content=fallback_answer))
    return ReActResult(
        sentence=sentence,
        correction=sentence,
        rule=fallback_rule,
        answer=fallback_answer,
        steps=steps,
        revised=revised,
        stopped_by_max_steps=True,
        num_ctx_used=num_ctx,
    )

def run_react_loop(
    client: Any,
    model: str,
    sentence: str,
    tool_registry: ToolRegistry = default_registry,
    max_steps: int = REACT_DEFAULT_MAX_STEPS,
    num_ctx: int = REACT_DEFAULT_NUM_CTX,
    trace_callback: Callable[[ReActStep], None] | None = None,
    hint: str | None = None,
    should_stop: Callable[[], bool] | None = None,
    system_template: str | None = None,
) -> ReActResult:
    if num_ctx > REACT_MAX_NUM_CTX:
        raise ValueError(f"ReAct corrector num_ctx must remain around 800 tokens (got {num_ctx}).")

    template = system_template if system_template and "{tools_instructions}" in system_template \
        else REACT_SYSTEM_PROMPT
    system_prompt = template.format(tools_instructions=tool_registry.get_prompt_instructions())
    question = f"Question: Check and correct this Japanese sentence: {sentence}"
    if hint:
        question += f"\nA first check suggested (verify it, it may be wrong): {hint}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    steps: list[ReActStep] = []
    stats: list[Any] = []
    revised = False

    def record_step(step_obj: ReActStep) -> None:
        steps.append(step_obj)
        if trace_callback:
            trace_callback(step_obj)

    for _ in range(max_steps):
        if should_stop is not None and should_stop():
            break
        raw_output = _consume_client_reply(client, model, messages, num_ctx=num_ctx, stats=stats)
        parsed = parse_react_output(raw_output)
        verified = any(step.step == "tool_call" for step in steps)

        if parsed.get("thought"):
            record_step(ReActStep(step="think", content=parsed["thought"]))
        if parsed.get("verification") and verified:
            record_step(ReActStep(step="verification", content=parsed["verification"]))
            revised = revised or _is_revision(parsed)

        if parsed["type"] == "action":
            _handle_action_step(parsed, _until_action(raw_output), tool_registry, messages, record_step)
            continue

        if parsed["type"] == "final_answer" and not verified:
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": MUST_VERIFY})
            continue

        if parsed["type"] == "final_answer":
            result = _build_final_result(parsed, sentence, steps, revised, num_ctx, record_step)
            result.stats = stats
            return result

        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"Format error: {parsed.get('message')} Reply with 'Action:' or 'Final Answer:'.",
        })

    result = _build_fallback_result(sentence, steps, revised, max_steps, num_ctx, record_step)
    result.stats = stats
    return result

def _is_revision(parsed: dict[str, Any]) -> bool:
    explicit = (parsed.get("revised") or "").strip().lower()
    if explicit.startswith(("wrong", "mistaken")):
        return True
    if explicit.startswith(("confirmed", "correct", "right")):
        return False
    if explicit.startswith(("yes", "no")) and not re.search(r"\b(was|is) (correct|right|confirmed)\b",
                                                          (parsed.get("verification") or "").lower()):
        return explicit.startswith("yes")
    return _detect_revision(parsed.get("verification"))

def _without_romaji(line: str) -> str:
    line = line.split("—")[0]
    return re.sub(r"\s*[(（][\x00-\x7f\u0100-\u017f\s]*[)）]\s*$", "", line).strip()

def _until_action(raw_output: str) -> str:
    match = re.search(r"Action:\s*\{.*?\}\s*(?:\n|$)", raw_output, re.DOTALL)
    return raw_output[:match.end()].rstrip() if match else raw_output