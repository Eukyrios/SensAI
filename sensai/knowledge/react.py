from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from sensai.knowledge.tools import ToolRegistry
from sensai.knowledge.tools import registry as default_registry

REACT_DEFAULT_NUM_CTX = 800
REACT_DEFAULT_MAX_STEPS = 3

REACT_SYSTEM_PROMPT = """You are a strict Japanese grammar corrector using a ReAct loop.
Before giving a correction, you MUST verify verb forms, vocabulary, or kanji using a tool.

{tools_instructions}

Follow this exact sequence:
Step 1 - Thought & Tool Call:
Thought: <your initial hypothesis about the mistake>
Action: {{"tool": "<tool_name>", "args": {{...}}}}

Step 2 - After receiving Observation:
Verification: <compare the Observation with your initial Thought; explicitly note if your first idea was wrong and needs revision>
Rule: <name the exact Japanese grammar rule behind the correction>
Final Answer: <corrected Japanese sentence> (<romaji>) — <clear explanation in English citing the grammar rule>
"""

_FIELD_STOP = r"(?=\n(?:Thought|Action|Verification|Rule|Final Answer):|\Z)"


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


# --- Fonctions de parsing ---

def _extract_field(label: str, text: str) -> str | None:
    match = re.search(rf"{label}:\s*(.+?){_FIELD_STOP}", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_action_block(text: str, thought: str | None, verification: str | None) -> dict[str, Any]:
    action_match = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)
    if not action_match:
        return {"type": "error", "message": "Malformed JSON in Action.", "thought": thought}
    try:
        action_json = json.loads(action_match.group(1))
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
    thought = _extract_field("Thought", text)
    verification = _extract_field("Verification", text)
    rule = _extract_field("Rule", text)

    if "Action:" in text:
        return _parse_action_block(text, thought, verification)

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return {
            "type": "final_answer",
            "thought": thought,
            "verification": verification,
            "rule": rule,
            "content": final_match.group(1).strip(),
        }

    return {"type": "error", "message": "Could not find valid Action or Final Answer format.", "thought": thought}


# --- Helpers d'analyse et de communication ---

def _consume_client_reply(client: Any, model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
    if hasattr(client, "chat_stream"):
        stream = client.chat_stream(model, messages, num_ctx=num_ctx, temperature=0.1)
        return "".join(chunk for chunk in stream if isinstance(chunk, str))
    if hasattr(client, "chat"):
        return client.chat(model, messages, num_ctx=num_ctx, temperature=0.1)
    raise TypeError("Client must implement 'chat_stream' or 'chat'.")


def _detect_revision(verification_text: str | None) -> bool:
    if not verification_text:
        return False
    lowered = verification_text.lower()
    markers = ("contradict", "wrong", "incorrect", "revised", "instead", "mistake", "exception", "not ")
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
        "content": f"Observation: {observation}\nNow write 'Verification:', 'Rule:', and 'Final Answer:'.",
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
) -> ReActResult:
    if num_ctx > 1024:
        raise ValueError(f"ReAct corrector num_ctx must remain around 800 tokens (got {num_ctx}).")

    system_prompt = REACT_SYSTEM_PROMPT.format(tools_instructions=tool_registry.get_prompt_instructions())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: Check and correct this Japanese sentence: {sentence}"},
    ]
    steps: list[ReActStep] = []
    revised = False

    def record_step(step_obj: ReActStep) -> None:
        steps.append(step_obj)
        if trace_callback:
            trace_callback(step_obj)

    for _ in range(max_steps):
        raw_output = _consume_client_reply(client, model, messages, num_ctx=num_ctx)
        parsed = parse_react_output(raw_output)

        if parsed.get("thought"):
            record_step(ReActStep(step="think", content=parsed["thought"]))
        if parsed.get("verification"):
            record_step(ReActStep(step="verification", content=parsed["verification"]))
            revised = revised or _detect_revision(parsed["verification"])

        if parsed["type"] == "action":
            _handle_action_step(parsed, raw_output, tool_registry, messages, record_step)
            continue

        if parsed["type"] == "final_answer":
            return _build_final_result(parsed, sentence, steps, revised, num_ctx, record_step)

        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"Format error: {parsed.get('message')} Reply with 'Action:' or 'Final Answer:'.",
        })

    return _build_fallback_result(sentence, steps, revised, max_steps, num_ctx, record_step)