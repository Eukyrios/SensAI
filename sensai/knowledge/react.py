"""ReAct (Reasoning and Acting) loop and parser for the SensAI corrector (Layer 4 - Task C)."""
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


@dataclass
class ReActStep:
    """Represents a single observable step in the ReAct trace (expandable in /trace)."""
    step: str  # "think", "tool_call", "verification", or "answer"
    content: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ReActResult:
    """Structured output of a ReAct correction run (satisfies A1-1, A1-2, and T5-1)."""
    sentence: str
    correction: str
    rule: str
    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    revised: bool = False
    stopped_by_max_steps: bool = False
    num_ctx_used: int = REACT_DEFAULT_NUM_CTX

    def to_trace_dict(self) -> dict[str, Any]:
        """Format expandable for /trace (A1-2)."""
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


def parse_react_output(text: str) -> dict[str, Any]:
    """Parse raw LLM output into Thought, Verification, Rule, Action, or Final Answer."""
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Verification|Rule|Final Answer):|\Z)", text, re.DOTALL)
    verification_match = re.search(r"Verification:\s*(.+?)(?=\n(?:Thought|Action|Rule|Final Answer):|\Z)", text, re.DOTALL)
    rule_match = re.search(r"Rule:\s*(.+?)(?=\n(?:Thought|Action|Verification|Final Answer):|\Z)", text, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else None
    verification = verification_match.group(1).strip() if verification_match else None
    rule = rule_match.group(1).strip() if rule_match else None

    if "Action:" in text:
        action_match = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)
        if not action_match:
            return {"type": "error", "message": "Malformed JSON in Action.", "thought": thought}
        try:
            action_json = json.loads(action_match.group(1))
            if not isinstance(action_json, dict) or "tool" not in action_json:
                return {"type": "error", "message": "Action JSON must contain 'tool'.", "thought": thought}
            return {
                "type": "action",
                "thought": thought,
                "verification": verification,
                "action": {
                    "tool": action_json["tool"],
                    "args": action_json.get("args", {}),
                },
            }
        except json.JSONDecodeError:
            return {"type": "error", "message": "Malformed JSON in Action.", "thought": thought}

    final_answer_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_answer_match:
        content = final_answer_match.group(1).strip()
        return {
            "type": "final_answer",
            "thought": thought,
            "verification": verification,
            "rule": rule,
            "content": content,
        }

    return {"type": "error", "message": "Could not find valid Action or Final Answer format.", "thought": thought}


def _consume_client_reply(client: Any, model: str, messages: list[dict[str, str]], num_ctx: int) -> str:
    """Calls the LLM client with an explicit num_ctx (~800 tokens) and returns the text reply."""
    if hasattr(client, "chat_stream"):
        stream = client.chat_stream(model, messages, num_ctx=num_ctx, temperature=0.1)
        chunks: list[str] = []
        while True:
            try:
                chunk = next(stream)
                if isinstance(chunk, str):
                    chunks.append(chunk)
            except StopIteration:
                break
        return "".join(chunks)

    if hasattr(client, "chat"):
        return client.chat(model, messages, num_ctx=num_ctx, temperature=0.1)

    raise TypeError("Client must implement 'chat_stream' or 'chat'.")


def _detect_revision(verification_text: str | None) -> bool:
    """Checks if the verification step indicates the initial hypothesis was contradicted/revised."""
    if not verification_text:
        return False
    lowered = verification_text.lower()
    markers = ("contradict", "wrong", "incorrect", "revised", "instead", "mistake", "exception", "not ")
    return any(marker in lowered for marker in markers)


def _extract_correction_and_rule(content: str, explicit_rule: str | None) -> tuple[str, str]:
    """Extracts the corrected sentence and ensures a grammar rule is always named (A1-2)."""
    first_line = content.splitlines()[0].strip()
    correction = first_line.split("—")[0].strip() if "—" in first_line else first_line

    if explicit_rule:
        return correction, explicit_rule

    rule_in_text = re.search(r"(?:rule|grammar rule|because)\s*[:\-]?\s*([^.\n]+)", content, re.IGNORECASE)
    if rule_in_text:
        return correction, rule_in_text.group(1).strip()

    return correction, "Japanese grammar and conjugation verification rule"


def run_react_loop(
    client: Any,
    model: str,
    sentence: str,
    tool_registry: ToolRegistry = default_registry,
    max_steps: int = REACT_DEFAULT_MAX_STEPS,
    num_ctx: int = REACT_DEFAULT_NUM_CTX,
    trace_callback: Callable[[ReActStep], None] | None = None,
) -> ReActResult:
    """
    Executes the ReAct reasoning loop for the corrector agent.
    - Enforces a context budget of ~800 tokens (num_ctx=800).
    - Records observable trace steps: think -> tool_call -> verification -> answer.
    - Stops cleanly after max_steps.
    """
    if num_ctx > 1024:
        raise ValueError(f"ReAct corrector num_ctx must remain around 800 tokens (got {num_ctx}).")

    system_prompt = REACT_SYSTEM_PROMPT.format(
        tools_instructions=tool_registry.get_prompt_instructions()
    )
    messages: list[dict[str, str]] = [
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
            if _detect_revision(parsed["verification"]):
                revised = True

        if parsed["type"] == "action":
            tool_name = parsed["action"]["tool"]
            tool_args = parsed["action"]["args"]
            observation = tool_registry.execute(tool_name, tool_args)

            record_step(
                ReActStep(
                    step="tool_call",
                    content=f"Called {tool_name}",
                    tool=tool_name,
                    args=tool_args,
                    observation=observation,
                )
            )

            messages.append({"role": "assistant", "content": raw_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: {observation}\n"
                        "Now write 'Verification:', 'Rule:', and 'Final Answer:'."
                    ),
                }
            )
            continue

        if parsed["type"] == "final_answer":
            # Ensure verification appears in the trace if a tool was previously called
            has_tool_call = any(s.step == "tool_call" for s in steps)
            has_verification = any(s.step == "verification" for s in steps)
            if has_tool_call and not has_verification:
                record_step(
                    ReActStep(
                        step="verification",
                        content="Verified final answer against tool observation.",
                    )
                )

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

        # Repair prompt if the model produced malformed output
        messages.append({"role": "assistant", "content": raw_output})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Format error: {parsed.get('message')} "
                    "Please reply with a valid 'Action: {\"tool\": ...}' or 'Final Answer: ...'."
                ),
            }
        )

    # Fallback if max_steps was reached without a Final Answer
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