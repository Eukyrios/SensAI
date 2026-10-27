"""Automated Evaluation & Hallucination Detection (EV1).

Claim-by-claim verification of LLM responses against provided context/sources.
"""
from dataclasses import dataclass, field
import json
from typing import Literal

from ..llm.ollama import OllamaClient


ClaimStatus = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]


@dataclass
class ClaimVerification:
    claim: str
    status: ClaimStatus
    reasoning: str


@dataclass
class VerificationReport:
    claims: list[ClaimVerification] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def supported_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "SUPPORTED")

    @property
    def contradicted_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "CONTRADICTED")

    @property
    def unverified_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "UNVERIFIED")

    @property
    def faithfulness_score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return round(self.supported_claims / self.total_claims, 2)


class LLMJudge:
    """Evaluates agent responses for factual consistency and hallucination detection."""

    def __init__(self, client: OllamaClient, judge_model: str | None = None) -> None:
        self.client = client
        self.judge_model = judge_model

    def verify_response(self, response_text: str, context: str) -> VerificationReport:
        """Verifies a response against reference context claim-by-claim."""
        if not response_text.strip() or not context.strip():
            return VerificationReport()

        prompt = f"""You are an impartial AI judge evaluating factual consistency.
Analyze the following RESPONSE against the provided CONTEXT.

Step 1: Break the RESPONSE down into standalone factual claims.
Step 2: Verify each claim against the CONTEXT. Assign one status:
- "SUPPORTED": Directly stated or entailed by the CONTEXT.
- "CONTRADICTED": Direct contradiction with the CONTEXT.
- "UNVERIFIED": Information not present in or verifiable from the CONTEXT.

CONTEXT:
{context}

RESPONSE TO EVALUATE:
{response_text}

Respond strictly in valid JSON format with the following structure:
{{
  "claims": [
    {{
      "claim": "text of the claim",
      "status": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED",
      "reasoning": "short explanation"
    }}
  ]
}}
"""

        try:
            model_to_use = self.judge_model or getattr(self.client, "model", "qwen2.5")
            raw_output = ""

            for chunk in self.client.generate_stream(prompt=prompt, model=model_to_use):
                if hasattr(chunk, "text"):
                    raw_output += chunk.text
                elif isinstance(chunk, str):
                    raw_output += chunk

            cleaned_json = raw_output.strip()
            if "```json" in cleaned_json:
                cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_json:
                cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned_json)
            reports = []
            for item in parsed.get("claims", []):
                reports.append(
                    ClaimVerification(
                        claim=item.get("claim", ""),
                        status=item.get("status", "UNVERIFIED"),
                        reasoning=item.get("reasoning", ""),
                    )
                )
            return VerificationReport(claims=reports)

        except Exception as e:
            return VerificationReport(
                claims=[
                    ClaimVerification(
                        claim="Response evaluation",
                        status="UNVERIFIED",
                        reasoning=f"Evaluation failed: {str(e)}",
                    )
                ]
            )