from __future__ import annotations

from src.avut.dataset import MCQSample
from src.mspragcot.client import GeminiClient


class PragReasoner:
    def __init__(self, client: GeminiClient) -> None:
        """Args: GeminiClient instance. Returns: None."""
        self.client = client

    def reason_and_answer(self, sample: MCQSample, prompt: str) -> str:
        """Args: AVUT sample + final reasoning prompt. Returns: predicted option text (A/B/C/D expected)."""
        text = self.client.generate(prompt, stage="reason_and_answer")
        return text
