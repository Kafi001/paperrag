"""Answer generation.

Defines a small Generator interface with a local implementation. The
abstraction is deliberate: swapping to a hosted API (OpenAI, Anthropic,
HF Inference) means writing one new class, not rewriting the pipeline.

The local generator uses flan-t5-base, which is instruction-tuned. That
matters for RAG: a raw language model tends to continue the prompt, whereas
an instruction-tuned one will actually follow "answer using only the context".
"""

from __future__ import annotations

from typing import Protocol

from .config import Config, default_config


PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


class Generator(Protocol):
    """Anything that can turn a question plus context into an answer."""

    def generate(self, question: str, context: str) -> str: ...


class LocalGenerator:
    """Runs a seq2seq model locally via transformers.

    The model is loaded lazily on first use so importing this module (in
    tests, or to check the API is up) doesn't trigger a model download.
    """

    def __init__(self, config: Config = default_config) -> None:
        self.config = config
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        self._pipe = pipeline(
            "text2text-generation",
            model=self.config.generator_model,
            device=-1,  # CPU; set to 0 for GPU
        )

    def generate(self, question: str, context: str) -> str:
        """Produce an answer grounded in the supplied context."""
        self._ensure_loaded()
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        output = self._pipe(
            prompt,
            max_new_tokens=self.config.max_answer_tokens,
            do_sample=False,  # deterministic: same question, same answer
        )
        return output[0]["generated_text"].strip()


class EchoGenerator:
    """Test double that returns the context verbatim.

    Lets us test the retrieval pipeline end-to-end in CI without downloading
    a model or burning several seconds per test.
    """

    def generate(self, question: str, context: str) -> str:
        return f"[echo] {question} | context chars: {len(context)}"
