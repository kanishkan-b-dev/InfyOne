"""
V2 Grounded Answer Generation Service.
Implements strict hallucination-free generation using retrieved context.
"""
import os
import logging
from typing import Optional, List
from openai import OpenAI

from app.services.retriever_service import RetrievedDocument

logger = logging.getLogger(__name__)

V2_GROUNDED_SYSTEM_PROMPT = """You are the official AI Assistant for Athena Institute of Technology (AIT) operating in strict grounded mode.

CRITICAL GROUNDING RULES:
1. Answer the user's question using ONLY the facts explicitly stated in the RETRIEVED KNOWLEDGE BASE CONTEXT provided below.
2. DO NOT use external world knowledge, personal assumptions, or logical extrapolation to fill in gaps.
3. NEVER invent or hallucinate facts, numbers, dates, statistics, URLs, emails, faculty names, department names, course names, or facilities.
4. DO NOT extrapolate lists. For example, if the context lists specific recruiters or courses, list ONLY those exact ones. Never mention unlisted entities (e.g. Tesla, Boeing, Deloitte, Siemens, Goldman Sachs) unless they are verbatim in the context.
5. If the retrieved context is empty, unrelated, or does not contain enough information to answer the question accurately, you MUST state explicitly:
   "I'm sorry, that information is not available in the Athena Institute of Technology knowledge base. Please contact the admissions office or visit the official website for further assistance."
6. Maintain a helpful, professional, and clear tone. Cite or refer to the specific policy or document title when appropriate.
"""

V2_GROUNDED_USER_PROMPT = """RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION:
{question}

Provide an accurate, strictly grounded response based solely on the context above:"""


class GroundedGeneratorV2:
    """
    Generates answers strictly grounded in retrieved documents using OpenAI.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,  # 0.0 for deterministic, fact-grounded responses
    ):
        self.model_name = (
            model_name
            or os.getenv("OPENAI_MODEL_V2")
            or os.getenv("OPENAI_MODEL")
            or "gpt-5.4-mini"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for GroundedGeneratorV2.")

        self._client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        question: str,
        context: str,
        documents: Optional[List[RetrievedDocument]] = None,
    ) -> str:
        """
        Generate grounded response from context.
        """
        if not context or not context.strip():
            return (
                "I'm sorry, that information is not available in the Athena Institute of "
                "Technology knowledge base. Please contact the admissions office or visit "
                "the official website for further assistance."
            )

        user_content = V2_GROUNDED_USER_PROMPT.format(
            context=context.strip(),
            question=question.strip(),
        )

        # Newer models (GPT-5 series, reasoning models) require max_completion_tokens
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": V2_GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": 800,
        }

        response = self._client.chat.completions.create(**kwargs)

        return response.choices[0].message.content.strip()


def get_grounded_generator_v2() -> GroundedGeneratorV2:
    return GroundedGeneratorV2()
