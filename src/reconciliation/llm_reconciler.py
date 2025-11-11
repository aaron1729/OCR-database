"""
LLM-based reconciliation of multiple OCR results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
from difflib import SequenceMatcher

from ..ocr_engines.base import OCRResult
from .llm_providers import LLMProvider


@dataclass
class Discrepancy:
    """Represents a discrepancy between OCR results."""
    position: str  # Description of where in the text
    variants: List[str]  # Different versions from OCR engines
    engines: List[str]  # Which engines produced which variants
    similarity: float  # How similar the variants are (0-1)
    context: str = ""  # Surrounding text for context


@dataclass
class ReconciliationResult:
    """Result of LLM reconciliation."""
    merged_text: str
    confidence: float
    discrepancies: List[Discrepancy] = field(default_factory=list)
    ocr_results: List[OCRResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMReconciler:
    """Reconciles multiple OCR results using LLM analysis."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        flag_threshold: float = 0.3,
        min_engines_required: int = 2
    ):
        """
        Initialize reconciler.

        Args:
            llm_provider: LLM provider to use
            flag_threshold: Similarity threshold below which to flag discrepancies
            min_engines_required: Minimum OCR results needed
        """
        self.llm_provider = llm_provider
        self.flag_threshold = flag_threshold
        self.min_engines_required = min_engines_required

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()

    def find_discrepancies(self, ocr_results: List[OCRResult]) -> List[Discrepancy]:
        """
        Find discrepancies between OCR results.

        Args:
            ocr_results: List of OCR results to compare

        Returns:
            List of identified discrepancies
        """
        if len(ocr_results) < 2:
            return []

        discrepancies = []

        # Compare full texts for major differences
        texts = [result.full_text for result in ocr_results]
        engines = [result.engine_name for result in ocr_results]

        # Check pairwise similarity
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self.calculate_similarity(texts[i], texts[j])

                if similarity < self.flag_threshold:
                    discrepancies.append(Discrepancy(
                        position="full_document",
                        variants=[texts[i], texts[j]],
                        engines=[engines[i], engines[j]],
                        similarity=similarity,
                        context=""
                    ))

        # TODO: Add more granular line-by-line or block-by-block comparison

        return discrepancies

    def create_reconciliation_prompt(
        self,
        ocr_results: List[OCRResult],
        discrepancies: List[Discrepancy]
    ) -> str:
        """
        Create prompt for LLM reconciliation.

        Args:
            ocr_results: OCR results to reconcile
            discrepancies: Identified discrepancies

        Returns:
            Formatted prompt
        """
        prompt = """You are an expert at reconciling multiple OCR transcriptions of historical handwritten documents from the 1800s.

I have multiple OCR results from different engines for the same document. Your task is to:
1. Analyze all versions carefully
2. Produce the most accurate merged transcription
3. Identify and explain any remaining uncertainties

Here are the OCR results:

"""

        for i, result in enumerate(ocr_results, 1):
            prompt += f"--- OCR Engine {i}: {result.engine_name} (confidence: {result.average_confidence:.2f}) ---\n"
            prompt += f"{result.full_text}\n\n"

        if discrepancies:
            prompt += "\nIdentified discrepancies:\n"
            for i, disc in enumerate(discrepancies, 1):
                prompt += f"\n{i}. Location: {disc.position}\n"
                prompt += f"   Similarity: {disc.similarity:.2f}\n"
                for engine, variant in zip(disc.engines, disc.variants):
                    preview = variant[:100] + "..." if len(variant) > 100 else variant
                    prompt += f"   - {engine}: {preview}\n"

        prompt += """

Please provide your response in the following JSON format:
{
    "merged_text": "The reconciled text here...",
    "confidence": 0.95,
    "reasoning": "Brief explanation of your reconciliation decisions",
    "uncertainties": [
        {
            "location": "Description of where in the text",
            "issue": "What is uncertain",
            "possible_readings": ["option1", "option2"]
        }
    ]
}

Focus on producing the most accurate transcription. When in doubt, favor the reading that makes the most sense in context.
"""

        return prompt

    def reconcile(
        self,
        ocr_results: List[OCRResult],
        page_number: Optional[int] = None
    ) -> ReconciliationResult:
        """
        Reconcile multiple OCR results using LLM.

        Args:
            ocr_results: List of OCR results to reconcile
            page_number: Optional page number for tracking

        Returns:
            ReconciliationResult with merged text
        """
        if len(ocr_results) < self.min_engines_required:
            raise ValueError(
                f"Need at least {self.min_engines_required} OCR results, "
                f"got {len(ocr_results)}"
            )

        # Find discrepancies
        discrepancies = self.find_discrepancies(ocr_results)

        # Create reconciliation prompt
        prompt = self.create_reconciliation_prompt(ocr_results, discrepancies)

        # System prompt
        system_prompt = (
            "You are an expert paleographer and historical document analyst "
            "specializing in 19th century handwritten correspondence."
        )

        # Get LLM response
        llm_response = self.llm_provider.generate(prompt, system_prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from response (in case there's extra text)
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                result_data = json.loads(json_str)
            else:
                # If no JSON found, create a basic result
                result_data = {
                    "merged_text": llm_response,
                    "confidence": 0.7,
                    "reasoning": "LLM did not return structured response",
                    "uncertainties": []
                }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            result_data = {
                "merged_text": llm_response,
                "confidence": 0.5,
                "reasoning": "Failed to parse LLM response as JSON",
                "uncertainties": []
            }

        # Convert uncertainties to discrepancies
        llm_discrepancies = []
        for uncertainty in result_data.get('uncertainties', []):
            llm_discrepancies.append(Discrepancy(
                position=uncertainty.get('location', 'unknown'),
                variants=uncertainty.get('possible_readings', []),
                engines=['llm_analysis'],
                similarity=0.0,
                context=uncertainty.get('issue', '')
            ))

        # Combine with original discrepancies
        all_discrepancies = discrepancies + llm_discrepancies

        return ReconciliationResult(
            merged_text=result_data['merged_text'],
            confidence=result_data.get('confidence', 0.5),
            discrepancies=all_discrepancies,
            ocr_results=ocr_results,
            metadata={
                'llm_reasoning': result_data.get('reasoning', ''),
                'num_ocr_engines': len(ocr_results),
                'page_number': page_number,
                'llm_provider': self.llm_provider.model
            }
        )
