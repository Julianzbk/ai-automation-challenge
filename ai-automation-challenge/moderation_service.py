import asyncio
from typing import Optional
from models import ModerationRequest, ModerationResult, ViolationType
from mock_clients import MockOpenAIClient, MockAnthropicClient

EPSILON = 0.000001

class ModerationService:
    """
    Content moderation service using OpenAI's moderation API.

    Current behavior:
    - Uses OpenAI moderation API
    - Returns binary safe/unsafe decision
    - Threshold is hardcoded
    """
    

    def __init__(self, openai_key: str, anthropic_key: str):
        self.openai_client = MockOpenAIClient(api_key=openai_key)
        self.anthropic_client = MockAnthropicClient(api_key=anthropic_key)
        self.hate_threshold = 0.38 - EPSILON
        self.violence_threshold = 0.72 + EPSILON
        self.sexual_threshold = 0.68 + EPSILON
        self.spam_threshold = 0.42 - EPSILON

    async def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """Moderate content using OpenAI."""
        response = await self.openai_client.moderations.create(input=request.content)
        result = response.results[0]

        scores = result.category_scores
        category_map = {
            "hate": ViolationType.HATE_SPEECH,
            "violence": ViolationType.VIOLENCE,
            "sexual": ViolationType.ADULT_CONTENT,
            "spam": ViolationType.SPAM,
        }

        category_thresholds = {
            "hate": self.hate_threshold,
            "violence": self.violence_threshold,
            "sexual": self.sexual_threshold,
            "spam": self.spam_threshold,
        }

        # Collect every category whose score meets or exceeds its own threshold
        flagged_categories = {
            k: getattr(scores, k)
            for k in scores
            if getattr(scores, k) >= category_thresholds[k]
        }

        max_score = max(getattr(scores, k) for k in scores)
        is_safe = not flagged_categories

        violation_type = ViolationType.NONE
        if flagged_categories:
            top_category = max(flagged_categories, key=flagged_categories.get)
            violation_type = category_map.get(top_category, ViolationType.NONE)

        return ModerationResult(
            is_safe=is_safe,
            confidence=max_score,
            violation_type=violation_type,
            flagged_categories=flagged_categories,
            reasoning="Automated moderation check",
            provider="openai"
        )
