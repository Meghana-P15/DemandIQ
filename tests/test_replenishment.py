import pytest
from app.services.replenishment_service import priority_for_score, calculate_priority

@pytest.mark.parametrize(("score","expected"), [
    (0.75, "CRITICAL"), (0.749, "HIGH"),
    (0.55, "HIGH"), (0.549, "MEDIUM"),
    (0.35, "MEDIUM"), (0.349, "LOW"),
    (0.0, "LOW"), (1.0, "CRITICAL"),
])
def test_priority_boundaries(score, expected):
    assert priority_for_score(score)[0] == expected

def test_weighted_priority_formula():
    result = calculate_priority(1, 1, 1, 1, 1)
    assert result["replenishment_priority_score"] == pytest.approx(1.0)
    assert result["priority"] == "CRITICAL"
