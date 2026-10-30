from dataclasses import dataclass

WEIGHTS = {"demand": 0.35, "risk": 0.30, "volatility": 0.15, "anomaly": 0.10, "promotion": 0.10}

def priority_for_score(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "CRITICAL", "Review / replenish first"
    if score >= 0.55:
        return "HIGH", "Prioritize replenishment"
    if score >= 0.35:
        return "MEDIUM", "Monitor closely"
    return "LOW", "Normal monitoring"

def calculate_priority(demand_score: float, risk_score: float, volatility_score: float, anomaly_signal: float, promotion_score: float) -> dict:
    score = (WEIGHTS["demand"] * demand_score + WEIGHTS["risk"] * risk_score + WEIGHTS["volatility"] * volatility_score + WEIGHTS["anomaly"] * anomaly_signal + WEIGHTS["promotion"] * promotion_score)
    priority, action = priority_for_score(score)
    return {"replenishment_priority_score": score, "priority": priority, "recommended_action": action}
