from __future__ import annotations

from app.decision.weights import build_weights
from app.schemas.product import ProductSpec
from app.schemas.recommendation import ProductScore, RecommendationResult
from app.schemas.user_profile import UserProfile
from app.tracing import trace_span

REPAIR_SCORE = {"low": 10.0, "medium": 7.0, "high": 4.0}
# 预算维度中性分：预算未知、或该机型 ZOL 缺价（price<=0）时使用。
# 缺价不能当成「0 元完美匹配预算」拿满分，只能中性处理，既不奖励也不误伤。
NEUTRAL_PRICE_SCORE = 7.0


def recommend(products: list[ProductSpec], profile: UserProfile) -> RecommendationResult:
    with trace_span(
        "recommendation_engine",
        input_data={"products": [p.name for p in products], "profile": profile.model_dump(mode="json")},
    ) as (span, end_span):
        if len(products) < 2:
            raise ValueError("At least two products are required for recommendation.")

        weights = build_weights(profile)
        scored = [_score_product(product, profile, weights) for product in products]
        scored.sort(key=lambda item: item.total_score, reverse=True)

        winner = scored[0]
        runner_up = scored[1]
        margin = max(0.0, winner.total_score - runner_up.total_score)
        confidence = _confidence(profile, products, margin)

        result = RecommendationResult(
            winner_id=winner.product_id,
            winner_name=winner.product_name,
            confidence=confidence,
            scores=scored,
            key_reasons=_build_key_reasons(winner, runner_up, profile),
            risks=_build_risks(scored, profile),
            reversal_conditions=_build_reversal_conditions(winner, runner_up, profile),
            missing_information=_missing_information(profile),
        )

        end_span(output={
            "winner": result.winner_name,
            "confidence": result.confidence,
            "scores": {s.product_name: s.total_score for s in scored},
        })
        return result


def _score_product(product: ProductSpec, profile: UserProfile, weights: dict[str, float]) -> ProductScore:
    dimension_scores = {
        "price": _price_score(product, profile),
        "performance": float(product.chip_tier),
        "camera": float(product.camera_tier),
        "battery": float(product.battery_tier),
        "screen": float(product.screen_tier),
        "portability": _portability_score(product, profile),
        "storage": _storage_score(product, profile),
        "stability": float(product.stability_tier),
        "repair": REPAIR_SCORE[product.repair_risk],
    }

    weighted_total = sum(dimension_scores[name] * weights[name] for name in weights)
    weight_total = sum(weights.values())
    total = weighted_total / weight_total * 10

    penalties = _constraint_penalties(product, profile)
    total -= len(penalties) * 8

    strengths = _strengths(product, dimension_scores)
    return ProductScore(
        product_id=product.id,
        product_name=product.name,
        total_score=round(max(total, 0), 2),
        dimension_scores={key: round(value, 2) for key, value in dimension_scores.items()},
        penalties=penalties,
        strengths=strengths,
    )


def _price_score(product: ProductSpec, profile: UserProfile) -> float:
    # 缺价（ZOL 无电商报价，price<=0）不参与预算匹配，给中性分，避免 0 元被当成完美匹配。
    if product.price <= 0:
        return NEUTRAL_PRICE_SCORE
    if profile.budget is None:
        return NEUTRAL_PRICE_SCORE
    if product.price <= profile.budget:
        saving_ratio = (profile.budget - product.price) / max(profile.budget, 1)
        return min(10.0, 8.0 + saving_ratio * 2)
    over_ratio = (product.price - profile.budget) / max(profile.budget, 1)
    if over_ratio <= 0.1 and profile.budget_flexibility != "low":
        return 6.5
    if over_ratio <= 0.2 and profile.budget_flexibility == "high":
        return 5.5
    return max(1.0, 5.0 - over_ratio * 12)


def _portability_score(product: ProductSpec, profile: UserProfile) -> float:
    score = float(product.portability_tier)
    if profile.max_weight_g and product.weight_g > profile.max_weight_g:
        score -= min(4.0, (product.weight_g - profile.max_weight_g) / 10)
    return max(1.0, score)


def _storage_score(product: ProductSpec, profile: UserProfile) -> float:
    if profile.min_storage_gb is None:
        return 8.0 if product.storage_gb >= 256 else 6.5
    if product.storage_gb >= profile.min_storage_gb:
        return 10.0
    return max(1.0, product.storage_gb / profile.min_storage_gb * 8)


def _constraint_penalties(product: ProductSpec, profile: UserProfile) -> list[str]:
    penalties: list[str] = []
    if profile.budget and product.price > profile.budget * 1.15 and profile.budget_flexibility == "low":
        penalties.append("明显超出预算")
    if profile.os_preference and product.os != profile.os_preference:
        penalties.append("系统偏好不匹配")
    if profile.min_storage_gb and product.storage_gb < profile.min_storage_gb:
        penalties.append("存储低于最低要求")
    if profile.max_weight_g and product.weight_g > profile.max_weight_g:
        penalties.append("重量超过可接受范围")
    return penalties


def _strengths(product: ProductSpec, scores: dict[str, float]) -> list[str]:
    labels = {
        "price": "价格匹配度高",
        "performance": "性能强",
        "camera": "拍照能力强",
        "battery": "续航表现好",
        "screen": "屏幕素质好",
        "portability": "便携性好",
        "storage": "存储更宽裕",
        "stability": "稳定性好",
        "repair": "维修风险较低",
    }
    return [labels[key] for key, value in scores.items() if value >= 8.5][:4]


def _confidence(profile: UserProfile, products: list[ProductSpec], margin: float) -> float:
    score = 0.55
    if profile.budget is not None:
        score += 0.08
    if profile.primary_scenarios:
        score += 0.12
    if profile.usage_years is not None:
        score += 0.06
    if all(product.price > 0 for product in products):
        score += 0.05
    score += min(0.14, margin / 100)
    return round(min(score, 0.92), 2)


def _build_key_reasons(winner: ProductScore, runner_up: ProductScore, profile: UserProfile) -> list[str]:
    reasons = [f"{winner.product_name} 的综合匹配分更高：{winner.total_score} vs {runner_up.total_score}。"]
    diffs = []
    for key, value in winner.dimension_scores.items():
        delta = value - runner_up.dimension_scores.get(key, 0)
        if delta >= 1.0:
            diffs.append((key, delta))
    labels = {
        "price": "价格",
        "performance": "性能",
        "camera": "拍照",
        "battery": "续航",
        "screen": "屏幕",
        "portability": "便携",
        "storage": "存储",
        "stability": "稳定性",
        "repair": "维修风险",
    }
    for key, _ in sorted(diffs, key=lambda item: item[1], reverse=True)[:3]:
        reasons.append(f"在{labels[key]}维度上，{winner.product_name} 更符合当前画像。")
    if winner.penalties:
        reasons.append(f"不过它也有约束扣分：{'、'.join(winner.penalties)}。")
    return reasons


def _build_risks(scores: list[ProductScore], profile: UserProfile) -> list[str]:
    risks: list[str] = []
    for score in scores:
        risks.extend([f"{score.product_name}：{penalty}" for penalty in score.penalties])
        if score.dimension_scores.get("repair", 10) <= 5:
            risks.append(f"{score.product_name}：维修成本或维修风险偏高。")
    if not risks:
        risks.append("当前只基于 ZOL 原始参数和规则打分，尚未接入真实价格、维修和评测证据。")
    return risks[:5]


def _build_reversal_conditions(winner: ProductScore, runner_up: ProductScore, profile: UserProfile) -> list[str]:
    conditions = []
    if profile.budget is not None:
        conditions.append("如果实际成交价变化较大，预算维度可能导致推荐反转。")
    conditions.append("如果你把某个单一维度提高到最高优先级，例如只看拍照或只看游戏性能，结论可能变化。")
    if profile.repair_sensitivity != "high":
        conditions.append("如果你非常在意维修成本和售后风险，需要接入维修证据后再复核。")
    return conditions[:3]


def _missing_information(profile: UserProfile) -> list[str]:
    missing = []
    if profile.budget is None:
        missing.append("预算上限")
    if not profile.primary_scenarios:
        missing.append("主要使用场景")
    if profile.usage_years is None:
        missing.append("预期使用年限")
    missing.append("真实实时价格")
    missing.append("维修和评测证据")
    return missing
