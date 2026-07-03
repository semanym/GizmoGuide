"""把手机指标（PhoneMetrics DTO）按规则表映射成打分引擎需要的 ProductSpec。

打分引擎 (app/decision/engine.py) 只认识固定结构的 ProductSpec：价格、各维度
1-10 等级分、存储、重量等。数据源是 ZOL 爬下来的原始参数，先由
app/decision/phone_metrics.py 解析成强类型 DTO（电池 mAh、刷新率 Hz、光圈 f 值、
三防 IP 等级 …），本模块只负责「类型化数值 -> 1-10 等级分」的规则换算。

规则表集中在文件顶部（*_TIERS / *_SCORES），不散落在函数里，方便调整。
所有分数都源自 DTO 里的真实字段；DTO 里是 None（ZOL 无此数据）时才落到中性默认值，
绝不臆造。性能维度依赖芯片档位表 app/decision/chip_tiers.py。

维度与引擎一致：performance / camera / battery / screen / portability /
stability（机身耐用度）/ repair（维修风险）。

设计原则：只做「DTO 数值 -> 等级分」的规则换算，绝不修改 engine.py / weights.py。
"""
from __future__ import annotations

from typing import Optional

from app.decision.chip_tiers import DEFAULT_TIER as CHIP_DEFAULT_TIER
from app.decision.chip_tiers import chip_tier
from app.decision.phone_metrics import PhoneMetrics
from app.schemas.product import ProductSpec
from app.schemas.product_metadata import ProductRecord

# ── 规则阈值表（降序匹配，取「值 >= 下限」的最高档）──────────────────────

BATTERY_TIERS: list[tuple[int, int]] = [
    (7000, 10),
    (6000, 9),
    (5500, 8),
    (5000, 7),
    (4500, 6),
    (4000, 5),
    (3500, 4),
    (3000, 3),
    (0, 2),
]

CHARGE_TIERS: list[tuple[int, int]] = [
    (120, 10),
    (100, 9),
    (80, 8),
    (66, 7),
    (44, 6),
    (33, 5),
    (18, 4),
    (0, 3),
]

REFRESH_TIERS: list[tuple[int, int]] = [
    (144, 10),
    (120, 9),
    (90, 7),
    (60, 5),
    (0, 4),
]

# 主摄像素（万像素），2 亿 = 20000 万
MAIN_CAM_MP_TIERS: list[tuple[int, int]] = [
    (20000, 10),
    (10000, 9),
    (6400, 8),
    (5000, 7),
    (4800, 7),
    (1200, 5),
    (0, 4),
]

# 三防等级（IP 编号越高越强，缺字段落默认值不加不减）→ 机身耐用度基础分
IP_RATING_TIERS: list[tuple[int, int]] = [
    (68, 10),  # IP68/IP69 顶级防尘防水
    (67, 9),
    (66, 8),
    (65, 7),
    (64, 6),
    (0, 5),  # 有三防字段但等级偏低
]

# 屏幕材质关键词 -> 基础分
SCREEN_MATERIAL_SCORES: dict[str, int] = {
    "ltpo": 10,
    "amoled": 8,
    "oled": 8,
    "tft": 4,
    "lcd": 4,
}

# 光圈 f 值 -> 分（越小进光越多越好）；(上限, 分)，取「值 <= 上限」的最高档
APERTURE_TIERS: list[tuple[float, int]] = [
    (1.6, 10),
    (1.8, 8),
    (2.0, 6),
    (float("inf"), 5),
]

# 摄像头颗数 -> 分
CAMERA_COUNT_SCORES: dict[int, int] = {2: 5, 3: 7, 4: 9, 5: 10, 6: 10}

# 中性默认分（对应字段 ZOL 无数据时使用，不臆造高低）
BATTERY_DEFAULT = 5
CHARGE_DEFAULT = 3
REFRESH_DEFAULT = 4
CAMERA_DEFAULT = 5
SCREEN_DEFAULT = 5
STABILITY_DEFAULT = 5

# 影像各子项在 camera 维度里的权重
CAMERA_WEIGHTS = {"mp": 0.5, "aperture": 0.2, "ois": 0.2, "count": 0.1}
# 续航：电池容量与充电速度的权重
BATTERY_WEIGHTS = {"capacity": 0.6, "charge": 0.4}
# 屏幕：材质与刷新率的权重（原亮度维度因覆盖率不足已移除）
SCREEN_WEIGHTS = {"material": 0.55, "refresh": 0.45}
# 影像 OIS 光学防抖档位
OIS_OPTICAL_SCORE = 10
OIS_NON_OPTICAL_SCORE = 6
# 机身耐用度：金属中框相对塑料更结实的调整量
METAL_FRAME_BONUS = 1
PLASTIC_FRAME_PENALTY = 1


def _tier_from(value: Optional[float], table: list[tuple[float, int]], default: int) -> int:
    """降序阈值表：取「值 >= 下限」的最高档；value 为 None 时返回默认分。"""
    if value is None:
        return default
    for floor, score in table:
        if value >= floor:
            return score
    return default


def _tier_at_most(value: Optional[float], table: list[tuple[float, int]], default: int) -> int:
    """升序上限表：取「值 <= 上限」的第一档；value 为 None 时返回默认分。"""
    if value is None:
        return default
    for ceiling, score in table:
        if value <= ceiling:
            return score
    return default


def _clamp(value: float) -> int:
    return max(1, min(10, round(value)))


# ── 各维度评分规则（只读 PhoneMetrics 的类型化字段）──────────────────────

def performance_tier(m: PhoneMetrics) -> int:
    return chip_tier(m.cpu_model) if m.cpu_model else CHIP_DEFAULT_TIER


def battery_tier(m: PhoneMetrics) -> int:
    capacity = _tier_from(m.battery_mah, BATTERY_TIERS, BATTERY_DEFAULT)
    charge = _tier_from(m.wired_charge_w, CHARGE_TIERS, CHARGE_DEFAULT)
    combined = capacity * BATTERY_WEIGHTS["capacity"] + charge * BATTERY_WEIGHTS["charge"]
    return _clamp(combined)


def screen_tier(m: PhoneMetrics) -> int:
    material = SCREEN_MATERIAL_SCORES.get(m.screen_material, SCREEN_DEFAULT)
    refresh = _tier_from(m.refresh_hz, REFRESH_TIERS, REFRESH_DEFAULT)
    combined = material * SCREEN_WEIGHTS["material"] + refresh * SCREEN_WEIGHTS["refresh"]
    return _clamp(combined)


def camera_tier(m: PhoneMetrics) -> int:
    mp = _tier_from(m.main_camera_mp, MAIN_CAM_MP_TIERS, CAMERA_DEFAULT)
    aperture = _tier_at_most(m.aperture_f, APERTURE_TIERS, CAMERA_DEFAULT)
    if m.optical_stabilization is None:
        ois = CAMERA_DEFAULT
    else:
        ois = OIS_OPTICAL_SCORE if m.optical_stabilization else OIS_NON_OPTICAL_SCORE
    count = CAMERA_COUNT_SCORES.get(m.camera_count, CAMERA_DEFAULT) if m.camera_count else CAMERA_DEFAULT
    combined = (
        mp * CAMERA_WEIGHTS["mp"]
        + aperture * CAMERA_WEIGHTS["aperture"]
        + ois * CAMERA_WEIGHTS["ois"]
        + count * CAMERA_WEIGHTS["count"]
    )
    return _clamp(combined)


def portability_tier(m: PhoneMetrics) -> int:
    weight = m.weight_g
    if not weight:
        return 6
    if weight <= 165:
        return 10
    if weight <= 180:
        return 9
    if weight <= 195:
        return 7
    if weight <= 210:
        return 5
    if weight <= 230:
        return 4
    return 3


def stability_tier(m: PhoneMetrics) -> int:
    """机身耐用度：三防 IP 等级为主，金属/塑料中框做微调。

    stability 维度在无软件可靠性数据的前提下，用 ZOL 真实存在的物理耐用信号
    （三防 IP 等级、机身材质）近似。缺字段落中性默认，不因缺数据而误伤。
    """
    base = _tier_from(m.ip_rating, IP_RATING_TIERS, STABILITY_DEFAULT)
    if m.metal_frame is True:
        base += METAL_FRAME_BONUS
    elif m.metal_frame is False:
        base -= PLASTIC_FRAME_PENALTY
    return _clamp(base)


def repair_risk(m: PhoneMetrics) -> str:
    """维修风险：以机身后盖材质近似。玻璃后盖易碎、更换成本高 -> 风险偏高；
    塑料/复合后盖成本低、更易维修 -> 风险低；ZOL 无材质数据时给中性 medium。
    """
    if m.glass_back is None:
        return "medium"
    return "medium" if m.glass_back else "low"


def build_product_spec(record: ProductRecord) -> ProductSpec:
    """ProductRecord(原始 ZOL 数据) -> ProductSpec(规则算出的等级分)。"""
    m = PhoneMetrics.from_record(record)
    return ProductSpec(
        id=record.name,
        name=record.name,
        aliases=sorted(record.aliases - {record.name}),
        brand=m.brand,
        os=m.os,
        price=m.price_yuan,
        release_year=m.release_year or 2024,
        chip_tier=performance_tier(m),
        camera_tier=camera_tier(m),
        battery_tier=battery_tier(m),
        screen_tier=screen_tier(m),
        portability_tier=portability_tier(m),
        stability_tier=stability_tier(m),
        storage_gb=m.storage_gb,
        weight_g=m.weight_g,
        repair_risk=repair_risk(m),
        notes=[],
    )


def build_product_specs(records: list[ProductRecord]) -> list[ProductSpec]:
    return [build_product_spec(record) for record in records]
