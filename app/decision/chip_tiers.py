"""芯片性能档位表：把 ZOL 的 CPU型号 字符串映射成 1-10 的性能档位分。

ZOL 的 CPU 字段是自由文本，带厂商前缀和空格差异（"高通 骁龙8 Gen3"、
"高通骁龙8 Gen3"、"联发科 天玑9400"、"海思 麒麟 9020"、"苹果 A18 Pro"…）。
本模块先把字符串归一化，再按「芯片家族 -> 型号档位」的有序规则表查档。

档位是同代跨厂商拉通的相对性能位次（10 = 当前顶级旗舰，1 = 功能机级），
供打分引擎的 performance 维度使用。查不到时返回 DEFAULT_TIER 中性档，不臆造。

规则表基于各 SoC 的公开定位与代际关系（Snapdragon / 天玑 / 麒麟 / Apple A /
Exynos / 紫光展锐 / Helio / Google Tensor），可按需要微调。
"""
from __future__ import annotations

import re

DEFAULT_TIER = 5

# ── 归一化 ──────────────────────────────────────────────────────────────
# 去掉不影响判定的厂商/修饰噪声词，统一大小写与空格，便于正则匹配。
_NOISE_WORDS = (
    "高通", "qualcomm", "联发科", "mediatek", "海思", "hisilicon", "huawei",
    "苹果", "apple", "三星", "samsung", "紫光展锐", "展讯", "spreadtrum",
    "unisoc", "google", "移动平台", "处理器", "芯片",
    "领先版", "巅峰版", "满血版", "超能版", "星速版", "风驰版", "降频版",
    "for galaxy", "monster版", "monster", "elite版",
)


def _normalize(cpu_model: str) -> str:
    text = cpu_model.casefold().strip()
    text = text.replace("（", "(").replace("）", ")")
    for word in _NOISE_WORDS:
        text = text.replace(word, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 各家族有序档位表：(正则, 档位)，从高到低匹配，命中第一个即返回 ──────
# 数字型号有跨厂商重叠（9020 是麒麟、9300 是天玑），所以先分家族再查表。

_SNAPDRAGON: list[tuple[str, int]] = [
    (r"8\s*elite\s*gen\s*5", 10),
    (r"8\s*elite", 10),
    (r"8\s*gen\s*5", 10),
    (r"8\s*gen\s*3|第三代骁龙8", 9),
    (r"8\s*gen\s*2|第二代骁龙8", 9),
    (r"8\s*gen\s*1|第一代骁龙8", 8),
    (r"888\s*plus|888\+", 8),
    (r"888", 8),
    (r"8s\s*gen\s*4", 8),
    (r"8s\s*gen\s*3", 8),
    (r"8\+\s*gen\s*1|8\s*plus\s*gen\s*1", 8),
    (r"8\+\s*4g", 7),
    (r"870", 7),
    (r"865\s*plus|865\+", 7),
    (r"865", 7),
    (r"7\+\s*gen\s*[23]|7\s*plus\s*gen\s*[23]", 8),
    (r"7\s*gen\s*[34]", 6),
    (r"7\s*gen\s*1", 6),
    (r"7s\s*gen\s*[34]", 6),
    (r"第二代骁龙7s|7s\s*gen\s*2", 6),
    (r"78[028]g|778g", 6),
    (r"768g|765g|750g", 5),
    (r"6\s*gen\s*[1345]", 5),
    (r"6s\s*gen\s*4", 4),
    (r"69[05]", 4),
    (r"6\d\d", 4),  # 680 / 662 等
    (r"4\s*gen\s*[12]", 3),
    (r"48[0]\s*plus|480", 3),
    (r"460", 2),
]

_DIMENSITY: list[tuple[str, int]] = [
    (r"9500", 10),
    (r"9400", 9),
    (r"9300", 9),
    (r"9200", 8),
    (r"9000", 8),
    (r"8[45]50", 8),
    (r"8500", 8),
    (r"8400", 8),
    (r"8[23]50", 7),
    (r"8300", 7),
    (r"8250|8200", 7),
    (r"8100", 7),
    (r"8000", 7),
    (r"7500", 6),
    (r"74\d\d|73\d\d|72\d\d|70[25]\d", 6),
    (r"13\d\d|12\d\d|11\d\d|10[08]0|1000", 6),
    (r"9[02]0(?!\d)", 5),  # 920 / 900
    (r"81[05]|800u?|800", 5),
    (r"72[05]|700", 4),
    (r"64\d\d|63\d\d|60[268]0|6020", 4),
]

_KIRIN: list[tuple[str, int]] = [
    (r"9030", 9),
    (r"9020", 8),
    (r"9010", 8),
    (r"9000", 8),
    (r"80[02]0", 7),
    (r"99[05]|985", 7),
    (r"820", 5),
    (r"710", 4),
]

_APPLE: list[tuple[str, int]] = [
    (r"a19\s*pro", 10),
    (r"a19", 9),
    (r"a18\s*pro", 10),
    (r"a18", 9),
    (r"a17\s*pro", 9),
    (r"a16", 9),
    (r"a15", 8),
    (r"a14", 7),
    (r"a13", 6),
    (r"a12", 6),
]

_EXYNOS: list[tuple[str, int]] = [
    (r"2400", 9),
    (r"1480", 6),
    (r"1380", 5),
    (r"1080", 5),
    (r"880", 4),
]

_UNISOC: list[tuple[str, int]] = [
    (r"t8[23]00", 5),
    (r"t760", 4),
    (r"t6\d\d", 3),
]

_HELIO: list[tuple[str, int]] = [
    (r"g8[01]", 3),
    (r"g25", 2),
    (r"p65", 3),
    (r"p35", 2),
]


def _match(normalized: str, table: list[tuple[str, int]]) -> int | None:
    for pattern, tier in table:
        if re.search(pattern, normalized):
            return tier
    return None


def chip_tier(cpu_model: str) -> int:
    """CPU 型号文本 -> 1-10 性能档位分；无法识别时返回 DEFAULT_TIER。"""
    if not cpu_model or not cpu_model.strip():
        return DEFAULT_TIER
    text = _normalize(cpu_model)

    if "tensor" in text:
        return 7  # Google Tensor 系列，旗舰定位但能效受限
    if "mt6260" in text:
        return 1  # 功能机平台

    # 家族识别（关键词命中即在该家族表内查档）
    if re.search(r"骁龙|snapdragon", text) or re.search(r"gen\s*\d", text):
        tier = _match(text, _SNAPDRAGON)
        if tier is not None:
            return tier
    if "天玑" in text or "dimensity" in text:
        tier = _match(text, _DIMENSITY)
        if tier is not None:
            return tier
    if "麒麟" in text or "kirin" in text:
        tier = _match(text, _KIRIN)
        if tier is not None:
            return tier
    if "exynos" in text:
        tier = _match(text, _EXYNOS)
        if tier is not None:
            return tier
    if re.search(r"\ba\d{2}\b", text) or re.search(r"^a\d{2}", text):
        tier = _match(text, _APPLE)
        if tier is not None:
            return tier
    if "helio" in text:
        tier = _match(text, _HELIO)
        if tier is not None:
            return tier
    if re.search(r"虎贲|\bt\d{3,4}\b", text):
        tier = _match(text, _UNISOC)
        if tier is not None:
            return tier

    return DEFAULT_TIER
