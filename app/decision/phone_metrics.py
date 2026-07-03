"""把 ZOL 原始参数（ProductRecord.data）一次性解析成强类型 DTO。

打分逻辑 (app/decision/spec_scoring.py) 只应该读这里解析好的类型化字段
（battery_mah、refresh_hz、aperture_f、ip_rating …），而不是在打分函数里对
原始 dict 反复跑正则。所有「原始文本 -> 数值/枚举」的脏活都收敛在本模块。

字段只承载 ZOL 真实存在、可解析的信息；解析不出来就是 None，绝不臆造。
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.schemas.product_metadata import ProductRecord

OS = Literal["ios", "android"]

IOS_KEYWORDS = ("ios", "iphone", "苹果", "apple")
# 屏幕材质归一化：命中即取该枚举（顺序决定优先级，越靠前越高端）
SCREEN_MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    ("ltpo", "ltpo"),
    ("amoled", "amoled"),
    ("oled", "oled"),
    ("tft", "tft"),
    ("lcd", "lcd"),
]
# 机身材质关键词
GLASS_KEYWORDS = ("玻璃",)
METAL_KEYWORDS = ("金属", "铝", "不锈钢", "钛")
CHINESE_COUNT = {"二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}

# 品牌归一化：型号名里命中关键词即取对应品牌。关键词均为小写（中文 casefold 后不变），
# 列表按「更具体优先」排序，避免子品牌被母品牌吞掉（Redmi/小米、iQOO/vivo、荣耀/华为、一加/OPPO）。
# ZOL 数据没有独立的品牌字段，只能从型号名推断；命中不了就是「未知」，绝不臆造。
BRAND_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Apple", ("iphone", "苹果", "apple", "ipad")),
    ("荣耀", ("荣耀", "honor")),
    ("华为", ("华为", "huawei", "mate", "nova", "畅享", "麦芒", "pura")),
    ("Redmi", ("redmi", "红米")),
    ("小米", ("小米", "xiaomi")),
    ("iQOO", ("iqoo",)),
    ("vivo", ("vivo",)),
    ("一加", ("一加", "oneplus")),
    ("OPPO", ("oppo",)),
    ("realme", ("真我", "realme")),
    ("三星", ("三星", "samsung", "galaxy", "sm-")),
    ("努比亚", ("努比亚", "nubia", "红魔", "redmagic")),
    ("魅族", ("魅族", "meizu")),
    ("moto", ("moto", "motorola")),
    ("中兴", ("中兴", "zte")),
    ("联想", ("联想", "lenovo", "拯救者")),
    ("谷歌", ("谷歌", "pixel", "google")),
    ("诺基亚", ("诺基亚", "nokia")),
    ("金立", ("金立",)),
    ("VERTU", ("vertu",)),
    ("WIKO", ("wiko",)),
    ("HTC", ("htc",)),
    ("亮见", ("亮见",)),
]


def _detect_brand(name: str) -> str:
    lowered = name.casefold()
    for brand, keywords in BRAND_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return brand
    return "未知"


def _first_int(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text.replace(",", ""), re.IGNORECASE)
    if match and match.group(1):
        return int(match.group(1))
    return None


def _first_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text.replace(",", ""), re.IGNORECASE)
    if match and match.group(1):
        return float(match.group(1))
    return None


def _pick(data: dict[str, Any], *keys: str) -> str:
    """按优先级取第一个非空字段（params 优先，其次顶层）。"""
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    for key in keys:
        value = params.get(key)
        if value is None:
            value = data.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _parse_screen_material(text: str) -> str:
    lowered = text.casefold()
    for keyword, canonical in SCREEN_MATERIAL_KEYWORDS:
        if keyword in lowered:
            return canonical
    return ""


def _parse_main_megapixels(text: str) -> Optional[int]:
    """从 '后置摄像头1：5000万像素 ...' 或 '2亿像素' 抽最高万像素数。"""
    best: Optional[int] = None
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(亿|万)?\s*像素", text):
        num = float(match.group(1))
        wan = num * 10000 if match.group(2) == "亿" else num
        best = int(wan) if best is None else max(best, int(wan))
    return best


def _parse_aperture(text: str) -> Optional[float]:
    """取最大光圈（f 值最小）。'后置摄像头1：f/1.4~f/4.0 ...' -> 1.4。"""
    values = [float(v) for v in re.findall(r"f/\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)]
    return min(values) if values else None


def _parse_camera_count(text: str) -> Optional[int]:
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"([二三四五六七八])\s*摄", text)
    return CHINESE_COUNT.get(match.group(1)) if match else None


def _parse_ip_rating(text: str) -> Optional[int]:
    ratings = [int(v) for v in re.findall(r"IP(\d{2})", text, re.IGNORECASE)]
    return max(ratings) if ratings else None


class PhoneMetrics(BaseModel):
    """从 ZOL 原始参数解析出的强类型手机指标。解析不出即 None。"""

    # 身份 / 元信息
    brand: str
    os: OS
    price_yuan: int
    release_year: Optional[int]
    # 续航
    battery_mah: Optional[int]
    wired_charge_w: Optional[int]
    # 屏幕
    refresh_hz: Optional[int]
    screen_material: str  # ltpo / amoled / oled / lcd / tft / ""
    # 影像
    main_camera_mp: Optional[int]  # 万像素
    aperture_f: Optional[float]
    # None=无防抖字段(未知) / True=OIS光学防抖 / False=有防抖但非光学(如仅EIS)
    optical_stabilization: Optional[bool]
    camera_count: Optional[int]
    # 性能
    cpu_model: str
    # 存储 / 便携
    storage_gb: int
    ram_gb: int
    weight_g: int
    # 机身耐用度（三防 + 材质）
    ip_rating: Optional[int]
    glass_back: Optional[bool]
    metal_frame: Optional[bool]

    @classmethod
    def from_record(cls, record: ProductRecord) -> "PhoneMetrics":
        data = record.data
        name = _pick(data, "title", "name", "产品型号") or record.name
        material_text = _pick(data, "机身材质")
        return cls(
            brand=_detect_brand(name),
            os=cls._detect_os(data, name),
            price_yuan=_first_int(_pick(data, "电商报价", "price_text", "参考价"), r"(\d+)") or 0,
            release_year=_first_int(
                _pick(data, "国内发布时间", "上市日期", "国外发布时间"), r"(20[12]\d)"
            ),
            battery_mah=_first_int(_pick(data, "电池容量"), r"(\d{3,5})\s*mAh"),
            wired_charge_w=_first_int(_pick(data, "有线充电", "充电功率"), r"(\d+)\s*w"),
            refresh_hz=_first_int(_pick(data, "屏幕刷新率"), r"(\d+)\s*Hz"),
            screen_material=_parse_screen_material(_pick(data, "屏幕材质")),
            main_camera_mp=_parse_main_megapixels(_pick(data, "像素")),
            aperture_f=_parse_aperture(_pick(data, "光圈")),
            optical_stabilization=cls._parse_ois(_pick(data, "防抖功能")),
            camera_count=_parse_camera_count(_pick(data, "摄像头总数")),
            cpu_model=_pick(data, "CPU型号", "处理器"),
            storage_gb=cls._parse_storage(_pick(data, "ROM容量", "存储容量")),
            ram_gb=_first_int(_pick(data, "RAM容量", "运行内存"), r"(\d+)\s*GB") or 0,
            weight_g=_first_int(_pick(data, "重量"), r"(\d+)\s*g") or 0,
            ip_rating=_parse_ip_rating(_pick(data, "三防功能")),
            glass_back=cls._has_glass_back(material_text),
            metal_frame=cls._has_metal_frame(material_text),
        )

    @staticmethod
    def _detect_os(data: dict[str, Any], name: str) -> OS:
        haystack = " ".join(
            [_pick(data, "操作系统"), _pick(data, "出厂系统内核"), name]
        ).casefold()
        return "ios" if any(k in haystack for k in IOS_KEYWORDS) else "android"

    @staticmethod
    def _parse_ois(text: str) -> Optional[bool]:
        if not text.strip():
            return None
        return "OIS" in text.upper()

    @staticmethod
    def _parse_storage(text: str) -> int:
        tb = _first_int(text, r"(\d+)\s*TB")
        if tb:
            return tb * 1024
        return _first_int(text, r"(\d+)\s*GB") or 0

    @staticmethod
    def _has_glass_back(material_text: str) -> Optional[bool]:
        if not material_text:
            return None
        return any(k in material_text for k in GLASS_KEYWORDS)

    @staticmethod
    def _has_metal_frame(material_text: str) -> Optional[bool]:
        if not material_text:
            return None
        return any(k in material_text for k in METAL_KEYWORDS)
