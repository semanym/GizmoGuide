"""ZOL (中关村在线) 手机参数爬虫。

只抓硬件参数详情 + 电商报价，原样保留参数页的全部 key-value，
不裁剪成业务 schema（产品结构后续会重新设计，先把原始数据落盘）。

数据流：
    列表页 -> 详情页链接(/cell_phone/index{ID}.shtml)
          -> 详情页提取 series id -> 参数页(/{series}/{ID}/param.shtml)
          -> 解析 <tr><th>名</th><td>值</td> 表格
          -> 价格取详情页 .price-type（权威），参数页「电商报价」兜底

站点特征：
    - 全站 GBK 编码
    - 反爬：带 Referer 即可，无验证码（请控制频率）
    - 价格不在参数页，而在详情页的 .price-type 价格区（纯数字）

用法：
    python scripts/scrape/zol_specs.py --limit 60
    python scripts/scrape/zol_specs.py --pages 3 --out scripts/data/zol_specs_raw.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://detail.zol.com.cn"
LIST_URL = BASE + "/cell_phone_index/subcate57_0_list_1_0_1_2_0_{page}.html"
HOME_URL = BASE + "/cell_phone/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

DETAIL_RE = re.compile(r"/cell_phone/index(\d+)\.shtml")
SERIES_RE = re.compile(r"/(\d+)/(\d+)/param\.shtml")

# 详情页价格容器：.price-type 稳定返回纯数字价格（.price 带“开售”噪音，
# span.price 是对比串价）。参数页的「电商报价」只有部分机型有，价格真正的
# 权威来源是详情页，因此以详情页为主、参数页为兜底。
PRICE_SELECTOR = ".price-type"
PRICE_RE = re.compile(r"(\d{3,7})")


def fetch(url: str, referer: str = HOME_URL) -> str | None:
    """抓取并按 GBK 解码。失败返回 None。"""
    headers = dict(HEADERS, Referer=referer)
    try:
        resp = requests.get(url, headers=headers, timeout=25)
    except requests.RequestException as exc:
        print(f"  [net error] {url} -> {exc}")
        return None
    if resp.status_code != 200:
        print(f"  [http {resp.status_code}] {url}")
        return None
    html = resp.content.decode("gbk", errors="ignore")
    if "service.zol.com.cn/checking" in html:
        print(f"  [anti-bot] {url}")
        return None
    return html


def collect_detail_ids(pages: int) -> list[str]:
    """从列表页（含首页兜底）收集详情页产品 ID。"""
    ids: list[str] = []
    seen: set[str] = set()

    def absorb(html: str | None) -> None:
        if not html:
            return
        for pid in DETAIL_RE.findall(html):
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)

    # 首页通常稳定返回热门机型，作为兜底
    absorb(fetch(HOME_URL))
    for page in range(1, pages + 1):
        url = LIST_URL.format(page=page)
        print(f"[list] page {page}: {url}")
        absorb(fetch(url, referer=HOME_URL))
        time.sleep(random.uniform(1.0, 2.0))
    return ids


def find_param_url(detail_html: str, pid: str) -> str | None:
    """从详情页 HTML 提取参数页绝对 URL。"""
    m = SERIES_RE.search(detail_html)
    if not m:
        return None
    series, found_pid = m.group(1), m.group(2)
    if found_pid != pid:
        # 详情页里可能混入对比机型链接，按当前 pid 兜底拼接
        return f"{BASE}/{series}/{pid}/param.shtml"
    return f"{BASE}/{series}/{pid}/param.shtml"


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.replace("纠错", "").replace("\xa0", " ").strip()


def parse_params(param_html: str) -> dict[str, str]:
    """解析参数页表格为 {参数名: 参数值}，原样保留所有字段。"""
    soup = BeautifulSoup(param_html, "html.parser")
    params: dict[str, str] = {}
    for tr in soup.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        name = clean(th.get_text(" ", strip=True))
        # 去掉值里的“查看外观图/更多…”等导航锚文本噪音
        for a in td.find_all("a"):
            href = a.get("href", "")
            anchor = a.get_text()
            if (
                any(k in href for k in ("/pic", "paihang", "/sj/", "more"))
                or "更多" in anchor
                or "排行" in anchor
                or "查看" in anchor
            ):
                a.extract()
        value = clean(td.get_text(" ", strip=True))
        # 清理残留的导航箭头与尾随标点
        value = re.sub(r"\s*[>＞]\s*", " ", value).strip(" ，,、")
        value = re.sub(r"\s+", " ", value).strip()
        if name and value:
            params[name] = value
    return params


def extract_title(detail_html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", detail_html, re.S)
    if not m:
        return ""
    raw = m.group(1).strip()
    # 优先：形如 【vivo X Fold6 12GB/256GB】报价_参数_…
    inner = re.search(r"【(.*?)】", raw)
    if inner:
        return inner.group(1).strip()
    # 兜底：形如 小米MIX FOLD 4报价_参数_图片… → 截断营销后缀
    title = re.split(r"报价|_参数|参数_|怎么样|\(参考价", raw)[0]
    return title.strip()


def extract_price(detail_html: str, params: dict[str, str]) -> str:
    """提取电商报价，返回 '￥6499' 形式；取不到返回空串。

    价格权威来源是「详情页」的 .price-type（纯数字）；参数页的「电商报价」
    只有部分机型有，作为兜底。个别机型详情页写「价格面议」无数字，此时也兜底
    到参数页；都取不到就留空（绝不臆造价格）。
    """
    soup = BeautifulSoup(detail_html, "html.parser")
    el = soup.select_one(PRICE_SELECTOR)
    if el:
        match = PRICE_RE.search(el.get_text(" ", strip=True))
        if match:
            return f"￥{match.group(1)}"
    return params.get("电商报价", "")


def scrape_one(pid: str) -> dict | None:
    detail_url = f"{BASE}/cell_phone/index{pid}.shtml"
    detail_html = fetch(detail_url)
    if not detail_html:
        return None
    title = extract_title(detail_html)
    param_url = find_param_url(detail_html, pid)
    if not param_url:
        print(f"  [no param url] {pid} ({title})")
        return None
    param_html = fetch(param_url, referer=detail_url)
    if not param_html:
        return None
    params = parse_params(param_html)
    if not params:
        print(f"  [empty params] {pid} ({title})")
        return None
    price_text = extract_price(detail_html, params)
    return {
        "zol_id": pid,
        "title": title,
        "detail_url": detail_url,
        "param_url": param_url,
        "price_text": price_text,
        "params": params,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2, help="列表页翻页数")
    ap.add_argument("--limit", type=int, default=60, help="最多抓取机型数")
    ap.add_argument(
        "--out",
        default="scripts/data/zol_specs_raw.json",
        help="输出 JSON 路径",
    )
    args = ap.parse_args()

    ids = collect_detail_ids(args.pages)[: args.limit]
    print(f"\n收集到 {len(ids)} 个机型 ID，开始抓参数…\n")

    results: list[dict] = []
    for i, pid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] id={pid}")
        record = scrape_one(pid)
        if record:
            print(f"  OK: {record['title']} | {len(record['params'])} 参数 | {record['price_text']}")
            results.append(record)
        time.sleep(random.uniform(1.2, 2.5))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成：{len(results)}/{len(ids)} 款写入 {out_path}")


if __name__ == "__main__":
    main()
