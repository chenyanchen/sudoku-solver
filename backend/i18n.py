"""The site's two languages: English at /, Simplified Chinese at /zh/.

Each language is its own URL (with hreflang alternates), so search engines
index both. Pages are rendered once from frontend/index.html; `{{key}}` is
replaced with the HTML-escaped text, and the frontend's own strings are
injected as JSON.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from functools import cache
from pathlib import Path

BASE_URL = os.getenv("SUDOKU_BASE_URL", "https://sudokusolver.dpdns.org").rstrip("/")
PATHS = {"en": "/", "zh": "/zh/"}
_TEMPLATE = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "html_lang": "en",
        "og_locale": "en_US",
        "other_lang_name": "中文",
        "title": "Sudoku Solver – Solve a Sudoku from a Photo or Screenshot",
        "description": "Upload a photo or screenshot of a Sudoku puzzle and get the solution in seconds. "
                       "Fix any misread digit and solve again. Free, no sign-up.",
        "h1": "Sudoku Solver",
        "subtitle": "Upload a photo or screenshot of a Sudoku puzzle to solve it",
        "upload_text": "Click to upload or drag and drop",
        "upload_hint": "JPG, PNG or WEBP, up to 10 MB",
        "storage_notice": "Uploaded images are stored and used only to improve recognition.",
        "original_image": "Your image",
        "detected_grid": "Detected grid",
        "solve_button": "Solve again",
        "solving": "Solving…",
        "recognized_puzzle": "Recognized puzzle",
        "fix_hint": "Misread a digit? Click a cell to fix it, then solve again.",
        "resolve_button": "Solve with my corrections",
        "solution": "Solution",
        "how_title": "How it works",
        "how_1": "Take a photo of a Sudoku in a newspaper or book, or a screenshot of a Sudoku app or website.",
        "how_2": "Upload it. The solver finds the grid, reads the digits and solves the puzzle.",
        "how_3": "Check the recognized digits against your image. If one is wrong, fix it and solve again.",
        "faq_title": "FAQ",
        "faq_free_q": "Is it free?",
        "faq_free_a": "Yes. No sign-up, no ads.",
        "faq_images_q": "Which images work?",
        "faq_images_a": "Screenshots and photos of printed puzzles, including photos taken at an angle or of a "
                        "screen. The whole 9×9 grid must be visible and in focus. Handwritten digits are not "
                        "read reliably yet.",
        "faq_wrong_q": "Why is a digit wrong?",
        "faq_wrong_a": "Glare, blur or an unusual font can make a digit hard to read. A puzzle is solved only "
                       "when its digits have exactly one solution, so a misread usually shows up as "
                       "\"no solution\" rather than a wrong answer. Fix the cell and solve again; your "
                       "correction helps improve recognition.",
        "faq_rate_q": "What is the solve rate?",
        "faq_rate_a": "The share of distinct uploaded images that were solved without corrections. We publish "
                      "it so you can watch recognition improve.",
        "faq_privacy_q": "What do you store?",
        "faq_privacy_a": "Your image, whether it was solved, how long it took, and any digits you corrected. "
                         "No IP address and no file name. Images are used only to improve recognition "
                         "and are never published.",
        "footer": "Recognition: OpenCV, a grid-corner CNN and a digit CNN, running on our server.",
    },
    "zh": {
        "html_lang": "zh-Hans",
        "og_locale": "zh_CN",
        "other_lang_name": "English",
        "title": "数独求解器 – 拍照或截图即可解数独",
        "description": "上传数独的照片或截图，几秒内得到答案。识别错的数字可以直接改正后重新求解。免费，无需注册。",
        "h1": "数独求解器",
        "subtitle": "上传数独的照片或截图，立即求解",
        "upload_text": "点击上传，或把图片拖到这里",
        "upload_hint": "支持 JPG、PNG、WEBP，最大 10 MB",
        "storage_notice": "上传的图片会被保存，仅用于改进识别。",
        "original_image": "你的图片",
        "detected_grid": "识别到的网格",
        "solve_button": "重新求解",
        "solving": "求解中…",
        "recognized_puzzle": "识别出的题目",
        "fix_hint": "有数字识别错了？点击格子改正，再重新求解。",
        "resolve_button": "按我的修改求解",
        "solution": "答案",
        "how_title": "使用方法",
        "how_1": "拍下报纸或书上的数独，或者截取数独 App、网站的屏幕。",
        "how_2": "上传图片。求解器会找到网格、识别数字并求解。",
        "how_3": "对照原图检查识别出的数字。如有错误，改正后重新求解。",
        "faq_title": "常见问题",
        "faq_free_q": "免费吗？",
        "faq_free_a": "免费，无需注册，没有广告。",
        "faq_images_q": "什么样的图片可以识别？",
        "faq_images_a": "截图，以及印刷数独的照片，包括斜拍的照片和翻拍屏幕的照片。需要完整拍到 9×9 网格并且清晰对焦。"
                        "手写数字暂时还不能可靠识别。",
        "faq_wrong_q": "为什么有数字识别错了？",
        "faq_wrong_a": "反光、模糊或少见的字体都会让数字难以识别。只有当识别出的数字恰好有唯一解时才会给出答案，"
                       "所以识别错误通常表现为“无解”，而不是给出错误答案。改正那个格子后重新求解即可，"
                       "你的修改也会帮助我们改进识别。",
        "faq_rate_q": "解出率是什么？",
        "faq_rate_a": "所有上传过的不同图片中，无需修改就被解出的比例。我们公开这个数字，你可以看到识别能力的进步。",
        "faq_privacy_q": "你们会保存什么？",
        "faq_privacy_a": "你上传的图片、是否解出、耗时，以及你改正过的数字。不记录 IP 地址和文件名。"
                         "图片只用于改进识别，不会公开。",
        "footer": "识别：OpenCV、网格角点 CNN 与数字 CNN，运行在我们的服务器上。",
    },
}

# Strings the frontend shows; {name} placeholders are filled in by app.js.
JS_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "stats": "Solved {solved} of {total} uploaded puzzles ({percentage}%)",
        "notImage": "Please select an image file",
        "tooLarge": "Image larger than 10 MB",
        "noImage": "Please select an image first",
        "uploadFailed": "Upload failed ({status})",
        "noGrid": "Could not find a Sudoku grid in the image. Make sure the whole grid is visible.",
        "solved": "Puzzle solved!",
        "misread": "Some digits may be misread: compare with your image, fix the cells, then solve again.",
        "solvedCorrected": "Puzzle solved with your corrections. Thank you!",
        "digitsOnly": "Cells accept only digits 1–9 (leave blank for empty).",
        "noSolution": "These digits have no solution. Check them against your image.",
        "multipleSolutions": "These digits have more than one solution. Check for a missed digit.",
        "partial": "These digits have {count} solutions; here is one. The {open} highlighted cells can take "
        "other digits in the other solutions. If your puzzle has a single answer, check for a missed digit.",
        "invalid": "Some digits repeat in a row, column or box. Check them against your image.",
        "network": "Could not reach the server: {error}",
        "cell": "Row {row}, column {col}",
    },
    "zh": {
        "stats": "已解出 {solved} / {total} 道上传的题目（{percentage}%）",
        "notImage": "请选择图片文件",
        "tooLarge": "图片超过 10 MB",
        "noImage": "请先选择一张图片",
        "uploadFailed": "上传失败（{status}）",
        "noGrid": "图片中没有找到数独网格，请确保完整拍到整个网格。",
        "solved": "求解成功！",
        "misread": "部分数字可能识别错了：请对照原图，改正格子后重新求解。",
        "solvedCorrected": "已按你的修改求解，谢谢！",
        "digitsOnly": "格子只能填 1–9（空格留空）。",
        "noSolution": "这些数字无解，请对照原图检查。",
        "multipleSolutions": "这些数字有多个解，请检查是否漏了数字。",
        "partial": "这些数字有 {count} 个解，下面是其中一个。标黄的 {open} 个格子在其他解里可以填别的数字。"
        "如果你的题目应该只有一个答案，请检查是否漏了数字。",
        "invalid": "有数字在同一行、列或宫里重复了，请对照原图检查。",
        "network": "无法连接服务器：{error}",
        "cell": "第 {row} 行，第 {col} 列",
    },
}


def _json_ld(lang: str) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": TEXT[lang]["h1"],
        "url": BASE_URL + PATHS[lang],
        "description": TEXT[lang]["description"],
        "applicationCategory": "GameApplication",
        "operatingSystem": "Any",
        "inLanguage": TEXT[lang]["html_lang"],
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    }
    return _script_json(data)


def _script_json(data: object) -> str:
    """JSON safe inside a <script> element."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


@cache
def render(lang: str) -> str:
    other = "zh" if lang == "en" else "en"
    values = {key: html.escape(value) for key, value in TEXT[lang].items()}
    values.update(
        url=BASE_URL + PATHS[lang],
        url_en=BASE_URL + PATHS["en"],
        url_zh=BASE_URL + PATHS["zh"],
        other_url=PATHS[other],
        other_lang=TEXT[other]["html_lang"],
        json_ld=_json_ld(lang),
        js_strings=_script_json(JS_TEXT[lang]),
        asset_version=_asset_version(),
    )
    page = _TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise ValueError(f"Unfilled placeholder in {_TEMPLATE}: {page[page.index('{{'):][:40]}")
    return page


def _asset_version() -> str:
    """Content hash for /static URLs: Cloudflare caches them for hours, so a
    changed file needs a new URL to reach visitors."""
    digest = hashlib.sha256()
    for name in ("app.js", "style.css"):
        digest.update((_TEMPLATE.parent / name).read_bytes())
    return digest.hexdigest()[:12]


def sitemap() -> str:
    alternates = "".join(
        f'<xhtml:link rel="alternate" hreflang="{TEXT[lang]["html_lang"]}" href="{BASE_URL + path}"/>'
        for lang, path in PATHS.items()
    )
    urls = "".join(f"<url><loc>{BASE_URL + path}</loc>{alternates}</url>" for path in PATHS.values())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        f'xmlns:xhtml="http://www.w3.org/1999/xhtml">{urls}</urlset>\n'
    )


def robots() -> str:
    return f"User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {BASE_URL}/sitemap.xml\n"
