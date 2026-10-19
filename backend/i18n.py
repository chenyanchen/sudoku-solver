"""The site's two languages: English at /, Simplified Chinese at /zh/.

Each language is its own URL (with hreflang alternates), so search engines
index both. The solver page is rendered once from frontend/index.html;
`{{key}}` is replaced with the HTML-escaped text, and the frontend's own
strings are injected as JSON. Articles are frontend/article.html around a
body from frontend/articles/<slug>.<lang>.html, kept as HTML.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from functools import cache
from pathlib import Path

# The public site: the static pages on GitHub Pages (scripts/build_static_site.py).
# Canonical URLs, the sitemap and promotion all name it.
BASE_URL = os.getenv("SUDOKU_BASE_URL", "https://chenyanchen.github.io/sudoku-solver").rstrip("/")
# Our server: the API behind those pages. Its own page URLs redirect to BASE_URL.
API_URL = os.getenv("SUDOKU_API_URL", "https://sudokusolver.dpdns.org").rstrip("/")
# Cloudflare Web Analytics for the static pages (the server's pages got the
# beacon from Cloudflare's proxy); empty: no beacon.
CF_BEACON_TOKEN = ""
PATHS = {"en": "/", "zh": "/zh/"}
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
_TEMPLATE = _FRONTEND / "index.html"

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "html_lang": "en",
        "og_locale": "en_US",
        "other_lang_name": "中文",
        "title": "Sudoku Solver – Solve a Sudoku from a Photo or Screenshot",
        "description": "Upload a photo or screenshot of a Sudoku puzzle, or type it in, and get the solution in "
                       "seconds. Fix any misread digit and solve again. Free, no sign-up.",
        "h1": "Sudoku Solver",
        "subtitle": "Upload a photo or screenshot of a Sudoku puzzle to solve it",
        "upload_text": "Click to upload or drag and drop",
        "upload_hint": "JPG, PNG or WEBP, up to 10 MB",
        "storage_notice": "Uploaded images are stored and used only to improve recognition.",
        "manual_button": "No image? Type in the puzzle",
        "share_button": "Share the solution",
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
        "faq_wrong_a": "Glare, blur or an unusual font can make a digit hard to read. Compare the recognized "
                       "puzzle with your image; a misread digit often leaves it with no solution. Fix the "
                       "cell and solve again; your correction helps improve recognition.",
        "faq_rate_q": "What is the solve rate?",
        "faq_rate_a": "The share of distinct uploaded images that were solved without corrections. We publish "
                      "it so you can watch recognition improve.",
        "faq_privacy_q": "What do you store?",
        "faq_privacy_a": "Your image, whether it was solved, how long it took, and any digits you corrected. "
                         "No IP address and no file name. Images are used only to improve recognition "
                         "and are never published.",
        "footer": "Recognition: OpenCV, a grid-corner CNN and a digit CNN, running on our server.",
        "how_link": "How the recognition works",
    },
    "zh": {
        "html_lang": "zh-Hans",
        "og_locale": "zh_CN",
        "other_lang_name": "English",
        "title": "数独求解器 – 拍照或截图即可解数独",
        "description": "上传数独的照片或截图，或者手动输入题目，几秒内得到答案。识别错的数字可以直接改正后重新求解。免费，无需注册。",
        "h1": "数独求解器",
        "subtitle": "上传数独的照片或截图，立即求解",
        "upload_text": "点击上传，或把图片拖到这里",
        "upload_hint": "支持 JPG、PNG、WEBP，最大 10 MB",
        "storage_notice": "上传的图片会被保存，仅用于改进识别。",
        "manual_button": "没有图片？手动输入题目",
        "share_button": "分享答案",
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
        "faq_wrong_a": "反光、模糊或少见的字体都会让数字难以识别。请对照原图检查识别出的题目，"
                       "识别错的数字常常会让题目无解。改正那个格子后重新求解即可，你的修改也会帮助我们改进识别。",
        "faq_rate_q": "解出率是什么？",
        "faq_rate_a": "所有上传过的不同图片中，无需修改就被解出的比例。我们公开这个数字，你可以看到识别能力的进步。",
        "faq_privacy_q": "你们会保存什么？",
        "faq_privacy_a": "你上传的图片、是否解出、耗时，以及你改正过的数字。不记录 IP 地址和文件名。"
                         "图片只用于改进识别，不会公开。",
        "footer": "识别：OpenCV、网格角点 CNN 与数字 CNN，运行在我们的服务器上。",
        "how_link": "识别是怎么做的",
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
        "solvedCorrected": "Puzzle solved with your corrections. Thank you!",
        "digitsOnly": "Cells accept only digits 1–9 (leave blank for empty).",
        "noSolution": "These digits have no solution. Check them against your image.",
        "invalid": "Some digits repeat in a row, column or box. Check them against your image.",
        "network": "Could not reach the server: {error}",
        "cell": "Row {row}, column {col}",
        "manualTitle": "Your puzzle",
        "manualHint": "Type the given digits; arrow keys move between cells.",
        "manualSolve": "Solve",
        "recognizedTitle": "Recognized puzzle",
        "recognizedHint": "Misread a digit? Click a cell to fix it, then solve again.",
        "resolve": "Solve with my corrections",
        "tooFewGivens": "Enter at least 17 digits: a puzzle with fewer has more than one solution.",
        "shareText": "Solved with Sudoku Solver: photo in, solution out.",
        "shareFile": "sudoku-solution.png",
        "brand": "Sudoku Solver",
    },
    "zh": {
        "stats": "已解出 {solved} / {total} 道上传的题目（{percentage}%）",
        "notImage": "请选择图片文件",
        "tooLarge": "图片超过 10 MB",
        "noImage": "请先选择一张图片",
        "uploadFailed": "上传失败（{status}）",
        "noGrid": "图片中没有找到数独网格，请确保完整拍到整个网格。",
        "solved": "求解成功！",
        "solvedCorrected": "已按你的修改求解，谢谢！",
        "digitsOnly": "格子只能填 1–9（空格留空）。",
        "noSolution": "这些数字无解，请对照原图检查。",
        "invalid": "有数字在同一行、列或宫里重复了，请对照原图检查。",
        "network": "无法连接服务器：{error}",
        "cell": "第 {row} 行，第 {col} 列",
        "manualTitle": "你的题目",
        "manualHint": "输入题目给出的数字，方向键可在格子间移动。",
        "manualSolve": "求解",
        "recognizedTitle": "识别出的题目",
        "recognizedHint": "有数字识别错了？点击格子改正，再重新求解。",
        "resolve": "按我的修改求解",
        "tooFewGivens": "请至少输入 17 个数字：少于 17 个的题目不止一个解。",
        "shareText": "用数独求解器解出来的：拍张照，答案就出来了。",
        "shareFile": "sudoku-solution.png",
        "brand": "数独求解器",
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
def render(lang: str, prefix: str = "", api_base: str = "") -> str:
    """The solver page. `prefix` is the path the site is served under and
    `api_base` the API's origin: both empty on our server; the static copy on
    GitHub Pages (scripts/build_static_site.py) lives under /sudoku-solver and
    calls this site's API. Canonical URLs always name this site."""
    other = "zh" if lang == "en" else "en"
    values = {key: html.escape(value) for key, value in TEXT[lang].items()}
    values.update(
        url=BASE_URL + PATHS[lang],
        url_en=BASE_URL + PATHS["en"],
        url_zh=BASE_URL + PATHS["zh"],
        other_url=prefix + PATHS[other],
        other_lang=TEXT[other]["html_lang"],
        how_url=prefix + ARTICLE_PATHS["how-it-works"][lang],
        manifest_url=prefix + PATHS[lang] + "manifest.webmanifest",
        static=prefix + "/static",
        json_ld=_json_ld(lang),
        js_strings=_script_json(JS_TEXT[lang]),
        site=_script_json({"prefix": prefix, "api": api_base}),
        analytics=_analytics(),
        asset_version=_asset_version(),
    )
    return _fill(_TEMPLATE, values)


def _analytics() -> str:
    if not CF_BEACON_TOKEN:
        return ""
    beacon = html.escape(json.dumps({"token": CF_BEACON_TOKEN}))
    return (f'<script defer src="https://static.cloudflareinsights.com/beacon.min.js" '
            f'data-cf-beacon="{beacon}"></script>')


# Articles: their own URL per language, listed in the sitemap.
ARTICLE_PATHS = {
    "how-it-works": {"en": "/how-it-works", "zh": "/zh/how-it-works"},
}

ARTICLE_TEXT: dict[str, dict[str, dict[str, str]]] = {
    "how-it-works": {
        "en": {
            "title": "How a Photo Becomes a Solved Sudoku – Sudoku Solver",
            "description": "Grid detection with OpenCV and a corner-heatmap CNN, a 38K-parameter digit CNN, "
                           "and the unique solution as the acceptance test: how Sudoku Solver reads photos "
                           "and screenshots.",
            "back": "Solve a Sudoku",
        },
        "zh": {
            "title": "一张照片如何变成解好的数独 – 数独求解器",
            "description": "OpenCV 与角点热力图 CNN 找网格，3.8 万参数的数字 CNN 读数字，用唯一解作为验收标准："
                           "数独求解器是怎样识别照片和截图的。",
            "back": "去解一道数独",
        },
    },
}


@cache
def render_article(slug: str, lang: str, prefix: str = "") -> str:
    other = "zh" if lang == "en" else "en"
    paths = ARTICLE_PATHS[slug]
    text = ARTICLE_TEXT[slug][lang]
    values = {key: html.escape(value) for key, value in text.items()}
    values.update(
        html_lang=TEXT[lang]["html_lang"],
        og_locale=TEXT[lang]["og_locale"],
        url=BASE_URL + paths[lang],
        url_en=BASE_URL + paths["en"],
        url_zh=BASE_URL + paths["zh"],
        other_url=prefix + paths[other],
        other_lang=TEXT[other]["html_lang"],
        other_lang_name=TEXT[lang]["other_lang_name"],
        home_url=prefix + PATHS[lang],
        static=prefix + "/static",
        footer=html.escape(TEXT[lang]["footer"]),
        json_ld=_script_json({
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": text["title"],
            "description": text["description"],
            "url": BASE_URL + paths[lang],
            "inLanguage": TEXT[lang]["html_lang"],
            "author": {"@type": "Person", "name": "Yanchen Chen"},
        }),
        analytics=_analytics(),
        asset_version=_asset_version(),
        body=(_FRONTEND / "articles" / f"{slug}.{lang}.html").read_text(encoding="utf-8")
        .replace("{{static}}", prefix + "/static"),
    )
    return _fill(_FRONTEND / "article.html", values)


def _fill(template: Path, values: dict[str, str]) -> str:
    page = template.read_text(encoding="utf-8")
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    if "{{" in page:
        raise ValueError(f"Unfilled placeholder in {template}: {page[page.index('{{'):][:40]}")
    return page


def _asset_version() -> str:
    """Content hash for /static URLs: Cloudflare caches them for hours, so a
    changed file needs a new URL to reach visitors."""
    digest = hashlib.sha256()
    for name in ("app.js", "style.css"):
        digest.update((_TEMPLATE.parent / name).read_bytes())
    return digest.hexdigest()[:12]


def manifest(lang: str, prefix: str = "") -> dict:
    """Web app manifest: installable, and a share target, so a screenshot can
    be shared straight to the solver (frontend/sw.js receives it)."""
    home = prefix + PATHS[lang]
    return {
        "name": TEXT[lang]["h1"],
        "short_name": TEXT[lang]["h1"],
        "description": TEXT[lang]["description"],
        "lang": TEXT[lang]["html_lang"],
        "id": home,
        "start_url": home,
        "scope": prefix + "/",
        "display": "standalone",
        "background_color": "#f8fafc",
        "theme_color": "#3b82f6",
        "icons": [
            {"src": prefix + "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": prefix + "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": prefix + "/static/icon-maskable-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "maskable"},
        ],
        "share_target": {
            "action": home + "share-target",
            "method": "POST",
            "enctype": "multipart/form-data",
            "params": {"files": [{"name": "image", "accept": ["image/*"]}]},
        },
    }


def sitemap() -> str:
    pages = [PATHS, *ARTICLE_PATHS.values()]
    urls = ""
    for paths in pages:
        alternates = "".join(
            f'<xhtml:link rel="alternate" hreflang="{TEXT[lang]["html_lang"]}" href="{BASE_URL + path}"/>'
            for lang, path in paths.items()
        )
        urls += "".join(f"<url><loc>{BASE_URL + path}</loc>{alternates}</url>" for path in paths.values())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        f'xmlns:xhtml="http://www.w3.org/1999/xhtml">{urls}</urlset>\n'
    )


# IndexNow (Bing, Yandex, …): the key is public by design; search engines
# fetch it from /<key>.txt to verify that URL submissions come from the site.
INDEXNOW_KEY = "0ce5f78acfb90d59db3a0b62f6d71f3d"


def all_urls() -> list[str]:
    return [BASE_URL + path for paths in (PATHS, *ARTICLE_PATHS.values()) for path in paths.values()]


def robots() -> str:
    """The server's robots.txt: its pages redirect to BASE_URL, whose sitemap
    is listed by the robots.txt at the root of chenyanchen.github.io."""
    return f"User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {BASE_URL}/sitemap.xml\n"
