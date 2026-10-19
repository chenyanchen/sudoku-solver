"""Bilingual pages and the files search engines read."""

import re

from fastapi.testclient import TestClient

from backend import i18n
from backend.main import app

client = TestClient(app)


def test_languages_translate_the_same_strings():
    assert i18n.TEXT["en"].keys() == i18n.TEXT["zh"].keys()
    assert i18n.JS_TEXT["en"].keys() == i18n.JS_TEXT["zh"].keys()


def test_each_language_has_its_own_url_with_alternates():
    en, zh = client.get("/"), client.get("/zh/")
    assert en.status_code == zh.status_code == 200
    assert '<html lang="en">' in en.text and '<html lang="zh-Hans">' in zh.text
    for page, url in ((en.text, i18n.BASE_URL + "/"), (zh.text, i18n.BASE_URL + "/zh/")):
        assert f'<link rel="canonical" href="{url}">' in page
        assert 'hreflang="x-default"' in page and "{{" not in page
    assert "数独求解器" in zh.text and "window.I18N" in zh.text
    redirect = client.get("/zh", follow_redirects=False)
    assert redirect.status_code == 301 and redirect.headers["location"] == "/zh/"



def test_static_urls_carry_a_content_hash():
    # Cloudflare caches /static for hours; a new hash is a new URL.
    page = client.get("/").text
    version = i18n._asset_version()
    assert f'src="/static/app.js?v={version}"' in page and f'href="/static/style.css?v={version}"' in page
    assert client.get(f"/static/app.js?v={version}").status_code == 200


def test_robots_and_sitemap():
    robots = client.get("/robots.txt").text
    assert "Disallow: /api/" in robots and "sitemap.xml" in robots
    sitemap = client.get("/sitemap.xml")
    assert sitemap.headers["content-type"].startswith("application/xml")
    assert sitemap.text.count("<loc>") == 4
    assert f"<loc>{i18n.BASE_URL}/zh/how-it-works</loc>" in sitemap.text


def test_articles_render_in_both_languages():
    en, zh = client.get("/how-it-works"), client.get("/zh/how-it-works")
    assert en.status_code == zh.status_code == 200
    assert '<html lang="en">' in en.text and '<html lang="zh-Hans">' in zh.text
    assert f'<link rel="canonical" href="{i18n.BASE_URL}/zh/how-it-works">' in zh.text
    assert "{{" not in en.text and "{{" not in zh.text
    for page in (en.text, zh.text):
        for src in re.findall(r'<img src="([^"]+)"', page):
            assert client.get(src).status_code == 200
    assert 'href="/how-it-works"' in client.get("/").text


def test_installable_with_a_share_target():
    for path, home in (("/manifest.webmanifest", "/"), ("/zh/manifest.webmanifest", "/zh/")):
        manifest = client.get(path)
        assert manifest.headers["content-type"].startswith("application/manifest+json")
        data = manifest.json()
        assert data["start_url"] == home and data["share_target"]["action"] == home + "share-target"
        for icon in data["icons"]:
            assert client.get(icon["src"]).status_code == 200
    assert '<link rel="manifest" href="/zh/manifest.webmanifest">' in client.get("/zh/").text
    worker = client.get("/sw.js")
    assert worker.status_code == 200 and "share-target" in worker.text
    # Before the service worker is active, a share lands on the page.
    fallback = client.post("/zh/share-target", follow_redirects=False)
    assert fallback.status_code == 303 and fallback.headers["location"] == "/zh/"


def test_solve_reports_why_it_failed():
    clash = [[0] * 9 for _ in range(9)]
    clash[0][0] = clash[0][1] = 5
    assert client.post("/api/v1/sudoku:solve", json={"grid": {"cells": clash}}).json()["status"] == "invalid"
    stuck = [[0] * 9 for _ in range(9)]
    stuck[0][:8] = range(1, 9)
    stuck[1][8] = 9  # r1c9 can only be 9
    assert client.post("/api/v1/sudoku:solve", json={"grid": {"cells": stuck}}).json()["status"] == "no_solution"


def test_pages_answer_head_requests():
    # Crawlers and uptime checks send HEAD; FastAPI's GET routes alone answer 405.
    for path in ("/", "/zh/", "/how-it-works", "/robots.txt", "/sitemap.xml"):
        assert client.head(path).status_code == 200, path
