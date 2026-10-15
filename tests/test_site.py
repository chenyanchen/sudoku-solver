"""Bilingual pages and the files search engines read."""

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
    assert sitemap.text.count("<loc>") == 2


def test_solve_reports_why_it_failed():
    clash = [[0] * 9 for _ in range(9)]
    clash[0][0] = clash[0][1] = 5
    assert client.post("/api/v1/sudoku:solve", json={"grid": {"cells": clash}}).json()["status"] == "invalid"
    stuck = [[0] * 9 for _ in range(9)]
    stuck[0][:8] = range(1, 9)
    stuck[1][8] = 9  # r1c9 can only be 9
    assert client.post("/api/v1/sudoku:solve", json={"grid": {"cells": stuck}}).json()["status"] == "no_solution"
