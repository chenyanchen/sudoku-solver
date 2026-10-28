"""Build the static copy of the site served by GitHub Pages.

This is the public site (i18n.BASE_URL): the server's own pages
(backend/i18n.py), optionally under a path prefix, with API calls sent to the
server (i18n.API_URL), whose page URLs redirect here.

    python scripts/build_static_site.py --out _site

Only the standard library is needed (the GitHub Actions workflow runs it
without installing the project).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import i18n  # noqa: E402

STATIC_FILES = ("app.js", "style.css", "favicon.svg", "apple-touch-icon.png",
                "icon-192.png", "icon-512.png", "icon-maskable-512.png")


def build(out: Path, prefix: str, api_base: str) -> None:
    if out.exists():
        shutil.rmtree(out)

    def write(path: str, text: str) -> None:
        # GitHub Pages serves /x from x.html and /x/ from x/index.html.
        target = out / (path.strip("/") or "index.html")
        if path.endswith("/") and path != "/":
            target = target / "index.html"
        elif not path.endswith("/"):
            target = target.with_name(target.name + ".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    for lang, path in i18n.PATHS.items():
        write(path, i18n.render(lang, prefix, api_base))
        manifest = out / path.strip("/") / "manifest.webmanifest"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(i18n.manifest(lang, prefix), ensure_ascii=False), encoding="utf-8")
    for slug, paths in i18n.ARTICLE_PATHS.items():
        for lang, path in paths.items():
            write(path, i18n.render_article(slug, lang, prefix))

    frontend = REPO_ROOT / "frontend"
    for name in STATIC_FILES:
        (out / "static").mkdir(parents=True, exist_ok=True)
        shutil.copy2(frontend / name, out / "static" / name)
    shutil.copytree(frontend / "articles" / "img", out / "static" / "articles" / "img")
    shutil.copy2(frontend / "sw.js", out / "sw.js")
    (out / "sitemap.xml").write_text(i18n.sitemap(), encoding="utf-8")
    (out / "robots.txt").write_text(i18n.robots(), encoding="utf-8")
    # IndexNow verifies submissions under this path with the key file here.
    (out / f"{i18n.INDEXNOW_KEY}.txt").write_text(i18n.INDEXNOW_KEY, encoding="utf-8")
    # Search Console verifies the site by this file at its root; keep it.
    for verification in frontend.glob("google*.html"):
        shutil.copy2(verification, out / verification.name)
    (out / ".nojekyll").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "_site")
    parser.add_argument("--prefix", default="", help="path the site is served under")
    parser.add_argument("--api", default=i18n.API_URL, help="origin of the API")
    args = parser.parse_args()
    build(args.out, args.prefix.rstrip("/"), args.api.rstrip("/"))
    print(f"Built {args.out}")


if __name__ == "__main__":
    main()
