"""Main FastAPI application for Sudoku Solver."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles

from . import i18n
from .api.routes import _get_cnn_reader, router

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    """Eagerly validate CNN OCR model so misconfiguration fails at startup."""
    _get_cnn_reader()
    yield


app = FastAPI(
    title="Sudoku Solver API",
    description="API for solving Sudoku puzzles from images or JSON grids",
    version="1.0.0",
    lifespan=_app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


def page(path: str, **kwargs):
    """A site route: GET and HEAD (crawlers and uptime checks send HEAD)."""
    return app.api_route(path, methods=["GET", "HEAD"], include_in_schema=False, **kwargs)


def _moved(path: str) -> RedirectResponse | None:
    """In production (SUDOKU_REDIRECT_PAGES set) the pages live on the public
    site (GitHub Pages) and this server is only its API; locally it serves them."""
    if os.getenv("SUDOKU_REDIRECT_PAGES"):
        return RedirectResponse(i18n.BASE_URL + path, status_code=301)
    return None


@page("/", response_class=HTMLResponse)
def index_en():
    return _moved("/") or i18n.render("en")


@page("/zh/", response_class=HTMLResponse)
def index_zh():
    return _moved("/zh/") or i18n.render("zh")


@page("/zh")
def index_zh_slash():
    return RedirectResponse("/zh/", status_code=301)


@page("/how-it-works", response_class=HTMLResponse)
def how_it_works_en():
    return _moved("/how-it-works") or i18n.render_article("how-it-works", "en")


@page("/zh/how-it-works", response_class=HTMLResponse)
def how_it_works_zh():
    return _moved("/zh/how-it-works") or i18n.render_article("how-it-works", "zh")


@page("/manifest.webmanifest")
def manifest_en():
    return JSONResponse(i18n.manifest("en"), media_type="application/manifest+json")


@page("/zh/manifest.webmanifest")
def manifest_zh():
    return JSONResponse(i18n.manifest("zh"), media_type="application/manifest+json")


@page("/sw.js")
def service_worker():
    # Served from the root so its scope covers the share-target URLs.
    return FileResponse(_FRONTEND_DIR / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.post("/share-target", include_in_schema=False)
@app.post("/zh/share-target", include_in_schema=False)
def share_target_fallback(request: Request):
    """The service worker answers shares; this only runs before it is active."""
    return RedirectResponse(request.url.path.removesuffix("share-target"), status_code=303)


@page(f"/{i18n.INDEXNOW_KEY}.txt", response_class=PlainTextResponse)
def indexnow_key():
    return i18n.INDEXNOW_KEY


@page("/robots.txt", response_class=PlainTextResponse)
def robots():
    return i18n.robots()


@page("/sitemap.xml")
def sitemap():
    return _moved("/sitemap.xml") or Response(i18n.sitemap(), media_type="application/xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
