"""Main FastAPI application for Sudoku Solver."""

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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index_en():
    return i18n.render("en")


@app.get("/zh/", response_class=HTMLResponse, include_in_schema=False)
def index_zh():
    return i18n.render("zh")


@app.get("/zh", include_in_schema=False)
def index_zh_slash():
    return RedirectResponse("/zh/", status_code=301)


@app.get("/how-it-works", response_class=HTMLResponse, include_in_schema=False)
def how_it_works_en():
    return i18n.render_article("how-it-works", "en")


@app.get("/zh/how-it-works", response_class=HTMLResponse, include_in_schema=False)
def how_it_works_zh():
    return i18n.render_article("how-it-works", "zh")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest_en():
    return JSONResponse(i18n.manifest("en"), media_type="application/manifest+json")


@app.get("/zh/manifest.webmanifest", include_in_schema=False)
def manifest_zh():
    return JSONResponse(i18n.manifest("zh"), media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    # Served from the root so its scope covers the share-target URLs.
    return FileResponse(_FRONTEND_DIR / "sw.js", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})


@app.post("/share-target", include_in_schema=False)
@app.post("/zh/share-target", include_in_schema=False)
def share_target_fallback(request: Request):
    """The service worker answers shares; this only runs before it is active."""
    return RedirectResponse(request.url.path.removesuffix("share-target"), status_code=303)


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots():
    return i18n.robots()


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap():
    return Response(i18n.sitemap(), media_type="application/xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
