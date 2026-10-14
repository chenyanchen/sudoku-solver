"""Main FastAPI application for Sudoku Solver."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
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


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots():
    return i18n.robots()


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap():
    return Response(i18n.sitemap(), media_type="application/xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
