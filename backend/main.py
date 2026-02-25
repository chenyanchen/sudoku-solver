"""Main FastAPI application for Sudoku Solver."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import _get_cnn_reader, router

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    """Eagerly validate CNN OCR model so misconfiguration fails at startup."""
    _, error = _get_cnn_reader()
    if error:
        raise RuntimeError(f"Failed to load CNN model at startup: {error}")
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


@app.get("/")
async def root():
    """Root endpoint -- serve the frontend."""
    index_html = _FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "Sudoku Solver API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
