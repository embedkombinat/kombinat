import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kombinat.api.router import router as v1_router
from kombinat.config import get_settings
from kombinat.database import close_pool, create_pool, ping
from kombinat.expiry import expire_batches

logger = logging.getLogger(__name__)


async def _expiry_loop(app: FastAPI) -> None:
    """Background loop that expires stale batches every hour."""
    while True:
        await asyncio.sleep(3600)
        try:
            count = await expire_batches(app.state.db)
            if count:
                logger.info("Expired %d batches", count)
        except Exception:
            logger.exception("Error in batch expiry loop")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.db = await create_pool(settings.database_url)
    task = asyncio.create_task(_expiry_loop(app))
    yield
    task.cancel()
    # return_exceptions=True absorbs the task's own CancelledError without a
    # suppress() block that would also swallow a cancellation delivered to
    # this lifespan coroutine during a forced shutdown.
    await asyncio.gather(task, return_exceptions=True)
    await close_pool(app.state.db)


app = FastAPI(title="kombinat", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://embedkombinat.github.io", "http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(v1_router)


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    """Health check endpoint."""
    pool = app.state.db
    db_ok = await ping(pool)
    if db_ok:
        return JSONResponse(
            content={"status": "ok", "db": "connected"},
            status_code=200,
        )
    return JSONResponse(
        content={"status": "error", "db": "disconnected"},
        status_code=503,
    )
