from fastapi import APIRouter

from kombinat.api import annotations, batches, contributors, stats

router = APIRouter(prefix="/v1")
router.include_router(batches.router)
router.include_router(annotations.router)
router.include_router(stats.router)
router.include_router(contributors.router)
