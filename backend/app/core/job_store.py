# Copyright (c) 2026 Harsh Dwivedi
# Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0
# Commercial use requires written permission. See LICENSE for details.

"""
PieceWise — Abstract JobStore
Clean interface over job state storage.
Swap InMemoryJobStore for RedisJobStore with zero pipeline changes.

InMemoryJobStore  — development / single-worker deployments
RedisJobStore     — production / multi-worker / horizontally-scaled deployments
"""

from __future__ import annotations

import json
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from app.models.job import Job, JobStage, JobStatus, OutputBundle, STAGE_PROGRESS
from app.utils.logger import get_logger

log = get_logger(__name__)


# ─── Abstract Interface ──────────────────────────────────────────────────────

class JobStore(ABC):
    """
    Abstract base class for all job state backends.
    All methods are synchronous — async wrappers live in the pipeline layer.
    """

    @abstractmethod
    def create_job(self) -> Job:
        """Create a new job with PENDING status. Returns the Job."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Return Job by ID, or None if not found."""

    @abstractmethod
    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        stage: Optional[JobStage] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[OutputBundle] = None,
    ) -> None:
        """Partially update a job record. Only provided fields are changed."""

    def advance_stage(self, job_id: str, stage: JobStage) -> None:
        """
        Convenience: update stage and auto-set progress from STAGE_PROGRESS map.
        """
        self.update_job(
            job_id,
            stage=stage,
            progress=STAGE_PROGRESS.get(stage, 0),
            status=JobStatus.RUNNING if stage not in (
                JobStage.DONE, JobStage.FAILED
            ) else (
                JobStatus.DONE if stage == JobStage.DONE else JobStatus.FAILED
            ),
        )

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        self.update_job(
            job_id,
            status=JobStatus.FAILED,
            stage=JobStage.FAILED,
            error=error,
        )
        log.error("job_failed", job_id=job_id, error=error)


# ─── In-Memory Implementation ────────────────────────────────────────────────

class InMemoryJobStore(JobStore):
    """
    Thread-safe in-memory job store using a dict + RLock.
    Suitable for single-process development and testing.
    All data is lost on process restart.
    """

    def __init__(self) -> None:
        self._store: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create_job(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        with self._lock:
            self._store[job.job_id] = job
        log.info("job_created", job_id=job.job_id, backend="memory")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._store.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        stage: Optional[JobStage] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[OutputBundle] = None,
    ) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job is None:
                log.warning("update_job_not_found", job_id=job_id)
                return
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
            if result is not None:
                job.result = result
            job.updated_at = datetime.now(timezone.utc)
            self._store[job_id] = job

        log.debug(
            "job_updated",
            job_id=job_id,
            stage=stage.value if stage else None,
            progress=progress,
        )

    def count(self) -> int:
        """Return total number of jobs in store (useful for health checks)."""
        with self._lock:
            return len(self._store)


# ─── Redis Implementation ────────────────────────────────────────────────────

class RedisJobStore(JobStore):
    """
    Redis-backed job store for production multi-worker deployments.
    Jobs are JSON-serialised and stored with TTL expiry.
    Requires redis-py and a running Redis instance.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 86400) -> None:
        try:
            import redis as redis_lib
        except ImportError as e:
            raise ImportError(
                "redis package required for RedisJobStore. "
                "Install with: pip install redis"
            ) from e

        self._client = redis_lib.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = "piecewise:job:"

        # Verify connection on init
        self._client.ping()
        log.info("redis_job_store_connected", url=redis_url)

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def _serialize(self, job: Job) -> str:
        return job.model_dump_json()

    def _deserialize(self, raw: str) -> Job:
        return Job.model_validate_json(raw)

    def create_job(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))
        self._client.setex(self._key(job.job_id), self._ttl, self._serialize(job))
        log.info("job_created", job_id=job.job_id, backend="redis")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        raw = self._client.get(self._key(job_id))
        if raw is None:
            return None
        return self._deserialize(raw)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        stage: Optional[JobStage] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        result: Optional[OutputBundle] = None,
    ) -> None:
        raw = self._client.get(self._key(job_id))
        if raw is None:
            log.warning("update_job_not_found", job_id=job_id)
            return

        job = self._deserialize(raw)
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
        job.updated_at = datetime.now(timezone.utc)

        # Refresh TTL on every update
        self._client.setex(self._key(job_id), self._ttl, self._serialize(job))

        log.debug(
            "job_updated",
            job_id=job_id,
            stage=stage.value if stage else None,
            progress=progress,
        )