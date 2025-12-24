"""
Shared database for job storage.
Uses in-memory dict locally, designed to work with Modal's architecture.
"""

import os
from datetime import datetime
from typing import Any, Optional

# Simple in-memory storage
_jobs_store: dict = {}


def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID."""
    return _jobs_store.get(job_id)


def set_job(job_id: str, job_data: dict) -> None:
    """Save a job."""
    job_data['updated_at'] = datetime.utcnow().isoformat()
    _jobs_store[job_id] = job_data


def update_job(job_id: str, updates: dict) -> Optional[dict]:
    """Update a job and return the updated data."""
    if job_id in _jobs_store:
        _jobs_store[job_id].update(updates)
        _jobs_store[job_id]['updated_at'] = datetime.utcnow().isoformat()
        return _jobs_store[job_id]
    return None


def delete_job(job_id: str) -> bool:
    """Delete a job."""
    if job_id in _jobs_store:
        del _jobs_store[job_id]
        return True
    return False


def list_jobs() -> dict:
    """List all jobs."""
    return _jobs_store.copy()


# Legacy compatibility - expose as dict-like object
class JobsDB:
    """Dict-like interface for backwards compatibility."""

    def __getitem__(self, key: str) -> dict:
        job = get_job(key)
        if job is None:
            raise KeyError(key)
        return job

    def __setitem__(self, key: str, value: dict) -> None:
        set_job(key, value)

    def __contains__(self, key: str) -> bool:
        return get_job(key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        return get_job(key) or default

    def pop(self, key: str, default: Any = None) -> Any:
        job = get_job(key)
        if job:
            delete_job(key)
            return job
        return default


# Global instance
jobs_db = JobsDB()
