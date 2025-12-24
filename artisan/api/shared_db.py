"""
Shared database for job storage across Modal instances.
Uses file-based JSON storage for persistence.
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Optional

# Use /tmp for Modal-compatible storage
DB_FILE = "/tmp/artisan_jobs.json"
_lock = threading.Lock()


def _load_db() -> dict:
    """Load jobs from file."""
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def _save_db(data: dict) -> None:
    """Save jobs to file."""
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f)
    except IOError:
        pass


def get_job(job_id: str) -> Optional[dict]:
    """Get a job by ID."""
    with _lock:
        db = _load_db()
        return db.get(job_id)


def set_job(job_id: str, job_data: dict) -> None:
    """Save a job."""
    with _lock:
        db = _load_db()
        job_data['updated_at'] = datetime.utcnow().isoformat()
        db[job_id] = job_data
        _save_db(db)


def update_job(job_id: str, updates: dict) -> Optional[dict]:
    """Update a job and return the updated data."""
    with _lock:
        db = _load_db()
        if job_id in db:
            db[job_id].update(updates)
            db[job_id]['updated_at'] = datetime.utcnow().isoformat()
            _save_db(db)
            return db[job_id]
        return None


def delete_job(job_id: str) -> bool:
    """Delete a job."""
    with _lock:
        db = _load_db()
        if job_id in db:
            del db[job_id]
            _save_db(db)
            return True
        return False


def list_jobs() -> dict:
    """List all jobs."""
    with _lock:
        return _load_db()


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


# Global instance for backwards compatibility
jobs_db = JobsDB()
