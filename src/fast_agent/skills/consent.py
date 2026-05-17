"""Per-server consent gate for Skills-over-MCP catalogs.

The MCP trust act ("I'm connecting to server X") authorizes the server's
tools, but doesn't necessarily authorize merging the server's whole skill
catalog into the model's system prompt — that's a separate surface with
its own risks (instruction-shaped content the model will follow as
workflow). This store records per-server approval of a specific catalog
fingerprint, so:

- A new server's skills don't auto-merge: the user reviews and approves.
- An already-approved server adding a new skill later re-prompts: the
  fingerprint changes when the catalog changes, invalidating consent.
- Approvals persist across sessions, keyed to the fast-agent home so
  trust scope matches project scope.

The fingerprint is `sha256(server_name | sorted (name, uri) pairs)`. It
intentionally ignores skill *body* — the catalog signature is "what
skills this server advertises," not "what each skill contains." A
malicious server that swaps a body without changing names/URIs would
slip past the fingerprint, but the model still reads through `read_skill`
which surfaces server attribution; this gate is consent for *what's
listed*, not a body-integrity check.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fast_agent.skills.registry import SkillManifest

CONSENT_FILENAME = "skill_consent.json"


def compute_catalog_fingerprint(
    server_name: str, manifests: Iterable[SkillManifest]
) -> str:
    """Compute the catalog fingerprint for `server_name`.

    Deterministic over manifest ordering so two sessions discovering the
    same catalog get the same fingerprint. URI is included so a server
    swapping the URI behind a name (e.g. pointing `git-workflow` at a
    different skill body) re-triggers consent.
    """
    pairs = sorted(
        (m.name, m.uri or "") for m in manifests if m.server_name == server_name
    )
    payload = json.dumps({"server": server_name, "skills": pairs}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConsentRecord:
    fingerprint: str
    approved_at: float  # epoch seconds


class SkillConsentStore:
    """Load and persist per-server skill-catalog approvals.

    The on-disk format is a flat JSON object keyed by server name:

        {
          "github": {"fingerprint": "sha256:...", "approved_at": 1715900000.0},
          ...
        }

    Writes are atomic (write to temp, rename). Reads are best-effort —
    a corrupt or missing file behaves as "no consent for anyone," which
    is the safe default.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[str, ConsentRecord] = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def is_approved(self, server_name: str, fingerprint: str) -> bool:
        """Return True iff `server_name`'s stored fingerprint matches.

        A missing record or a fingerprint mismatch both return False —
        the only path to True is "user explicitly approved this exact
        catalog at some point."
        """
        record = self._records.get(server_name)
        return record is not None and record.fingerprint == fingerprint

    def stored_fingerprint(self, server_name: str) -> str | None:
        record = self._records.get(server_name)
        return record.fingerprint if record else None

    def approve(self, server_name: str, fingerprint: str) -> None:
        self._records[server_name] = ConsentRecord(
            fingerprint=fingerprint, approved_at=time.time()
        )
        self._save()

    def revoke(self, server_name: str) -> bool:
        """Remove approval for `server_name`. Returns True if anything changed."""
        if server_name not in self._records:
            return False
        del self._records[server_name]
        self._save()
        return True

    def all_records(self) -> dict[str, ConsentRecord]:
        return dict(self._records)

    def _load(self) -> dict[str, ConsentRecord]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt file: behave as empty. Don't overwrite — preserve evidence.
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, ConsentRecord] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            fingerprint = value.get("fingerprint")
            approved_at = value.get("approved_at")
            if not isinstance(fingerprint, str) or not isinstance(
                approved_at, (int, float)
            ):
                continue
            records[key] = ConsentRecord(
                fingerprint=fingerprint, approved_at=float(approved_at)
            )
        return records

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: {"fingerprint": rec.fingerprint, "approved_at": rec.approved_at}
            for name, rec in self._records.items()
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


def default_consent_path(home: Path | None) -> Path:
    """Resolve the consent file path for a given fast-agent home.

    Falls back to `~/.fast-agent/skill_consent.json` if no home is
    configured — keeps headless usage from silently writing into cwd.
    """
    if home is None:
        return (Path.home() / ".fast-agent" / CONSENT_FILENAME).expanduser()
    return home / CONSENT_FILENAME
