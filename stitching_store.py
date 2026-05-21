"""File storage helpers for image-stitching sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

from sample_plane import SamplePlanePoint


@dataclass(frozen=True)
class TileRecord:
    row: int
    col: int
    x: int
    y: int
    z: int
    filename: str
    focus_score: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class StitchingSessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path | str, name: str | None = None) -> "StitchingSessionStore":
        root_path = Path(root)
        session_name = _safe_session_name(name) if name else time.strftime("stitching_%Y%m%d_%H%M%S")
        path = root_path / session_name
        suffix = 1
        while path.exists():
            path = root_path / f"{session_name}_{suffix:02d}"
            suffix += 1
        path.mkdir(parents=True, exist_ok=False)
        return cls(path)

    def save_tile(self, frame, record: TileRecord) -> TileRecord:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to save stitching tiles") from exc

        filename = record.filename or f"tile_r{record.row:03d}_c{record.col:03d}.png"
        output = self.path / filename
        ok = bool(cv2.imwrite(str(output), frame))
        if not ok:
            raise RuntimeError(f"failed to save tile image: {output}")
        return replace(record, filename=filename)

    def write_metadata(
        self,
        *,
        corners: list[SamplePlanePoint],
        tiles: list[TileRecord],
        settings: dict[str, object],
        plane: object | None = None,
    ) -> Path:
        payload: dict[str, object] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": settings,
            "corners": [corner.to_dict() for corner in corners],
            "tiles": [tile.to_dict() for tile in tiles],
        }
        if plane is not None and hasattr(plane, "to_dict"):
            payload["plane"] = plane.to_dict()
        path = self.path / "metadata.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _safe_session_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return cleaned or time.strftime("stitching_%Y%m%d_%H%M%S")
