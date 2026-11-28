"""Dwell-time logic: turn tracked vehicles into parking-violation events.

A vehicle is *dwelling* when its ground point is inside a restricted zone and
it is stationary (moved less than ``still_ratio`` x its box height over the
last ``still_window_s`` seconds). Once it has dwelled for ``threshold_s`` a
single ``violation`` event is raised; when it leaves or disappears for longer
than ``grace_s`` a ``cleared`` event closes it. All timing uses video
timestamps, so results do not depend on the source frame rate.

Two measures against false / duplicate alerts:
* the dwell timer counts stationary time only, so vehicles driving through or
  creeping in traffic do not accumulate time;
* if the tracker loses a vehicle and gives it a new id (e.g. after an
  occlusion), the new track inherits the old state when their boxes overlap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detector import Detection
from .zones import Zone


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class VehicleState:
    track_id: int
    cls_name: str
    zone: str
    bbox: tuple[float, ...]
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=deque)  # (t, x, y, h)
    stop_since: float | None = None
    violation_since: float | None = None  # time the alert was raised
    event_id: int | None = None
    aliases: list[int] = field(default_factory=list)

    def dwell(self, t: float) -> float:
        return 0.0 if self.stop_since is None else t - self.stop_since

    @property
    def status(self) -> str:
        if self.violation_since is not None:
            return "violation"
        return "dwelling" if self.stop_since is not None else "moving"


@dataclass
class DwellConfig:
    threshold_s: float = 30.0
    still_window_s: float = 2.0
    still_ratio: float = 0.15
    grace_s: float = 5.0
    reassoc_iou: float = 0.5
    dup_iou: float = 0.7  # a new box overlapping a live vehicle this much is a duplicate
    dup_point_ratio: float = 0.2  # ...or whose ground point is this close (x box height)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DwellConfig":
        d = d or {}
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


class ViolationMonitor:
    def __init__(self, zones: list[Zone], cfg: DwellConfig | None = None) -> None:
        self.zones = zones
        self.cfg = cfg or DwellConfig()
        self.states: dict[int, VehicleState] = {}
        self._next_event = 1

    # ------------------------------------------------------------------ helpers
    def _zone_of(self, det: Detection) -> str | None:
        p = det.bottom_center
        for z in self.zones:
            if z.contains(p):
                return z.name
        return None

    def _motion(self, st: VehicleState) -> str:
        """'moving' if any recent ground point is far from the latest one,
        'still' if all are close and they span the whole look-back window,
        'unknown' if they are close but cover too little time to tell yet."""
        t_last, x, y, h = st.history[-1]
        limit = self.cfg.still_ratio * max(h, 1.0)
        if any(((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 > limit for _, hx, hy, _ in st.history):
            return "moving"
        if t_last - st.history[0][0] >= self.cfg.still_window_s * 0.8:
            return "still"
        return "unknown"

    def _same_vehicle(self, st: VehicleState, det: Detection) -> bool:
        """Duplicate box test: strong overlap, or (for small boxes, where a few
        pixels of offset already ruin the IoU) nearly the same ground point."""
        if iou(st.bbox, det.xyxy) >= self.cfg.dup_iou:
            return True
        sx, sy = (st.bbox[0] + st.bbox[2]) / 2.0, st.bbox[3]
        dx, dy = det.bottom_center
        h = max(min(st.bbox[3] - st.bbox[1], det.height), 1.0)
        return ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5 <= self.cfg.dup_point_ratio * h

    def _inherit(self, det: Detection, t: float, seen: set[int]) -> VehicleState | None:
        """Find a recently lost state this new track probably belongs to."""
        best, best_iou = None, self.cfg.reassoc_iou
        for st in self.states.values():
            if st.track_id in seen or t - st.last_seen > self.cfg.grace_s:
                continue
            o = iou(st.bbox, det.xyxy)
            if o >= best_iou:
                best, best_iou = st, o
        return best

    # ------------------------------------------------------------------- update
    def update(self, t: float, detections: list[Detection]) -> list[dict]:
        """Process one frame at time ``t`` (seconds). Returns new events."""
        events: list[dict] = []
        seen: set[int] = set()

        # handle established tracks first so duplicates of them are recognised below
        detections = sorted(detections, key=lambda d: d.track_id not in self.states)
        for det in detections:
            zone = self._zone_of(det)
            if zone is None:
                continue  # outside every zone: not tracked for dwell
            tid = det.track_id
            if tid not in self.states and any(self._same_vehicle(self.states[s], det) for s in seen):
                continue  # second box on a vehicle already handled this frame
            st = self.states.get(tid)
            if st is None:
                st = self._inherit(det, t, seen)
                if st is not None:  # re-key the old state under the new id
                    self.states.pop(st.track_id)
                    st.aliases.append(st.track_id)
                    st.track_id = tid
                else:
                    st = VehicleState(tid, det.cls_name, zone, det.xyxy, first_seen=t, last_seen=t)
                self.states[tid] = st
            seen.add(tid)

            st.zone, st.bbox, st.last_seen = zone, det.xyxy, t
            x, y = det.bottom_center
            st.history.append((t, x, y, det.height))
            # Keep the window, plus the newest sample from before it: on a slow,
            # bursty edge CPU frames can be >1 s apart, and the motion check must
            # still be able to compare across such gaps.
            while len(st.history) > 2 and t - st.history[1][0] >= self.cfg.still_window_s:
                st.history.popleft()

            motion = self._motion(st)
            if motion == "moving":
                if st.violation_since is None:
                    st.stop_since = None  # really moved before an alert: reset the timer
            elif motion == "still" and st.stop_since is None:
                st.stop_since = st.history[0][0]  # it has been still for the whole window
            # motion == "unknown" (too little history yet): leave the timer as it is

            if st.violation_since is None and st.dwell(t) >= self.cfg.threshold_s:
                st.violation_since = t
                st.event_id = self._next_event
                self._next_event += 1
                events.append(self._event("violation", st, t))

        # vehicles that left the zone or vanished for longer than the grace period
        for tid in list(self.states):
            st = self.states[tid]
            if tid not in seen and t - st.last_seen > self.cfg.grace_s:
                if st.violation_since is not None:
                    events.append(self._event("cleared", st, st.last_seen))
                del self.states[tid]
        return events

    def finish(self, t: float) -> list[dict]:
        """Close any open violations at the end of the stream."""
        events = [self._event("cleared", st, t) for st in self.states.values() if st.violation_since is not None]
        self.states.clear()
        return events

    def _event(self, kind: str, st: VehicleState, t: float) -> dict:
        return {
            "event": kind,
            "event_id": st.event_id,
            "track_id": st.track_id,
            "track_aliases": list(st.aliases),
            "zone": st.zone,
            "class": st.cls_name,
            "t": round(t, 2),
            "stopped_at": round(st.stop_since, 2) if st.stop_since is not None else None,
            "alert_at": round(st.violation_since, 2) if st.violation_since is not None else None,
            "dwell_s": round(t - st.stop_since, 1) if st.stop_since is not None else 0.0,
            "bbox": [int(v) for v in st.bbox],
        }
