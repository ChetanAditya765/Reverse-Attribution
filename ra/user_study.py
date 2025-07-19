"""
Light-weight user-study framework for Reverse-Attribution.

Supports two experiments:
1.  **Trust Calibration** – how RA explanations change user-reported trust.  
2.  **Debugging Time** – how quickly users locate failures with/without RA.

The module is UI-agnostic.  Supply a UI front-end (e.g. Streamlit) that calls
`UserStudySession` methods.  All results are stored as JSON/CSV for later
analysis with `UserStudyAnalyzer`.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json, csv, time
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Core dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class InteractionRecord:
    participant_id: str
    sample_id: str
    condition: str            # "ra" or "baseline"
    task: str                 # "trust" or "debug"
    response: float | int
    response_time: float      # sec
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StudyConfig:
    study_name: str
    out_dir: str = "user_study_data"
    trust_scale: int = 5


# --------------------------------------------------------------------------- #
# Session manager
# --------------------------------------------------------------------------- #
class UserStudySession:
    """Collects responses from a single participant."""

    def __init__(self, participant_id: str, cfg: StudyConfig):
        self.pid = participant_id
        self.cfg = cfg
        Path(cfg.out_dir).mkdir(exist_ok=True)
        self.records: List[InteractionRecord] = []

    # -------- Trust task --------------------------------------------------- #
    def record_trust(self, sample_id: str, condition: str,
                     before: float, after: float, meta: Dict[str, Any]) -> None:
        t_now = time.time()
        self.records.append(InteractionRecord(
            self.pid, sample_id, condition, "trust_before",
            before, t_now, meta))
        self.records.append(InteractionRecord(
            self.pid, sample_id, condition, "trust_after",
            after, time.time() - t_now, meta))

    # -------- Debug-time task --------------------------------------------- #
    def record_debug_time(self, sample_id: str, condition: str,
                          seconds: float, success: bool, meta: Dict[str, Any]) -> None:
        self.records.append(InteractionRecord(
            self.pid, sample_id, condition, "debug_time",
            seconds, seconds, {**meta, "success": success}))

    # -------- Persist ------------------------------------------------------ #
    def save(self) -> Path:
        """Append records to CSV (one file per study)."""
        fpath = Path(self.cfg.out_dir) / f"{self.cfg.study_name}.csv"
        is_new = not fpath.exists()
        with fpath.open("a", newline="") as fh:
            writer = csv.DictWriter(fh,
                                    fieldnames=InteractionRecord.__annotations__.keys())
            if is_new:
                writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_dict())
        self.records.clear()
        return fpath


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class UserStudyAnalyzer:
    """Aggregate and analyse study CSV."""

    def __init__(self, csv_path: str | Path, trust_scale: int = 5):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.trust_scale = trust_scale

    # ---------------- Trust metrics --------------------------------------- #
    def trust_change(self) -> Dict[str, float]:
        """Return mean Δtrust for RA vs baseline."""
        before = self.df[self.df.task == "trust_before"]
        after  = self.df[self.df.task == "trust_after"]

        merged = before.merge(after,
                              on=["participant_id", "sample_id", "condition"],
                              suffixes=("_b", "_a"))
        merged["delta"] = merged["response_a"] - merged["response_b"]

        return (merged.groupby("condition")["delta"]
                      .agg(["mean", "std", "count"])
                      .rename(columns={"mean": "avg_delta"})).to_dict("index")

    # ---------------- Debugging metrics ----------------------------------- #
    def debugging_stats(self) -> Dict[str, Any]:
        sub = self.df[self.df.task == "debug_time"]
        g   = sub.groupby("condition")

        out = {}
        out["mean_time"]   = g["response"].mean().to_dict()
        out["median_time"] = g["response"].median().to_dict()
        out["success_rate"] = (g.apply(lambda x: (x["meta"].str.contains("'success': True")
                                                  .mean()))
                               .to_dict())
        return out

    # ---------------- Export helper --------------------------------------- #
    def to_markdown(self) -> str:
        trust_md = pd.DataFrame(self.trust_change()).T.to_markdown(floatfmt=".3f")
        debug_md = pd.DataFrame(self.debugging_stats()).T.to_markdown(floatfmt=".2f")
        return ("### Trust-change results\n" + trust_md +
                "\n\n### Debugging-time results\n" + debug_md)


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def new_session(participant_id: str,
                study_name: str = "default_study",
                out_dir: str = "user_study_data") -> UserStudySession:
    return UserStudySession(participant_id, StudyConfig(study_name, out_dir))


if __name__ == "__main__":
    # Quick self-test
    sess = new_session("tester")
    sess.record_trust("s1", "ra", 2, 4, {"model": "BERT"})
    sess.record_debug_time("s2", "baseline", 65.3, True, {})
    csv_file = sess.save()
    print(f"Saved to {csv_file}")

    ana = UserStudyAnalyzer(csv_file)
    print(ana.to_markdown())
