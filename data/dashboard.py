from __future__ import annotations

import json
import sys
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

import os

# Cloud/public mode: if DATA_BASE_URL is set, dashboard loads CSVs & snapshots from that base URL (e.g. GitHub raw).
DATA_BASE_URL = os.getenv("DATA_BASE_URL", "").strip().rstrip("/")
CLOUD_MODE = bool(DATA_BASE_URL)


# Snapshot matching helper (used for multi-year history)
try:
    from app.match_utils import find_best_snapshot_match
except Exception:
    find_best_snapshot_match = None

# Optional: reuse pipeline scoring to compute SkillScore in historical snapshots
try:
    from config import HITTER_STATS, PITCHER_STATS
    from pipeline.score import add_percentiles_within_level, compute_skill_score
except Exception:
    HITTER_STATS = None
    PITCHER_STATS = None
    add_percentiles_within_level = None
    compute_skill_score = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess

from config import (
    HITTERS_OUT,
    PITCHERS_OUT,
    LEVEL_COL,
    ORG_COL,
    AGE_COL,
    FANTRAX_ROSTER_COL,
)

from pipeline.update_button import update_everything

# Global match overrides (approve-from-dashboard writes here)
try:
    from pipeline.match_overrides import append_override_row, make_name_org_key
except Exception:
    append_override_row = None  # type: ignore
    make_name_org_key = None  # type: ignore

st.set_page_config(page_title="Dynasty Prospect Dashboard", layout="wide")
st.title("Dynasty Prospect Evaluation Dashboard")
st.caption("Single skill-only score • Percentile-based within level • Age is context only")

HITTERS_PATH = Path(HITTERS_OUT)
PITCHERS_PATH = Path(PITCHERS_OUT)

# -------------------------
# Watchlist storage
# -------------------------
WATCH_DIR = ROOT / "data" / "user"
WATCH_PATH = WATCH_DIR / "watchlist.csv"


def _watch_key(player_name: str, org: str) -> str:
    return f"{str(org).strip().upper()}::{str(player_name).strip()}"


def load_watchlist() -> set[str]:
    try:
        if WATCH_PATH.exists():
            df = pd.read_csv(WATCH_PATH)
            if "watch_key" in df.columns:
                return set(df["watch_key"].dropna().astype(str).tolist())
    except Exception:
        pass
    return set()


def save_watchlist(keys: set[str]) -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"watch_key": sorted(keys)})
    df.to_csv(WATCH_PATH, index=False)


def toggle_watch(keys: set[str], key: str) -> set[str]:
    if key in keys:
        keys.remove(key)
    else:
        keys.add(key)
    save_watchlist(keys)
    return keys


# -------------------------
# Helpers
# -------------------------
def file_mtime(path: Path) -> str:
    if not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV either from local disk (default) or from DATA_BASE_URL (cloud mode)."""
    if not CLOUD_MODE:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_csv(path, engine="python")
            except Exception:
                return pd.DataFrame()

    rel = str(path).replace("\\", "/")
    url = f"{DATA_BASE_URL}/{rel}"
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()




def run_pipeline():
    cmd = [sys.executable, "-m", "pipeline.run_pipeline"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout)
    st.cache_data.clear()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _to_intish(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def _safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none"}:
        return ""
    return s.strip()


# -------------------------
# Projections color helpers (green / yellow / orange / red)
# -------------------------
def _proj_bucket_score(v):
    """
    For 0–100 score-like projections (SkillScore, RankScore, DynastyScore, etc.)
    Thresholds:
      green  >= 85
      yellow >= 70
      orange >= 55
      red    < 55
    """
    try:
        f = float(v)
    except Exception:
        return "gray"
    if pd.isna(f):
        return "gray"
    if f >= 85:
        return "green"
    if f >= 70:
        return "yellow"
    if f >= 55:
        return "orange"
    return "red"


def _proj_bucket_age_delta(v):
    """
    Age-for-level delta (lower is better):
      green  <= -1.0
      yellow <= 0.75
      orange <= 1.50
      red    >  1.50
    """
    try:
        f = float(v)
    except Exception:
        return "gray"
    if pd.isna(f):
        return "gray"
    if f <= -1.0:
        return "green"
    if f <= 0.75:
        return "yellow"
    if f <= 1.50:
        return "orange"
    return "red"


def _proj_color(bucket: str) -> str:
    return {
        "green": "background:#d6f5d6;color:#0b3d0b;",
        "yellow": "background:#fff4cc;color:#5a4300;",
        "orange": "background:#ffe0c2;color:#6b2f00;",
        "red": "background:#ffd6d6;color:#5a0000;",
        "gray": "background:#eeeeee;color:#333333;",
    }.get(bucket, "background:#eeeeee;color:#333333;")


def _proj_chip(label: str, value, bucket: str):
    v = _safe_str(value)
    if not v:
        v = "—"
    style = _proj_color(bucket)
    st.markdown(
        f'<span style="display:inline-block;padding:6px 10px;border-radius:999px;'
        f'font-size:0.9rem;margin:4px 6px 6px 0;{style}">'
        f"<b>{label}:</b> {v}</span>",
        unsafe_allow_html=True,
    )


def _link_button(label: str, url: str, *, key: str):
    """
    Prefer st.link_button when available, otherwise fallback to markdown link.
    """
    url = _safe_str(url)
    if not url:
        return
    if hasattr(st, "link_button"):
        try:
            st.link_button(label, url, use_container_width=True, key=key)
            return
        except Exception:
            pass
    st.markdown(f"- [{label}]({url})")


def _parse_grades_json(s: str) -> list[tuple[str, str]]:
    """
    Parse mlb_grades_json like {"Fastball":70,"Curveball":55,...}
    Returns sorted pairs.
    """
    s = _safe_str(s)
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                kk = _safe_str(k)
                vv = _safe_str(v)
                if kk and vv:
                    items.append((kk, vv))

            def _key(it: tuple[str, str]):
                name = it[0].lower()
                return (1, name) if name == "overall" else (0, name)

            return sorted(items, key=_key)
    except Exception:
        return []
    return []


def _trim_mlb_report_text(txt: str) -> str:
    """
    MLB prospect pages sometimes include huge extra sections after the report
    (Last Prospect / Next Prospect / Prospect Headlines / videos).
    Trim aggressively so the UI shows only the report.
    """
    txt = _safe_str(txt)
    if not txt:
        return ""

    markers = [
        "Last Prospect",
        "Next Prospect",
        "Prospect Headlines",
        "Prospects Headlines",
        "Prospect Highlights",
        "Prospects News",
        "More Prospects News",
        "Prospect Highlight",
    ]

    cut = len(txt)
    for m in markers:
        i = txt.find(m)
        if i != -1:
            cut = min(cut, i)

    trimmed = txt[:cut].strip()
    trimmed = "\n".join([line.rstrip() for line in trimmed.splitlines()]).strip()
    return trimmed


# -------------------------
# Skill-score statline helpers (STRICT allowlist)
# -------------------------
import re as _re


def _norm_colname(s: str) -> str:
    """
    Normalize col names for matching:
      - lowercase
      - collapse whitespace/underscores/dashes
      - keep %, +, / as meaningful
    """
    s = str(s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _format_stat_value(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    try:
        f = float(v)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        return f"{f:.3f}"
    except Exception:
        return _safe_str(v)


_HITTER_INPUTS = [
    "obp",
    "slg",
    "ops",
    "iso",
    "wrc+",
    "wrc",
    "woba",
    "k%",
    "bb%",
    "k-bb%",
]

_PITCHER_INPUTS = [
    "xfip",
    "siera",
    "fip",
    "k%",
    "bb%",
    "k-bb%",
    "gb%",
    "fb%",
    "ld%",
    "swstr",
    "swstr%",
    "csw",
    "csw%",
]

_VARIANTS = {
    "wrc+": ["wrc+", "wrc_plus", "wrc plus", "wrcp", "wrcp+"],
    "woba": ["woba", "wOBA".lower()],
    "k%": ["k%", "k percent", "k_pct", "k pct"],
    "bb%": ["bb%", "bb percent", "bb_pct", "bb pct"],
    "k-bb%": [
        "k-bb%",
        "k-bb percent",
        "k_bb%",
        "k-bb pct",
        "kbb%",
        "kbb pct",
        "k minus bb%",
        "k minus bb pct",
    ],
    "iso": ["iso"],
    "obp": ["obp"],
    "slg": ["slg"],
    "ops": ["ops"],
    "xfip": ["xfip", "x fip"],
    "siera": ["siera"],
    "fip": ["fip"],
    "gb%": ["gb%", "gb pct", "gb_pct", "groundball%", "groundball pct"],
    "fb%": ["fb%", "fb pct", "fb_pct", "flyball%", "flyball pct"],
    "ld%": ["ld%", "ld pct", "ld_pct", "linedrive%", "linedrive pct"],
    "swstr": ["swstr", "swstr%", "swstr pct", "swstr_pct", "swinging strike", "swinging strike%"],
    "csw": ["csw", "csw%", "csw pct", "csw_pct"],
}


def _expand_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out.append(t)
        out.extend(_VARIANTS.get(t, []))
    seen = set()
    final = []
    for x in out:
        nx = _norm_colname(x)
        if nx and nx not in seen:
            seen.add(nx)
            final.append(nx)
    return final


def _select_skill_inputs_only(df: pd.DataFrame, kind: str) -> list[str]:
    cols = list(df.columns)
    norm_map = {_norm_colname(c): c for c in cols}

    chosen: list[str] = []
    if "SkillScore" in df.columns:
        chosen.append("SkillScore")

    kind_u = str(kind).upper().strip()
    allow = _PITCHER_INPUTS if kind_u == "PITCHERS" else _HITTER_INPUTS
    allow_norm = _expand_tokens(allow)

    for tok in allow_norm:
        if tok in norm_map:
            actual = norm_map[tok]
            if actual not in chosen:
                chosen.append(actual)

    return chosen


# -------------------------
# Trend helpers (Prev snapshot + YoY snapshot)
# -------------------------
def _is_rankish(colname: str) -> bool:
    c = str(colname or "").lower()
    return "rank" in c and "coverage" not in c and "sources" not in c


def _trend_bucket_score_delta(v):
    """
    Score delta (higher is better):
      green  >= +5
      yellow >= +2
      orange >= -2
      red    <  -2
    """
    try:
        f = float(v)
    except Exception:
        return "gray"
    if pd.isna(f):
        return "gray"
    if f >= 5:
        return "green"
    if f >= 2:
        return "yellow"
    if f >= -2:
        return "orange"
    return "red"


def _trend_bucket_rank_delta(v):
    """
    Rank delta (lower is better):
      green  <= -10
      yellow <= -3
      orange <= +3
      red    >  +3
    """
    try:
        f = float(v)
    except Exception:
        return "gray"
    if pd.isna(f):
        return "gray"
    if f <= -10:
        return "green"
    if f <= -3:
        return "yellow"
    if f <= 3:
        return "orange"
    return "red"


def _trend_bucket_generic_delta(colname: str, v):
    if _is_rankish(colname):
        return _trend_bucket_rank_delta(v)
    return _trend_bucket_score_delta(v)


def _trend_chip(label: str, value, bucket: str):
    v = _safe_str(value)
    if not v:
        v = "—"
    style = _proj_color(bucket)
    st.markdown(
        f'<span style="display:inline-block;padding:6px 10px;border-radius:999px;'
        f'font-size:0.9rem;margin:4px 6px 6px 0;{style}">'
        f"<b>{label}:</b> {v}</span>",
        unsafe_allow_html=True,
    )


def _find_trend_cols(df: pd.DataFrame) -> list[str]:
    """
    Try to locate trend/delta columns without depending on exact names.
    We look for:
      - startswith trend_/delta_
      - contains delta_prev / delta_yoy / prev_delta / yoy_delta
      - contains 'prev' or 'yoy' along with 'delta' or 'change'
    """
    cols = []
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl.startswith("trend_") or cl.startswith("delta_"):
            cols.append(c)
            continue
        if "delta_prev" in cl or "delta_yoy" in cl:
            cols.append(c)
            continue
        if ("prev" in cl or "yoy" in cl) and ("delta" in cl or "change" in cl):
            cols.append(c)
            continue
    # de-dup while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# -------------------------
# Multi-year snapshot history helpers
# -------------------------
def _snapshots_dir_for_kind(kind: str) -> Path:
    k = (kind or "").strip().upper()
    sub = "hitters" if k == "HITTERS" else "pitchers"
    return Path("data") / "history" / "fangraphs_snapshots" / sub

def _available_snapshot_years(kind: str) -> list[int]:
    d = _snapshots_dir_for_kind(kind)
    if not CLOUD_MODE:
        if not d.exists():
            return []
        years = set()
        for p in d.glob("*.csv"):
            m = re.search(r"(19|20)\d{2}", p.name)
            if m:
                years.add(int(m.group(0)))
        return sorted(years)

    # Cloud mode: use _index.txt to list available snapshot CSVs.
    index_rel = str((d / "_index.txt")).replace("\\", "/")
    index_url = f"{DATA_BASE_URL}/{index_rel}"
    try:
        import requests
        r = requests.get(index_url, timeout=20)
        r.raise_for_status()
        years = set()
        for ln in r.text.splitlines():
            fn = ln.strip()
            if not fn.lower().endswith(".csv"):
                continue
            m = re.search(r"(19|20)\d{2}", fn)
            if m:
                years.add(int(m.group(0)))
        return sorted(years)
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def _load_snapshot_csv(path: Path, kind: str) -> pd.DataFrame:
    if not CLOUD_MODE:
        df = pd.read_csv(path)
        return _normalize_snapshot_df(df, kind=kind)

    rel = str(path).replace("\\", "/")
    url = f"{DATA_BASE_URL}/{rel}"
    df = pd.read_csv(url)
    return _normalize_snapshot_df(df, kind=kind)

def _normalize_snapshot_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Make historical snapshot CSVs look like our processed schema.

    Snapshot builder may copy raw FanGraphs exports which often use different column names.
    This function renames common variants and (optionally) computes SkillScore for the snapshot
    so multi-year charts can plot it.
    """
    out = df.copy()

    def _first_existing(cols):
        for c in cols:
            if c in out.columns:
                return c
        return None

    # Rename core identity fields
    name_src = _first_existing(["PlayerName", "Name", "playername", "Player", "Player Name"])
    org_src  = _first_existing(["Org", "Team", "Organization", "Org.", "MLB Team", "Club"])
    lvl_src  = _first_existing(["Level", "MiLB Level", "Lev", "Level (Reg)", "Minor Level"])
    age_src  = _first_existing(["Age", "age"])
    pid_src  = _first_existing(["playerId", "playerid", "PlayerId", "PlayerID", "ID", "player_id"])

    ren = {}
    if name_src and name_src != "PlayerName": ren[name_src] = "PlayerName"
    if org_src  and org_src  != "Org":        ren[org_src]  = "Org"
    if lvl_src  and lvl_src  != "Level":      ren[lvl_src]  = "Level"
    if age_src  and age_src  != "Age":        ren[age_src]  = "Age"
    if pid_src  and pid_src  != "playerId":   ren[pid_src]  = "playerId"
    if ren:
        out = out.rename(columns=ren)

    # Normalize Level strings if present
    if "Level" in out.columns:
        def _norm_level(v):
            if pd.isna(v): return ""
            s = str(v).strip().upper()
            s = s.replace("TRIPLE-A", "AAA").replace("DOUBLE-A", "AA")
            s = s.replace("HIGH-A", "A+").replace("LOW-A", "A")
            s = s.replace("ROOKIE", "R")
            return s
        out["Level"] = out["Level"].map(_norm_level)

    # Coerce common numeric fields
    for c in ["Age", "playerId"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Optional: compute SkillScore for that snapshot (within-level percentiles)
    # Only if we have the pipeline helpers available and Level exists.
    if add_percentiles_within_level and compute_skill_score and "Level" in out.columns:
        specs = HITTER_STATS if kind.upper().startswith("HIT") else PITCHER_STATS
        if specs:
            try:
                # Ensure required stat columns exist (create as NaN if missing)
                for stat in specs.keys():
                    if stat not in out.columns:
                        out[stat] = pd.NA
                out = add_percentiles_within_level(out, specs)
                out = compute_skill_score(out, specs, score_col="SkillScore")
            except Exception:
                # Non-fatal; we can still chart other metrics
                pass

    return out

def _pick_best_snapshot_for_year(kind: str, year: int) -> Path | None:
    d = _snapshots_dir_for_kind(kind)
    if not d.exists():
        return None
    # Prefer files whose filename contains the year, pick newest mtime
    candidates = [p for p in d.glob("*.csv") if str(year) in p.name]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]

def _match_player_row(snapshot: pd.DataFrame, player_id, player_name: str, org: str) -> pd.Series | None:
    """Return the best-matching snapshot row for the current player.

    This is intentionally **read-only** matching used for the multi-year chart:
    - Prefer IDs (playerId / MLBAM) when present
    - Fall back to normalized name (+ org when helpful)
    - Optionally use rapidfuzz if installed (via app.match_utils)

    If sidebar 'Show matching debug' is enabled, we attach:
      _match_method, _match_score, _match_debug (json-ish string)
    """
    if snapshot is None or snapshot.empty:
        return None

    # If the shared matcher exists, use it (safer + more robust than exact-only).
    if find_best_snapshot_match is not None:
        cur = {"playerId": player_id, "PlayerName": player_name, "Org": org}

        matched_row = None
        method = "no_match"
        score = None
        dbg = None

        # Support both signatures:
        #   (cur_dict, snap_df) -> (row, method, score)
        #   (cur_dict, snap_df, return_debug=True) -> (row, method, score, dbg)
        try:
            if show_match_debug:
                matched_row, method, score, dbg = find_best_snapshot_match(cur, snapshot, return_debug=True)  # type: ignore
            else:
                matched_row, method, score = find_best_snapshot_match(cur, snapshot)  # type: ignore
        except TypeError:
            # Older matcher without return_debug
            matched_row, method, score = find_best_snapshot_match(cur, snapshot)  # type: ignore
            dbg = None
        except Exception:
            matched_row = None

        if matched_row is not None:
            try:
                row = matched_row.copy()
                row["_match_method"] = method
                if score is not None:
                    row["_match_score"] = score
                if show_match_debug and dbg is not None:
                    try:
                        row["_match_debug"] = json.dumps(dbg, default=str)
                    except Exception:
                        row["_match_debug"] = str(dbg)
                return row
            except Exception:
                return matched_row

    # Fallback (no match_utils): exact playerId, then exact name+org
    sid = None
    if player_id is not None and str(player_id).strip() != "":
        try:
            sid = int(float(player_id))
        except Exception:
            sid = None

    if sid is not None and "playerId" in snapshot.columns:
        s = snapshot.copy()
        s["playerId"] = pd.to_numeric(s["playerId"], errors="coerce")
        hit = s.loc[s["playerId"] == sid]
        if not hit.empty:
            r = hit.iloc[0].copy()
            try:
                r["_match_method"] = "fg_id_fallback"
            except Exception:
                pass
            return r

    if "PlayerName" in snapshot.columns and "Org" in snapshot.columns:
        name = str(player_name or "").strip()
        o = str(org or "").strip()
        hit = snapshot.loc[
            (snapshot["PlayerName"].astype(str).str.strip() == name)
            & (snapshot["Org"].astype(str).str.strip() == o)
        ]
        if not hit.empty:
            r = hit.iloc[0].copy()
            try:
                r["_match_method"] = "name+org_fallback"
            except Exception:
                pass
            return r

    return None


def build_player_year_series(
    kind: str,
    years: list[int],
    player_id,
    player_name: str,
    org: str,
) -> pd.DataFrame:
    rows = []
    for y in years:
        p = _pick_best_snapshot_for_year(kind, y)
        if not p:
            continue
        snap = _load_snapshot_csv(p, kind)
        r0 = _match_player_row(snap, player_id, player_name, org)
        if r0 is None:
            continue
        d = r0.to_dict()
        d["Year"] = int(y)
        d["_snapshot_file"] = str(p)
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("Year")
    return out

def apply_filters(df: pd.DataFrame, key_prefix: str, watch_keys: set[str]) -> pd.DataFrame:
    cols = set(df.columns)

    level_vals = sorted(df[LEVEL_COL].dropna().unique().tolist()) if LEVEL_COL in cols else []
    org_vals = sorted(df[ORG_COL].dropna().unique().tolist()) if ORG_COL in cols else []
    roster_vals = sorted(df[FANTRAX_ROSTER_COL].dropna().unique().tolist()) if FANTRAX_ROSTER_COL in cols else []

    mlb_rank_col = _pick_col(df, ["mlb_top30_rank_mlb", "mlb_top30_rank", "mlb_rank", "mlb_top30Rank"])
    fg_orgrk_col = _pick_col(df, ["fg_org_rk", "fg_org_rank", "fg_orgRank", "fg_org_rank_num"])

    r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
    with r1c1:
        levels = st.multiselect("Level", level_vals, key=f"{key_prefix}_level")
    with r1c2:
        exclude_levels = st.multiselect("Exclude Level", level_vals, key=f"{key_prefix}_exclude_level")
    with r1c3:
        orgs = st.multiselect("Org", org_vals, key=f"{key_prefix}_org")
    with r1c4:
        exclude_orgs = st.multiselect("Exclude Org", org_vals, key=f"{key_prefix}_exclude_org")

    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns([1, 1, 1, 1, 1.2, 1])

    with r2c1:
        roster = st.multiselect("FA/Team", roster_vals, key=f"{key_prefix}_roster")
    with r2c2:
        exclude_roster = st.multiselect("Exclude FA/Team", roster_vals, key=f"{key_prefix}_exclude_roster")

    with r2c3:
        if mlb_rank_col:
            mlb_max = st.number_input(
                "MLB Top30 max rank (0=any)",
                min_value=0,
                max_value=30,
                value=0,
                key=f"{key_prefix}_mlb_rank_max",
            )
        else:
            mlb_max = 0
            st.number_input(
                "MLB Top30 max rank (0=any)",
                min_value=0,
                max_value=30,
                value=0,
                key=f"{key_prefix}_mlb_rank_max_disabled",
                disabled=True,
            )

    with r2c4:
        if fg_orgrk_col:
            fg_max = st.number_input(
                "FG Org max rank (0=any)",
                min_value=0,
                max_value=500,
                value=0,
                key=f"{key_prefix}_fg_org_rank_max",
            )
        else:
            fg_max = 0
            st.number_input(
                "FG Org max rank (0=any)",
                min_value=0,
                max_value=500,
                value=0,
                key=f"{key_prefix}_fg_org_rank_max_disabled",
                disabled=True,
            )

    with r2c5:
        name_query = st.text_input("Search name", value="", key=f"{key_prefix}_name_query").strip()
    with r2c6:
        only_watch = st.checkbox("Only Watchlist", value=False, key=f"{key_prefix}_only_watch")

    out = df

    if levels and LEVEL_COL in cols:
        out = out[out[LEVEL_COL].isin(levels)]
    if orgs and ORG_COL in cols:
        out = out[out[ORG_COL].isin(orgs)]
    if roster and FANTRAX_ROSTER_COL in cols:
        out = out[out[FANTRAX_ROSTER_COL].isin(roster)]

    if exclude_levels and LEVEL_COL in cols:
        out = out[~out[LEVEL_COL].isin(exclude_levels)]
    if exclude_orgs and ORG_COL in cols:
        out = out[~out[ORG_COL].isin(exclude_orgs)]
    if exclude_roster and FANTRAX_ROSTER_COL in cols:
        out = out[~out[FANTRAX_ROSTER_COL].isin(exclude_roster)]

    if mlb_rank_col and mlb_max and mlb_max > 0:
        rk = out[mlb_rank_col].apply(_to_intish)
        out = out[rk.notna() & (rk <= int(mlb_max))]

    if fg_orgrk_col and fg_max and fg_max > 0:
        rk = out[fg_orgrk_col].apply(_to_intish)
        out = out[rk.notna() & (rk <= int(fg_max))]

    player_col = _pick_col(out, ["PlayerName", "player_name", "Name", "player", "Player"])
    if name_query and player_col:
        nq = name_query.lower()
        out = out[out[player_col].astype(str).str.lower().str.contains(nq, na=False)]

    if only_watch and player_col and ORG_COL in out.columns:
        tmp_keys = out.apply(lambda r: _watch_key(r[player_col], r[ORG_COL]), axis=1)
        out = out[tmp_keys.isin(watch_keys)]

    if "SkillScore" in out.columns:
        out = out.sort_values("SkillScore", ascending=False)

    return out


def _build_table_view(df: pd.DataFrame, show_age: bool, watch_keys: set[str]) -> pd.DataFrame:
    player_col = _pick_col(df, ["PlayerName", "player_name", "Name", "player", "Player"])
    fg_pos_col = _pick_col(df, ["fg-pos", "fg_pos", "fg_position", "fg_primary_pos"])
    fg_orgrk_col = _pick_col(df, ["fg_org_rk", "fg_org_rank", "fg_orgRank", "fg_org_rank_num"])
    mlb_rank_col = _pick_col(df, ["mlb_top30_rank_mlb", "mlb_top30_rank", "mlb_rank", "mlb_top30Rank"])

    pitcherlist_col = _pick_col(
        df,
        ["pitcherlist_org_rank", "pl_org_rank", "pl_rank", "pitcherlist_rank", "pitcherlist_org_rk", "pl_org_rk"],
    )
    p1500_col = _pick_col(
        df,
        ["p1500_org_rank", "prospect1500_org_rank", "prospects1500_org_rank", "p1500_rank", "prospect1500_rank"],
    )

    ibw_col = _pick_col(df, ["ibw_rank"])

    out = pd.DataFrame(index=df.index)

    if player_col and ORG_COL in df.columns:
        out["⭐"] = df.apply(lambda r: "⭐" if _watch_key(r[player_col], r[ORG_COL]) in watch_keys else "", axis=1)

    desired = [
        ("Player", player_col),
        ("Org", ORG_COL if ORG_COL in df.columns else None),
        ("Level", LEVEL_COL if LEVEL_COL in df.columns else None),
        ("Age", AGE_COL if AGE_COL in df.columns else None),
        ("fg-pos", fg_pos_col),
        ("fg_org_rk", fg_orgrk_col),
        ("pitcherlist_org_rank", pitcherlist_col),
        ("p1500_org_rank", p1500_col),
        ("ibw_rank", ibw_col),
        ("FA/Team", FANTRAX_ROSTER_COL if FANTRAX_ROSTER_COL in df.columns else None),
        ("mlb_top30_rank_mlb", mlb_rank_col),
        ("Skill Score", "SkillScore" if "SkillScore" in df.columns else None),
        ("Rank Score", "RankScore" if "RankScore" in df.columns else None),
        ("Dynasty Score", "DynastyScore_65_35" if "DynastyScore_65_35" in df.columns else None),
        ("Rank Cov", "RankCoverage" if "RankCoverage" in df.columns else None),
        ("Rank Src", "RankSources" if "RankSources" in df.columns else None),
    ]

    for label, col in desired:
        if col:
            out[label] = df[col]

    if not show_age and "Age" in out.columns:
        out = out.drop(columns=["Age"], errors="ignore")

    if "mlb_top30_rank_mlb" in out.columns:
        out["mlb_top30_rank_mlb"] = out["mlb_top30_rank_mlb"].apply(_to_intish)
    if "pitcherlist_org_rank" in out.columns:
        out["pitcherlist_org_rank"] = out["pitcherlist_org_rank"].apply(_to_intish)
    if "p1500_org_rank" in out.columns:
        out["p1500_org_rank"] = out["p1500_org_rank"].apply(_to_intish)
    if "ibw_rank" in out.columns:
        out["ibw_rank"] = out["ibw_rank"].apply(_to_intish)

    return out


def _rankish_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("fg_")]
    preferred = [c for c in cols if any(k in c for k in ["rank", "top_100", "top100", "ovr", "overall", "org_rank", "org"])]
    return sorted(preferred) if preferred else []


def _scouting_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("fg_")]
    rank_cols = set(_rankish_cols(df))
    scouting = []
    for c in cols:
        if c in rank_cols:
            continue
        if any(
            k in c
            for k in [
                "fv",
                "future_value",
                "ofp",
                "risk",
                "eta",
                "report",
                "notes",
                "scout",
                "grade",
                "hit",
                "power",
                "run",
                "field",
                "arm",
                "stuff",
                "control",
                "command",
            ]
        ):
            scouting.append(c)
    return sorted(scouting)


def fallback_player_selectbox(df: pd.DataFrame, key_prefix: str) -> dict | None:
    player_col = _pick_col(df, ["PlayerName", "player_name", "Name", "player", "Player"])
    if not player_col:
        st.info("No player-name column found.")
        return None
    names = df[player_col].dropna().astype(str).tolist()
    if not names:
        st.info("No players in this view.")
        return None
    selected = st.selectbox(
        "Select a player to view details",
        options=["(none)"] + names,
        index=0,
        key=f"{key_prefix}_fallback_select",
    )
    if selected == "(none)":
        return None
    row = df[df[player_col].astype(str) == str(selected)].head(1)
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# -------------------------
# ProspectSavant display helpers
# -------------------------
def _ps_base_name(col: str) -> str:
    parts = str(col).split("_", 2)
    return parts[2] if len(parts) >= 3 else str(col)


def _ps_is_level(col: str, level_prefix: str) -> bool:
    return str(col).startswith(level_prefix)


def _ps_matches_any(base: str, wants: list[str]) -> bool:
    b = str(base).strip().lower()
    for w in wants:
        ww = str(w).strip().lower()
        if ww and ww in b:
            return True
    return False


def _ps_build_horizontal_row(r: dict, cols: list[str]) -> pd.DataFrame:
    pretty = {c: _ps_base_name(c) for c in cols}
    row = {pretty[c]: r.get(c) for c in cols}
    return pd.DataFrame([row])


# -------------------------
# Pitcher List UI helpers
# -------------------------
def _get_pitcherlist_fields(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "rank": _pick_col(df, ["pitcherlist_org_rank", "pl_org_rank", "pitcherlist_rank"]),
        "in_list": _pick_col(df, ["pitcherlist_in_list"]),
        "match_method": _pick_col(df, ["pitcherlist_match_method"]),
        "match_score": _pick_col(df, ["pitcherlist_match_score"]),
        "blurb": _pick_col(df, ["pitcherlist_blurb"]),
        "has_blurb": _pick_col(df, ["pitcherlist_has_blurb"]),
        "article_date": _pick_col(df, ["pitcherlist_blurb_article_date", "pitcherlist_article_date"]),
        "article_url": _pick_col(df, ["pitcherlist_blurb_article_url", "pitcherlist_article_url"]),
        "scraped_date": _pick_col(df, ["pitcherlist_blurb_scraped_date", "pitcherlist_scraped_date"]),
    }


def _get_mlb_fields(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "rank": _pick_col(df, ["mlb_top30_rank_mlb", "mlb_top30_rank", "mlb_rank", "mlb_top30Rank"]),
        "scraped_date": _pick_col(df, ["mlb_top30_scraped_date"]),
        "match_method": _pick_col(df, ["mlb_top30_match_method"]),
        "match_score": _pick_col(df, ["mlb_top30_match_score"]),
        "source_url": _pick_col(df, ["mlb_top30_source_url", "source_url", "mlb_source_url"]),
        "org_slug": _pick_col(df, ["mlb_top30_org_slug", "org_slug", "mlb_org_slug"]),
        "player_url": _pick_col(df, ["mlb_player_url", "mlb_url", "mlb_pipeline_url"]),
        "year": _pick_col(df, ["mlb_year"]),
        "grades_json": _pick_col(df, ["mlb_grades_json"]),
        "blurb": _pick_col(df, ["mlb_blurb"]),
    }


def _get_ibw_fields(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "rank": _pick_col(df, ["ibw_rank"]),
        "match_method": _pick_col(df, ["ibw_match_method"]),
        "match_score": _pick_col(df, ["ibw_match_score"]),
        "url": _pick_col(df, ["ibw_url"]),
        "blurb": _pick_col(df, ["ibw_blurb"]),
        "scraped_date": _pick_col(df, ["ibw_scraped_date"]),
        "year": _pick_col(df, ["ibw_year"]),
    }


def _fmt_num(x, nd=2) -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
        f = float(x)
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        return f"{f:.{nd}f}"
    except Exception:
        s = _safe_str(x)
        return s if s else "—"


def _pretty_label(col: str) -> str:
    c = str(col)
    return (
        c.replace("Proj_", "")
        .replace("proj_", "")
        .replace("projection_", "")
        .replace("projections_", "")
        .replace("__", "_")
        .replace("_", " ")
        .strip()
    )


def player_details_panel_from_row(
    df: pd.DataFrame,
    r: dict,
    show_age: bool,
    watch_keys: set[str],
    key_prefix: str,
    *,
    kind: str,
):
    player_col = _pick_col(df, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"
    fg_pos_col = _pick_col(df, ["fg-pos", "fg_pos", "fg_position", "fg_primary_pos"])
    fg_orgrk_col = _pick_col(df, ["fg_org_rk", "fg_org_rank", "fg_orgRank", "fg_org_rank_num"])

    title = _safe_str(r.get(player_col, ""))
    org_val = _safe_str(r.get(ORG_COL, ""))
    lvl_val = _safe_str(r.get(LEVEL_COL, ""))

    mlb = _get_mlb_fields(df)
    pl = _get_pitcherlist_fields(df)
    ibw = _get_ibw_fields(df)

    mlb_rank = r.get(mlb["rank"]) if mlb["rank"] else None
    mlb_year = _safe_str(r.get(mlb["year"])) if mlb["year"] else ""
    mlb_scraped = _safe_str(r.get(mlb["scraped_date"])) if mlb["scraped_date"] else ""
    mlb_org_slug = _safe_str(r.get(mlb["org_slug"])) if mlb["org_slug"] else ""

    bits = [title]
    if org_val:
        bits.append(f"({org_val})")
    if lvl_val:
        bits.append(f"— {lvl_val}")
    st.markdown(f"### {' '.join([b for b in bits if str(b).strip()]).strip()}")

    # -------------------------
    # Links
    # -------------------------
    with st.expander("Links", expanded=False):
        if title:
            q = quote_plus(f"{title} prospect")
            search_url = f"https://www.mlb.com/search?query={q}"
            _link_button("🔎 MLB Search (player)", search_url, key=f"{key_prefix}_mlb_search")

        team_url = ""
        if mlb["source_url"]:
            team_url = _safe_str(r.get(mlb["source_url"], ""))
        if not team_url:
            slug = mlb_org_slug
            if slug:
                if mlb_year:
                    team_url = f"https://www.mlb.com/milb/prospects/{mlb_year}/{slug}/"
                else:
                    team_url = f"https://www.mlb.com/milb/prospects/{slug}/"
        _link_button("🏟️ MLB Org Top 30 (team list)", team_url, key=f"{key_prefix}_mlb_team")

        deep = _safe_str(r.get(mlb["player_url"], "")) if mlb["player_url"] else ""
        if deep:
            _link_button("📌 MLB Prospect Card (direct)", deep, key=f"{key_prefix}_mlb_deep")

        # Baseball-Reference search (works for prospects)
        if title:
            br_q = quote_plus(title)
            br_url = f"https://www.baseball-reference.com/search/search.fcgi?search={br_q}"
            _link_button("📊 Baseball-Reference (search)", br_url, key=f"{key_prefix}_bbref_search")

        ibw_url = _safe_str(r.get(ibw["url"], "")) if ibw["url"] else ""
        if ibw_url:
            _link_button("🧱 IBW Prospect Ranking (source)", ibw_url, key=f"{key_prefix}_ibw_link")

        st.caption(
            "Tip: MLB direct prospect card is best for scouting grades + report. "
            "IBW link goes to the ranking post."
        )

    # -------------------------
    # Watchlist
    # -------------------------
    if org_val and title:
        wk = _watch_key(title, org_val)
        is_watched = wk in watch_keys

        cA, cB = st.columns([1, 3])
        with cA:
            btn_label = "⭐ Remove" if is_watched else "⭐ Watch"
            if st.button(btn_label, key=f"{key_prefix}_watch_toggle"):
                new_keys = toggle_watch(set(watch_keys), wk)
                st.session_state["watch_keys"] = new_keys
                st.rerun()
        with cB:
            st.caption(f"Watchlist: {'ON' if is_watched else 'OFF'}  •  Stored at: {WATCH_PATH}")

    # -------------------------
    # Badges
    # -------------------------
    badge_cols = st.columns(6)
    with badge_cols[0]:
        st.metric("Skill", f"{r.get('SkillScore', '')}")
    with badge_cols[1]:
        st.metric("Rank", f"{r.get('RankScore', '')}")
    with badge_cols[2]:
        st.metric("Dynasty", f"{r.get('DynastyScore_65_35', '')}")
    with badge_cols[3]:
        st.metric("FA/Team", f"{r.get(FANTRAX_ROSTER_COL, '')}")
    with badge_cols[4]:
        if mlb_rank is not None and str(mlb_rank).strip() not in ["", "nan", "None"]:
            st.metric("MLB Top 30", f"#{_to_intish(mlb_rank) if _to_intish(mlb_rank) is not None else mlb_rank}")
        else:
            st.metric("MLB Top 30", "—")
    with badge_cols[5]:
        st.metric("Age" if show_age else "Age", f"{r.get(AGE_COL, '')}" if show_age else "Hidden")

    # -------------------------
    # Projections (color-coded)
    # -------------------------
    with st.expander("Projections — color-coded (what the numbers mean)", expanded=False):
        st.caption(
            "Colors use shared thresholds:\n"
            "• 🟢 Green = elite\n"
            "• 🟡 Yellow = solid\n"
            "• 🟠 Orange = watchlist / mixed\n"
            "• 🔴 Red = risk / needs improvement\n"
            "For Age-for-level delta: negative is better (younger for the level)."
        )

        skill_v = r.get("SkillScore")
        rank_v = r.get("RankScore")
        dyn_v = r.get("DynastyScore_65_35")
        dyn_live_v = r.get("DynastyLive")

        _proj_chip("SkillScore", skill_v, _proj_bucket_score(skill_v))
        _proj_chip("RankScore", rank_v, _proj_bucket_score(rank_v))
        _proj_chip("DynastyScore", dyn_v, _proj_bucket_score(dyn_v))

        if dyn_live_v is not None and str(dyn_live_v).lower() not in {"", "nan", "none"}:
            _proj_chip("DynastyLive", dyn_live_v, _proj_bucket_score(dyn_live_v))

        age_delta_col = _pick_col(df, ["AgeForLevel_Delta", "age_for_level_delta", "age_delta", "age_for_level"])
        if age_delta_col:
            age_delta_v = r.get(age_delta_col)
            _proj_chip("AgeForLevel Δ", age_delta_v, _proj_bucket_age_delta(age_delta_v))
            st.caption(
                "AgeForLevel Δ interpretation:\n"
                "- Negative = younger than typical for this level (good)\n"
                "- Positive = older than typical for this level (risk)\n"
                "Thresholds: 🟢 ≤ -1.0, 🟡 ≤ 0.75, 🟠 ≤ 1.50, 🔴 > 1.50"
            )

        st.divider()
        st.markdown("**How to read these quickly**")
        st.write(
            "- **SkillScore**: what the current season performance says (relative to level).\n"
            "- **RankScore**: consensus from rank sources (weighted + coverage-adjusted).\n"
            "- **DynastyScore**: your blended “fantasy scouting” number.\n"
            "- **AgeForLevel Δ** (if present): projection helper — younger-for-level is a positive signal."
        )

    # -------------------------
    # Trends (snapshot deltas) — NEW
    # -------------------------
    with st.expander("Trends — snapshot deltas (Prev + YoY)", expanded=False):
        trend_cols = _find_trend_cols(df)

        if not trend_cols:
            st.info(
                "No trend/snapshot delta columns detected in the processed CSV for this view.\n\n"
                "If you just added snapshots, run **Refresh Data (pipeline only)** once to generate deltas."
            )
        else:
            st.caption(
                "These fields compare this player to past snapshots.\n"
                "• **Prev** = last saved snapshot\n"
                "• **YoY** = closest snapshot ~1 year ago\n\n"
                "Color logic:\n"
                "- For **scores**: 🟢 up is good, 🔴 down is bad\n"
                "- For **ranks**: 🟢 negative delta is good (rank number improved), 🔴 positive is bad"
            )
            with st.expander("Multi-year history", expanded=False):
                years = _available_snapshot_years(kind)
                if not years:
                    st.info("No multi-year snapshots found in data/history/fangraphs_snapshots for this view.")
                else:
                    # Optional: limit default range to last 4-6 years for readability
                    default_years = years[-4:] if len(years) >= 4 else years
                    pick_years = st.multiselect("Years", years, default=default_years)

                    series = build_player_year_series(
                        kind=kind,
                        years=pick_years,
                        player_id=r.get("playerId"),
                        player_name=r.get("PlayerName"),
                        org=r.get("Org"),
                    )

                    if series.empty:
                        st.info("No snapshot matches found for this player (by playerId or exact name+org).")
                    else:
                        st.caption("Chart is built from the per-year FanGraphs snapshots you generated.")
                        if kind == "HITTERS":
                            metric_choices = [c for c in ["SkillScore", "wRC+", "OBP", "SLG", "ISO", "K%", "BB%", "PA"] if c in series.columns]
                        else:
                            metric_choices = [c for c in ["SkillScore", "xFIP", "K%", "BB%", "K-BB%", "GB%", "ERA", "IP"] if c in series.columns]

                        if not metric_choices:
                            st.warning("No numeric metrics found in the snapshot rows for charting.")
                        else:
                            default_metric = "SkillScore" if "SkillScore" in metric_choices else metric_choices[0]
                            metrics = st.multiselect(
                                "Metrics to graph",
                                options=metric_choices,
                                default=[default_metric],
                                help="Pick one or more metrics. All are plotted on the same chart (no rescaling).",
                            )

                            if not metrics:
                                st.info("Select at least one metric to graph.")
                            else:
                                plot = series[["Year"] + metrics].copy()
                                for m in metrics:
                                    plot[m] = pd.to_numeric(plot[m], errors="coerce")
                                plot = plot.dropna(subset=["Year"])
                                plot = plot.sort_values("Year")
                                # Drop rows where ALL selected metrics are missing
                                plot = plot.dropna(subset=metrics, how="all")
                                if plot.empty:
                                    st.warning("Those metrics are not numeric for this player across the selected years.")
                                else:
                                    st.line_chart(plot.set_index("Year")[metrics])

                        with st.expander("Show snapshot rows used", expanded=False):
                            show_cols = [c for c in ["Year", "playerId", "PlayerName", "Org", "Level", "Age", "_snapshot_file"] + metric_choices if c in series.columns]
                            st.dataframe(series[show_cols], use_container_width=True, hide_index=True)


            # Highlight a few common deltas first (if present)
            preferred = []
            pref_keys = [
                "skill", "rankscore", "dynasty", "dynastyscore", "ageforlevel", "utility", "confidence", "risk",
                "mlb", "pitcherlist", "p1500", "ibw"
            ]
            for c in trend_cols:
                cl = str(c).lower()
                if any(k in cl for k in pref_keys):
                    preferred.append(c)

            # de-dupe, then take up to 10 “chips”
            seen = set()
            preferred2 = []
            for c in preferred:
                if c not in seen:
                    seen.add(c)
                    preferred2.append(c)
            chips = preferred2[:10] if preferred2 else trend_cols[:10]

            # chips
            for c in chips:
                v = r.get(c)
                bucket = _trend_bucket_generic_delta(c, v)
                _trend_chip(_pretty_label(c), _fmt_num(v, 2), bucket)

            st.divider()

            # Full table of trend fields
            rows = []
            for c in sorted(trend_cols, key=lambda x: str(x).lower()):
                val = r.get(c)
                rows.append((_pretty_label(c), _fmt_num(val, 2), c))
            tdf = pd.DataFrame(rows, columns=["Field", "Value", "_raw_col"])
            st.dataframe(tdf[["Field", "Value"]], width="stretch", hide_index=True)

    # -------------------------
    # Projections (raw list)
    # -------------------------
    with st.expander("Projections (dynasty utility / ETA / role / risk)", expanded=False):
        proj_cols = []
        for c in df.columns:
            cl = str(c).lower()
            if cl.startswith("proj_") or cl.startswith("projection_") or cl.startswith("projections_"):
                proj_cols.append(c)

        common_extras = [
            "AgeForLevel_Delta",
            "AgeForLevel_Score",
            "SampleReliability",
            "Proj_DynastyUtility",
            "Proj_ETA_Bucket",
            "Proj_Confidence",
            "Proj_Risk",
            "Proj_Impact",
            "Proj_Role",
            "Proj_SP_Prob",
            "Proj_HitProfile",
        ]
        for c in common_extras:
            if c in df.columns and c not in proj_cols:
                proj_cols.append(c)

        if not proj_cols:
            st.info(
                "No projection columns were found in the processed CSV. "
                "If projections exist in the pipeline, re-run the pipeline."
            )
        else:
            headline_pref = [
                "Proj_DynastyUtility",
                "Proj_ETA_Bucket",
                "Proj_Role",
                "Proj_Risk",
                "Proj_Confidence",
                "AgeForLevel_Delta",
            ]
            headline = [c for c in headline_pref if c in df.columns][:4]

            if headline:
                hc = st.columns(len(headline))
                for i, c in enumerate(headline):
                    with hc[i]:
                        if "bucket" in str(c).lower():
                            st.metric(_pretty_label(c), _safe_str(r.get(c)) or "—")
                        else:
                            st.metric(_pretty_label(c), _fmt_num(r.get(c), 2))

            rows = []
            for c in sorted(proj_cols, key=lambda x: str(x).lower()):
                val = r.get(c)
                if any(k in str(c).lower() for k in ["bucket", "role", "profile"]):
                    showv = _safe_str(val) or "—"
                else:
                    showv = _fmt_num(val, 2)
                rows.append((_pretty_label(c), showv))

            st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), width="stretch", hide_index=True)

    # -------------------------
    # At a glance
    # -------------------------
    with st.expander("At a glance", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.write("**Org**")
            st.write(r.get(ORG_COL, ""))
        with c2:
            st.write("**Level**")
            st.write(r.get(LEVEL_COL, ""))
        with c3:
            st.write("**fg-pos**")
            st.write(r.get(fg_pos_col, "") if fg_pos_col else "")
        with c4:
            st.write("**fg_org_rk**")
            st.write(r.get(fg_orgrk_col, "") if fg_orgrk_col else "")
        with c5:
            st.write("**Rank Cov**")
            st.write(r.get("RankCoverage", ""))
        with c6:
            st.write("**Rank Src**")
            st.write(r.get("RankSources", ""))

        if mlb_scraped:
            extra = f" • Year: {mlb_year}" if mlb_year else ""
            st.caption(f"MLB Top 30 scraped: {mlb_scraped}{extra}")

    # -------------------------
    # Skill Score stat line (inputs only)
    # -------------------------
    with st.expander("Skill Score stat line — inputs only", expanded=False):
        cols = _select_skill_inputs_only(df, kind=kind)
        if not cols or (cols == ["SkillScore"]):
            st.info(
                "No recognized skill-input columns were found in this dataset. "
                "If your processed CSV uses different column names, add them to the allowlist at the top of this file."
            )
        else:
            pairs = []
            for c in cols:
                if c in r:
                    pairs.append((c, _format_stat_value(r.get(c))))
            sdf = pd.DataFrame(pairs, columns=["Field", "Value"])
            st.dataframe(sdf, width="stretch", hide_index=True)
            st.caption(
                "This panel is strictly limited to the core stats that feed SkillScore. "
                "It intentionally excludes ProspectSavant/Statcast-style extras."
            )

    # -------------------------
    # MLB Pipeline dropdown
    # -------------------------
    with st.expander("MLB Pipeline Top 30 — scouting grades + report (if available)", expanded=False):
        rk_v = r.get(mlb["rank"]) if mlb["rank"] else None
        yr_v = _safe_str(r.get(mlb["year"])) if mlb["year"] else ""
        mm_v = _safe_str(r.get(mlb["match_method"])) if mlb["match_method"] else ""
        ms_v = r.get(mlb["match_score"]) if mlb["match_score"] else None
        deep_url = _safe_str(r.get(mlb["player_url"])) if mlb["player_url"] else ""
        org_slug_v = _safe_str(r.get(mlb["org_slug"])) if mlb["org_slug"] else ""

        gjson = _safe_str(r.get(mlb["grades_json"])) if mlb["grades_json"] else ""
        blurb_raw = _safe_str(r.get(mlb["blurb"])) if mlb["blurb"] else ""
        blurb = _trim_mlb_report_text(blurb_raw)

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.3, 2])
        with c1:
            st.metric("Rank", f"{_to_intish(rk_v) if _to_intish(rk_v) is not None else (_safe_str(rk_v) or '—')}")
        with c2:
            st.metric("Year", yr_v or "—")
        with c3:
            st.metric("Matched org", org_slug_v or "—")
        with c4:
            st.metric("Match", mm_v or "—")
        with c5:
            ms_int = _to_intish(ms_v)
            st.metric("Match score", f"{ms_int}" if ms_int is not None else (_safe_str(ms_v) or "—"))

        if org_val and org_slug_v:
            if org_slug_v.strip().upper() not in {org_val.strip().upper(), org_val.strip().lower()}:
                st.warning(
                    f"⚠ Org mismatch: FanGraphs Org shows **{org_val}**, but MLB Top 30 match came from **{org_slug_v}**. "
                    "This is common after trades."
                )

        if deep_url:
            _link_button("📌 Open MLB prospect card", deep_url, key=f"{key_prefix}_mlb_open_card")

        grades = _parse_grades_json(gjson)
        if grades:
            st.write("**Scouting grades**")
            gdf = pd.DataFrame(grades, columns=["Tool", "Grade"])
            st.dataframe(gdf, width="stretch", hide_index=True)
        else:
            st.info("No scouting grades found for this player in the MLB cache.")

        if blurb:
            st.write("**Scouting report**")
            st.write(blurb)
        else:
            st.info("No scouting report blurb found for this player in the MLB cache.")

        if blurb_raw and blurb_raw != blurb:
            with st.expander("Show raw MLB text (debug)", expanded=False):
                st.write(blurb_raw)

    # -------------------------
    # Pitcher List dropdown
    # -------------------------
    with st.expander("Pitcher List (Org List) — rank + blurb (if available)", expanded=False):
        rank_v = r.get(pl["rank"]) if pl["rank"] else None
        in_list_v = r.get(pl["in_list"]) if pl["in_list"] else None
        match_method_v = r.get(pl["match_method"]) if pl["match_method"] else None
        match_score_v = r.get(pl["match_score"]) if pl["match_score"] else None

        blurb_v = r.get(pl["blurb"]) if pl["blurb"] else None
        has_blurb_v = r.get(pl["has_blurb"]) if pl["has_blurb"] else None

        article_date_v = r.get(pl["article_date"]) if pl["article_date"] else None
        article_url_v = r.get(pl["article_url"]) if pl["article_url"] else None
        scraped_date_v = r.get(pl["scraped_date"]) if pl["scraped_date"] else None

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            st.metric("PL Org Rank", f"{_to_intish(rank_v) if _to_intish(rank_v) is not None else (_safe_str(rank_v) or '—')}")
        with c2:
            il = str(in_list_v).lower() if in_list_v is not None else ""
            st.metric("In list", "Yes" if il == "true" else ("No" if il == "false" else "—"))
        with c3:
            st.metric("Match", _safe_str(match_method_v) or "—")
        with c4:
            ms = _to_intish(match_score_v)
            st.metric("Match score", f"{ms}" if ms is not None else (_safe_str(match_score_v) or "—"))

        meta_bits = []
        if _safe_str(article_date_v):
            meta_bits.append(f"Article date: {_safe_str(article_date_v)}")
        if _safe_str(scraped_date_v):
            meta_bits.append(f"Scraped: {_safe_str(scraped_date_v)}")
        if meta_bits:
            st.caption(" • ".join(meta_bits))

        if _safe_str(article_url_v):
            _link_button("📄 Pitcher List article", _safe_str(article_url_v), key=f"{key_prefix}_pl_article")

        btxt = _safe_str(blurb_v)
        hb = str(has_blurb_v).lower() if has_blurb_v is not None else ""
        if btxt:
            st.write("**Blurb**")
            st.write(btxt)
        else:
            if hb == "true":
                st.info("Pitcher List blurb flag is true, but blurb text is empty.")
            else:
                st.info("No Pitcher List blurb found for this player (rank may still exist).")

    # -------------------------
    # IBW dropdown
    # -------------------------
    with st.expander("Imaginary Brick Wall (IBW) — rank + blurb (if available)", expanded=False):
        ibw_rank_v = r.get(ibw["rank"]) if ibw["rank"] else None
        ibw_mm_v = r.get(ibw["match_method"]) if ibw["match_method"] else None
        ibw_ms_v = r.get(ibw["match_score"]) if ibw["match_score"] else None
        ibw_url_v = r.get(ibw["url"]) if ibw["url"] else None
        ibw_blurb_v = r.get(ibw["blurb"]) if ibw["blurb"] else None
        ibw_scraped_v = r.get(ibw["scraped_date"]) if ibw["scraped_date"] else None
        ibw_year_v = r.get(ibw["year"]) if ibw["year"] else None

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            st.metric("IBW Rank", f"{_to_intish(ibw_rank_v) if _to_intish(ibw_rank_v) is not None else (_safe_str(ibw_rank_v) or '—')}")
        with c2:
            st.metric("Year", _safe_str(ibw_year_v) or "—")
        with c3:
            st.metric("Match", _safe_str(ibw_mm_v) or "—")
        with c4:
            ms = _to_intish(ibw_ms_v)
            st.metric("Match score", f"{ms}" if ms is not None else (_safe_str(ibw_ms_v) or "—"))

        if _safe_str(ibw_scraped_v):
            st.caption(f"Scraped: {_safe_str(ibw_scraped_v)}")

        if _safe_str(ibw_url_v):
            _link_button("🧱 Open IBW ranking post", _safe_str(ibw_url_v), key=f"{key_prefix}_ibw_article")

        btxt = _safe_str(ibw_blurb_v)
        if btxt:
            st.write("**Blurb**")
            st.write(btxt)
        else:
            st.info("No IBW blurb found for this player (rank may still exist).")

    # -------------------------
    # FanGraphs scouting/ranks
    # -------------------------
    with st.expander("FanGraphs ranks + scouting (if available)", expanded=False):
        rank_cols = _rankish_cols(df)
        scout_cols = _scouting_cols(df)

        if not rank_cols and not scout_cols:
            st.write("No FanGraphs scouting/rank columns found in this dataset yet.")
        else:
            if rank_cols:
                st.subheader("Ranks")
                rank_data = {c: r.get(c) for c in rank_cols if c in r}
                st.dataframe(pd.DataFrame([rank_data]).T.rename(columns={0: "value"}), width="stretch", hide_index=False)

            if scout_cols:
                st.subheader("Scouting")
                scout_data = {c: r.get(c) for c in scout_cols if c in r}
                st.dataframe(pd.DataFrame([scout_data]).T.rename(columns={0: "value"}), width="stretch", hide_index=False)

    # -------------------------
    # ProspectSavant
    # -------------------------
    with st.expander("ProspectSavant Leaders (A/AAA) — if available", expanded=False):
        ps_cols_all = [c for c in df.columns if str(c).startswith("ps_")]
        if not ps_cols_all:
            st.write("No ProspectSavant fields found for this player yet. (Run the ProspectSavant refresh.)")
        else:
            WANT_HITTERS = [
                "pa", "ab", "h", "2b", "3b", "hr", "rbi", "r", "bb", "so",
                "k", "k%", "bb%", "k-bb%", "csw", "swstr",
                "avg", "obp", "slg", "ops", "iso", "woba", "wrc", "wrc+",
                "sb", "cs", "ev", "max", "max ev", "hardhit", "barrel",
                "xwoba", "xslg", "babip",
            ]
            WANT_PITCHERS = [
                "ip", "era", "whip", "k/9", "bb/9", "hr/9",
                "k%", "bb%", "k-bb%", "gb%", "fb%", "ld%", "iffb",
                "fip", "xfip", "siera", "csw", "swstr",
                "stuff", "location", "pitching+", "velo", "vloc", "command",
            ]

            kind_u = str(kind).upper().strip()
            wants = WANT_PITCHERS if kind_u == "PITCHERS" else WANT_HITTERS

            ps_cols = []
            for c in ps_cols_all:
                base = _ps_base_name(c)
                if _ps_matches_any(base, wants):
                    ps_cols.append(c)

            if not ps_cols:
                st.write("ProspectSavant data found, but none matched the current allowlist for this view.")
            else:
                a_cols = [c for c in ps_cols if _ps_is_level(c, "ps_A_")]
                aaa_cols = [c for c in ps_cols if _ps_is_level(c, "ps_AAA_")]
                other_cols = [c for c in ps_cols if c not in set(a_cols + aaa_cols)]

                def _show_level(cols: list[str], title2: str):
                    if not cols:
                        return
                    cols_sorted = sorted(cols, key=lambda x: _ps_base_name(x).lower())
                    st.subheader(title2)
                    st.dataframe(_ps_build_horizontal_row(r, cols_sorted), width="stretch", hide_index=True)

                _show_level(a_cols, "Level: A")
                _show_level(aaa_cols, "Level: AAA")
                _show_level(other_cols, "Other / Unknown level")

    # -------------------------
    # Raw columns
    # -------------------------
    with st.expander("All other columns (raw)", expanded=False):
        at_a_glance = set(
            [
                player_col,
                ORG_COL if ORG_COL in df.columns else "",
                LEVEL_COL if LEVEL_COL in df.columns else "",
                AGE_COL if AGE_COL in df.columns else "",
                FANTRAX_ROSTER_COL if FANTRAX_ROSTER_COL in df.columns else "",
                "SkillScore",
                "RankScore",
                "DynastyScore_65_35",
                "RankCoverage",
                "RankSources",
            ]
        )
        if fg_pos_col:
            at_a_glance.add(fg_pos_col)
        if fg_orgrk_col:
            at_a_glance.add(fg_orgrk_col)
        if mlb["rank"]:
            at_a_glance.add(mlb["rank"])

        extra_cols = sorted([c for c in df.columns if c not in at_a_glance])
        if not extra_cols:
            st.write("No extra columns.")
        else:
            extras = {c: r.get(c) for c in extra_cols}
            st.dataframe(pd.DataFrame([extras]).T.rename(columns={0: "value"}), width="stretch")


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("Data Refresh")
    st.write(f"Hitters updated: **{file_mtime(HITTERS_PATH)}**")
    st.write(f"Pitchers updated: **{file_mtime(PITCHERS_PATH)}**")

    with st.expander("Quick actions", expanded=False):
        download_fangraphs = st.checkbox("Download FanGraphs too", value=True, key="sb_download_fangraphs")
        show_browser = st.checkbox("Show browser (debug)", value=False, key="sb_show_browser")
        manual_login = st.checkbox("Manual login (Fantrax only, if needed)", value=False, key="sb_manual_login")
        capture_exports = st.checkbox("One-time: Capture Fantrax export endpoints (manual clicks)", value=False, key="sb_capture_exports")
        capture_fangraphs = st.checkbox("One-time: Capture FanGraphs export endpoints (manual click)", value=False, key="sb_capture_fangraphs")

        refresh_prospectsavant_leaders = st.checkbox(
            "Refresh ProspectSavant Leaders (A/AAA)",
            value=False,
            key="sb_refresh_prospectsavant_leaders",
        )

    with st.expander("Rank sources (rare)", expanded=False):
        refresh_mlb_top30 = st.checkbox("Refresh MLB Pipeline Top 30", value=False, key="sb_refresh_mlb_top30")
        refresh_pitcherlist_org = st.checkbox("Refresh Pitcher List org ranks", value=False, key="sb_refresh_pitcherlist_org")
        refresh_pitcherlist_scouting = st.checkbox("Refresh Pitcher List scouting blurbs", value=False)
        show_match_debug = st.sidebar.checkbox("Show matching debug", value=False)
        refresh_prospect1500_org = st.checkbox("Refresh Prospects1500 org ranks", value=False, key="sb_refresh_prospect1500_org")
        refresh_ibw = st.checkbox("Refresh IBW Top Prospects (rare)", value=False, key="sb_refresh_ibw")

        st.caption("Tip: These are slower and only needed occasionally. Most of the time you just refresh FanGraphs + Fantrax.")

    st.divider()
    st.subheader("Display")
    show_details = st.checkbox("Show update details", value=True, key="sb_show_details")
    show_age = st.checkbox("Show Age column", value=True, key="sb_show_age")

    st.divider()
    st.subheader("Watchlist")
    st.caption(f"Stored at: {WATCH_PATH}")
    if st.button("Clear Watchlist", use_container_width=True, key="sb_clear_watch"):
        save_watchlist(set())
        st.session_state["watch_keys"] = set()
        st.success("Watchlist cleared.")
        st.rerun()

    if st.button("Update Data (download + rebuild)", use_container_width=True, type="primary", key="sb_update_data"):
        with st.spinner("Downloading latest CSVs and rebuilding..."):
            status = update_everything(
                run_pipeline_fn=run_pipeline,
                show_browser=show_browser,
                run_pipeline=True,
                manual_login=manual_login,
                capture_exports=capture_exports,
                download_fangraphs=download_fangraphs,
                capture_fangraphs=capture_fangraphs,
                refresh_mlb_top30=refresh_mlb_top30,
                refresh_pitcherlist_org=refresh_pitcherlist_org,
                refresh_pitcherlist_scouting=refresh_pitcherlist_scouting,
                refresh_prospect1500_org=refresh_prospect1500_org,
                refresh_prospectsavant_leaders=refresh_prospectsavant_leaders,
                refresh_ibw=refresh_ibw,
            )

        st.session_state["last_update_status"] = status
        st.cache_data.clear()
        st.rerun()

    if st.button("Refresh Data (pipeline only)", use_container_width=True, key="sb_refresh_pipeline"):
        with st.spinner("Running pipeline..."):
            try:
                run_pipeline()
                st.success("Refresh complete.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

# Ensure watchlist is loaded into session
if "watch_keys" not in st.session_state:
    st.session_state["watch_keys"] = load_watchlist()

watch_keys = st.session_state["watch_keys"]

# -------------------------
# Update Status
# -------------------------
status = st.session_state.get("last_update_status")
if status:
    st.subheader("Last Update Status")
    downloads = status.get("downloads", [])
    downloads_ok = status.get("downloads_ok")
    pipeline_ok = status.get("pipeline_ok")
    pipeline_error = status.get("pipeline_error")

    if downloads_ok:
        st.success("Downloads: OK")
    else:
        st.warning("Downloads: not fully successful (details below)")

    if show_details:
        for d in downloads:
            name = d.get("name", "Source")
            ok = d.get("ok", False)
            msg = d.get("message", "")
            path = d.get("path")

            if ok:
                st.write(f"✅ **{name}** — {msg}")
                if path:
                    st.caption(path)
            else:
                st.write(f"❌ **{name}** — {msg}")

    if pipeline_ok is True:
        st.success("Pipeline: OK")
    elif pipeline_ok is False:
        st.error(f"Pipeline failed: {pipeline_error}")
    else:
        st.info("Pipeline: not run")
else:
    st.info("No update run yet. Use the sidebar button to run **Update Data (download + rebuild)**.")

# -------------------------
# Main Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hitters", "Pitchers", "Targets", "Compare", "Diagnostics"])

# -------------------------
# Tab 1: Hitters
# -------------------------
with tab1:
    st.subheader("Hitters")
    if not HITTERS_PATH.exists():
        st.warning("No hitters output yet. Click Update Data.")
    else:
        df = load_csv(HITTERS_PATH)
        df2 = apply_filters(df, key_prefix="hitters", watch_keys=watch_keys)

        st.caption("Tip: Click a row to show player details below. Use ⭐ Watch in the details panel.")
        df_show = _build_table_view(df2, show_age=show_age, watch_keys=watch_keys)
        selection_obj = st.dataframe(
            df_show,
            width="stretch",
            height=650,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="hitters_table",
        )

        selected_row = None
        try:
            sel = getattr(selection_obj, "selection", None)
            if sel and getattr(sel, "rows", None):
                pos = sel.rows[0]
                original_index = df_show.index[pos]
                row = df2.loc[original_index]
                selected_row = row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict()
        except Exception:
            selected_row = None

        st.divider()
        if selected_row:
            player_details_panel_from_row(df2, selected_row, show_age=show_age, watch_keys=watch_keys, key_prefix="hitters", kind="HITTERS")
        else:
            st.info("Click a row above to view details.")
            fallback = fallback_player_selectbox(df2, key_prefix="hitters")
            if fallback:
                player_details_panel_from_row(df2, fallback, show_age=show_age, watch_keys=watch_keys, key_prefix="hitters", kind="HITTERS")

# -------------------------
# Tab 2: Pitchers
# -------------------------
with tab2:
    st.subheader("Pitchers")
    if not PITCHERS_PATH.exists():
        st.warning("No pitchers output yet. Click Update Data.")
    else:
        df = load_csv(PITCHERS_PATH)
        df2 = apply_filters(df, key_prefix="pitchers", watch_keys=watch_keys)

        st.caption("Tip: Click a row to show player details below. Use ⭐ Watch in the details panel.")
        df_show = _build_table_view(df2, show_age=show_age, watch_keys=watch_keys)
        selection_obj = st.dataframe(
            df_show,
            width="stretch",
            height=650,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="pitchers_table",
        )

        selected_row = None
        try:
            sel = getattr(selection_obj, "selection", None)
            if sel and getattr(sel, "rows", None):
                pos = sel.rows[0]
                original_index = df_show.index[pos]
                row = df2.loc[original_index]
                selected_row = row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict()
        except Exception:
            selected_row = None

        st.divider()
        if selected_row:
            player_details_panel_from_row(df2, selected_row, show_age=show_age, watch_keys=watch_keys, key_prefix="pitchers", kind="PITCHERS")
        else:
            st.info("Click a row above to view details.")
            fallback = fallback_player_selectbox(df2, key_prefix="pitchers")
            if fallback:
                player_details_panel_from_row(df2, fallback, show_age=show_age, watch_keys=watch_keys, key_prefix="pitchers", kind="PITCHERS")

# -------------------------
# Tab 3: Targets (unchanged)
# -------------------------
with tab3:
    st.subheader("Targets")
    st.caption("FA-only, non-MLB prospects. Use this to find adds and trade targets quickly.")

    hitters_df = load_csv(HITTERS_PATH) if HITTERS_PATH.exists() else pd.DataFrame()
    pitchers_df = load_csv(PITCHERS_PATH) if PITCHERS_PATH.exists() else pd.DataFrame()

    if hitters_df.empty and pitchers_df.empty:
        st.warning("No processed CSVs found yet. Run the pipeline first.")
    else:
        combined = pd.concat(
            [
                hitters_df.assign(_kind="HITTERS"),
                pitchers_df.assign(_kind="PITCHERS"),
            ],
            ignore_index=True,
        )

        def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        name_col = _pick_col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

        mlb_rank_col = _col(combined, ["mlb_top30_rank_mlb", "mlb_top30_rank", "mlb_rank"])
        pl_rank_col = _col(combined, ["pitcherlist_org_rank", "pl_org_rank", "pitcherlist_rank"])
        p1500_rank_col = _col(combined, ["p1500_org_rank", "prospect1500_org_rank", "prospects1500_org_rank"])
        ibw_rank_col = _col(combined, ["ibw_rank"])

        missing_required = []
        for req in [ORG_COL, LEVEL_COL, FANTRAX_ROSTER_COL]:
            if req not in combined.columns:
                missing_required.append(req)
        if missing_required:
            st.error(f"Missing required columns in processed CSV: {missing_required}")
        else:
            roster = combined[FANTRAX_ROSTER_COL].astype(str).str.strip().str.upper()
            lvl = combined[LEVEL_COL].astype(str).str.strip().str.upper()
            base = combined[(roster == "FA") & (lvl != "MLB")].copy()

            if base.empty:
                st.info("No rows match FA + non-MLB. (Check that FA/Team uses 'FA' and Level isn't blank.)")
            else:
                c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.2, 1.2])
                with c1:
                    kind_pick = st.selectbox("Type", ["BOTH", "HITTERS", "PITCHERS"], index=0, key="targets_kind")
                with c2:
                    min_skill = st.slider("Min SkillScore", 0, 100, 75, key="targets_min_skill")
                with c3:
                    dynasty_w = st.slider("Dynasty blend (Skill weight)", 0.0, 1.0, 0.65, 0.05, key="targets_dynasty_weight")
                with c4:
                    bucket = st.selectbox("Bucket", ["Performance edge", "Under-ranked", "Scout favorites", "Consensus"], index=0, key="targets_bucket")
                with c5:
                    max_rows = st.slider("Max rows", 25, 300, 100, key="targets_max_rows")

                levels = sorted(base[LEVEL_COL].dropna().astype(str).unique().tolist())
                levels = [x for x in levels if str(x).strip().upper() != "MLB"]
                orgs = sorted(base[ORG_COL].dropna().astype(str).unique().tolist())

                c6, c7 = st.columns([1.2, 1])
                with c6:
                    level_pick = st.multiselect("Levels", levels, default=levels, key="targets_levels")
                with c7:
                    org_pick = st.multiselect("Orgs", orgs, default=[], key="targets_orgs")

                st.markdown("#### Rank sources")
                rs1, rs2, rs3, rs4 = st.columns(4)
                with rs1:
                    use_mlb = st.checkbox("MLB", value=True, key="targets_use_mlb")
                with rs2:
                    use_pl = st.checkbox("Pitcher List", value=True, key="targets_use_pl")
                with rs3:
                    use_p1500 = st.checkbox("Prospects1500", value=True, key="targets_use_p1500")
                with rs4:
                    use_ibw = st.checkbox("IBW", value=True, key="targets_use_ibw")

                if not any([use_mlb, use_pl, use_p1500, use_ibw]):
                    st.warning("No rank sources selected. Turn at least one ON.")

                rank_mode = st.selectbox(
                    "Rank requirement",
                    [
                        "Ranked in ANY selected source",
                        "Ranked in ALL selected sources",
                        "IBW only",
                        "MLB only",
                        "Pitcher List only",
                        "Prospects1500 only",
                        "No rank requirement",
                    ],
                    index=0,
                    key="targets_rank_mode",
                )

                show_only_selected_rank_cols = st.checkbox(
                    "Show only selected rank columns",
                    value=True,
                    key="targets_show_only_selected_rank_cols",
                )

                out = base
                if kind_pick != "BOTH":
                    out = out[out["_kind"] == kind_pick]

                if level_pick:
                    out = out[out[LEVEL_COL].astype(str).isin(level_pick)]

                if org_pick:
                    out = out[out[ORG_COL].astype(str).isin(org_pick)]

                if "SkillScore" in out.columns:
                    out = out[out["SkillScore"].notna() & (out["SkillScore"] >= min_skill)]
                else:
                    st.warning("SkillScore not found — Targets view works best when SkillScore exists.")

                def _rank_series(df: pd.DataFrame, colname: str | None) -> pd.Series:
                    if not colname or colname not in df.columns:
                        return pd.Series([pd.NA] * len(df), index=df.index)
                    return df[colname].apply(_to_intish)

                r_mlb = _rank_series(out, mlb_rank_col) if use_mlb else pd.Series([pd.NA] * len(out), index=out.index)
                r_pl = _rank_series(out, pl_rank_col) if use_pl else pd.Series([pd.NA] * len(out), index=out.index)
                r_p1500 = _rank_series(out, p1500_rank_col) if use_p1500 else pd.Series([pd.NA] * len(out), index=out.index)
                r_ibw = _rank_series(out, ibw_rank_col) if use_ibw else pd.Series([pd.NA] * len(out), index=out.index)

                rank_matrix = pd.concat([r_mlb, r_pl, r_p1500, r_ibw], axis=1)
                out["_best_scout_rank"] = rank_matrix.min(axis=1, skipna=True)
                out["_has_any_scout"] = out["_best_scout_rank"].notna()

                if rank_mode == "Ranked in ANY selected source":
                    out = out[out["_has_any_scout"]].copy()
                elif rank_mode == "Ranked in ALL selected sources":
                    required = []
                    if use_mlb:
                        required.append(r_mlb.notna())
                    if use_pl:
                        required.append(r_pl.notna())
                    if use_p1500:
                        required.append(r_p1500.notna())
                    if use_ibw:
                        required.append(r_ibw.notna())
                    if required:
                        mask = required[0]
                        for m in required[1:]:
                            mask = mask & m
                        out = out[mask].copy()
                elif rank_mode == "IBW only":
                    out = out[_rank_series(out, ibw_rank_col).notna()].copy()
                elif rank_mode == "MLB only":
                    out = out[_rank_series(out, mlb_rank_col).notna()].copy()
                elif rank_mode == "Pitcher List only":
                    out = out[_rank_series(out, pl_rank_col).notna()].copy()
                elif rank_mode == "Prospects1500 only":
                    out = out[_rank_series(out, p1500_rank_col).notna()].copy()
                elif rank_mode == "No rank requirement":
                    pass

                if "SkillScore" in out.columns and "RankScore" in out.columns:
                    rs = out["RankScore"]
                    sk = out["SkillScore"]
                    out["DynastyLive"] = sk.where(rs.isna(), dynasty_w * sk + (1.0 - dynasty_w) * rs)
                elif "DynastyScore_65_35" in out.columns:
                    out["DynastyLive"] = out["DynastyScore_65_35"]
                elif "SkillScore" in out.columns:
                    out["DynastyLive"] = out["SkillScore"]

                strong_rank = st.slider("Strong scout rank threshold", 1, 200, 75, key="targets_strong_rank")
                weak_rank = st.slider("Weak scout rank threshold", 50, 500, 200, key="targets_weak_rank")

                sort_primary = (
                    "DynastyLive"
                    if "DynastyLive" in out.columns
                    else ("DynastyScore_65_35" if "DynastyScore_65_35" in out.columns else ("SkillScore" if "SkillScore" in out.columns else None))
                )

                if bucket == "Performance edge":
                    if rank_mode == "No rank requirement":
                        out = out[~out["_has_any_scout"]].copy()
                    if sort_primary:
                        out = out.sort_values([sort_primary], ascending=False)
                elif bucket == "Under-ranked":
                    out = out[out["_has_any_scout"] & (out["_best_scout_rank"] >= int(weak_rank))].copy()
                    if sort_primary:
                        out = out.sort_values([sort_primary, "_best_scout_rank"], ascending=[False, False])
                    else:
                        out = out.sort_values(["_best_scout_rank"], ascending=False)
                elif bucket == "Scout favorites":
                    out = out[out["_has_any_scout"] & (out["_best_scout_rank"] <= int(strong_rank))].copy()
                    if sort_primary:
                        out = out.sort_values(["_best_scout_rank", sort_primary], ascending=[True, False])
                    else:
                        out = out.sort_values(["_best_scout_rank"], ascending=True)
                elif bucket == "Consensus":
                    out = out[out["_has_any_scout"] & (out["_best_scout_rank"] <= int(strong_rank))].copy()
                    if sort_primary:
                        out = out.sort_values([sort_primary, "_best_scout_rank"], ascending=[False, True])
                    else:
                        out = out.sort_values(["_best_scout_rank"], ascending=True)

                out = out.head(int(max_rows))

                st.markdown("### Results")

                cols = []
                for c in [name_col, ORG_COL, LEVEL_COL, "_kind", FANTRAX_ROSTER_COL]:
                    if c in out.columns:
                        cols.append(c)

                for c in ["SkillScore", "RankScore", "DynastyScore_65_35", "DynastyLive", "RankCoverage", "RankSources"]:
                    if c in out.columns:
                        cols.append(c)

                rank_cols_to_show = []
                if use_mlb and mlb_rank_col:
                    rank_cols_to_show.append(mlb_rank_col)
                if use_pl and pl_rank_col:
                    rank_cols_to_show.append(pl_rank_col)
                if use_p1500 and p1500_rank_col:
                    rank_cols_to_show.append(p1500_rank_col)
                if use_ibw and ibw_rank_col:
                    rank_cols_to_show.append(ibw_rank_col)

                if show_only_selected_rank_cols:
                    for c in rank_cols_to_show:
                        if c in out.columns and c not in cols:
                            cols.append(c)
                else:
                    for c in [mlb_rank_col, pl_rank_col, p1500_rank_col, ibw_rank_col]:
                        if c and c in out.columns and c not in cols:
                            cols.append(c)

                cols.append("_best_scout_rank")
                show = out[cols].copy()

                for c in [mlb_rank_col, pl_rank_col, p1500_rank_col, ibw_rank_col, "_best_scout_rank", "RankSources"]:
                    if c and c in show.columns:
                        show[c] = show[c].apply(_to_intish)

                st.dataframe(show, width="stretch", hide_index=True)

                st.caption(
                    "Buckets:\n"
                    "- Performance edge = high score but missing ranks (meaningful when rank requirement allows it)\n"
                    "- Under-ranked = high score but scout ranks are weak (high number)\n"
                    "- Scout favorites = strong scout rank (low number)\n"
                    "- Consensus = high score + strong scout rank\n"
                )

# -------------------------
# Tab 4: Compare (unchanged from your version)
# -------------------------
with tab4:
    st.subheader("Compare")
    st.caption("Pick 2–4 players and view them side-by-side (scores + ranks + blurbs + stat inputs).")

    hitters_df = load_csv(HITTERS_PATH) if HITTERS_PATH.exists() else pd.DataFrame()
    pitchers_df = load_csv(PITCHERS_PATH) if PITCHERS_PATH.exists() else pd.DataFrame()

    if hitters_df.empty and pitchers_df.empty:
        st.warning("No processed CSVs found yet. Run the pipeline first.")
    else:
        combined = pd.concat(
            [
                hitters_df.assign(_kind="HITTERS"),
                pitchers_df.assign(_kind="PITCHERS"),
            ],
            ignore_index=True,
        )

        name_col = _pick_col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

        c1, c2, c3 = st.columns([1, 1.2, 2])
        with c1:
            kind_pick = st.selectbox("Pool", ["BOTH", "HITTERS", "PITCHERS"], index=0, key="cmp_kind")
        with c2:
            show_age_cmp = st.checkbox("Show age in compare", value=False, key="cmp_show_age")
        with c3:
            st.caption("Tip: Use Watchlist ⭐ as your shortlist, then compare.")

        view = combined.copy()
        if kind_pick != "BOTH":
            view = view[view["_kind"] == kind_pick]

        if ORG_COL in view.columns and name_col in view.columns:
            view["_is_watch"] = view.apply(lambda r: _watch_key(r.get(name_col, ""), r.get(ORG_COL, "")) in watch_keys, axis=1)
            view = view.sort_values(["_is_watch"], ascending=False)

        def _label(r) -> str:
            n = _safe_str(r.get(name_col, ""))
            org = _safe_str(r.get(ORG_COL, ""))
            lvl = _safe_str(r.get(LEVEL_COL, ""))
            k = _safe_str(r.get("_kind", ""))
            return f"{n} — {org} — {lvl} — {k}"

        view2 = view.head(4000).copy()
        options = view2.apply(_label, axis=1).tolist()

        pick = st.multiselect("Select 2–4 players", options=options, default=[], key="cmp_pick")

        if len(pick) < 2:
            st.info("Select at least 2 players.")
        elif len(pick) > 4:
            st.warning("Limit is 4 players. Remove a selection.")
        else:
            label_to_idx = {options[i]: view2.index[i] for i in range(len(options))}
            selected_rows = []
            for lab in pick:
                idx = label_to_idx.get(lab)
                if idx is None:
                    continue
                row = view2.loc[idx]
                selected_rows.append(row.to_dict() if isinstance(row, pd.Series) else row.iloc[0].to_dict())

            summary_cols = [
                name_col, ORG_COL, LEVEL_COL, "_kind",
                "SkillScore", "RankScore", "DynastyScore_65_35",
                "RankCoverage", "RankSources",
                "mlb_top30_rank_mlb", "pitcherlist_org_rank", "p1500_org_rank", "ibw_rank",
            ]
            if show_age_cmp and AGE_COL in view2.columns:
                summary_cols.insert(3, AGE_COL)

            summary = pd.DataFrame(selected_rows)
            summary = summary[[c for c in summary_cols if c in summary.columns]]
            st.markdown("### Summary")
            st.dataframe(summary, width="stretch", hide_index=True)

            st.divider()
            st.markdown("### Details")

            for i, r in enumerate(selected_rows):
                kind = _safe_str(r.get("_kind", "HITTERS")).upper() or "HITTERS"
                with st.expander(f"{r.get(name_col, 'Player')} ({kind})", expanded=(i == 0)):
                    player_details_panel_from_row(
                        view2,
                        r,
                        show_age=show_age_cmp,
                        watch_keys=watch_keys,
                        key_prefix=f"compare_{i}",
                        kind=kind,
                    )

# -------------------------
# Tab 5: Diagnostics (your original logic)
# -------------------------
with tab5:
    st.subheader("Diagnostics")

    hitters_df = load_csv(HITTERS_PATH) if HITTERS_PATH.exists() else pd.DataFrame()
    pitchers_df = load_csv(PITCHERS_PATH) if PITCHERS_PATH.exists() else pd.DataFrame()

    if hitters_df.empty and pitchers_df.empty:
        st.warning("No processed CSVs found yet. Run the pipeline first.")
    else:
        combined = pd.concat(
            [
                hitters_df.assign(_kind="HITTERS"),
                pitchers_df.assign(_kind="PITCHERS"),
            ],
            ignore_index=True,
        )

        def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        mlb_rank_col = _col(combined, ["mlb_top30_rank_mlb", "mlb_top30_rank", "mlb_rank"])
        pl_rank_col = _col(combined, ["pitcherlist_org_rank", "pl_org_rank", "pitcherlist_rank"])
        p1500_rank_col = _col(combined, ["p1500_org_rank", "prospect1500_org_rank", "prospects1500_org_rank"])
        ibw_rank_col = _col(combined, ["ibw_rank"])

        mlb_mm_col = _col(combined, ["mlb_top30_match_method", "mlb_match_method"])
        mlb_ms_col = _col(combined, ["mlb_top30_match_score", "mlb_match_score"])

        pl_mm_col = _col(combined, ["pitcherlist_match_method"])
        pl_ms_col = _col(combined, ["pitcherlist_match_score"])

        p1500_mm_col = _col(combined, ["p1500_match_method", "prospects1500_match_method", "prospect1500_match_method"])
        p1500_ms_col = _col(combined, ["p1500_match_score", "prospects1500_match_score", "prospect1500_match_score"])

        ibw_mm_col = _col(combined, ["ibw_match_method"])
        ibw_ms_col = _col(combined, ["ibw_match_score"])

        ps_a_cols = [c for c in combined.columns if str(c).startswith("ps_A_")]
        ps_aaa_cols = [c for c in combined.columns if str(c).startswith("ps_AAA_")]

        def _has_any(row, cols):
            for c in cols:
                v = row.get(c)
                if v is not None and str(v).lower() not in {"", "nan", "none"}:
                    return True
            return False

        combined["_has_mlb"] = combined[mlb_rank_col].notna() if mlb_rank_col else False
        combined["_has_pl"] = combined[pl_rank_col].notna() if pl_rank_col else False
        combined["_has_p1500"] = combined[p1500_rank_col].notna() if p1500_rank_col else False
        combined["_has_ibw"] = combined[ibw_rank_col].notna() if ibw_rank_col else False

        if ps_a_cols:
            combined["_has_ps_A"] = combined.apply(lambda r: _has_any(r, ps_a_cols), axis=1)
        else:
            combined["_has_ps_A"] = False

        if ps_aaa_cols:
            combined["_has_ps_AAA"] = combined.apply(lambda r: _has_any(r, ps_aaa_cols), axis=1)
        else:
            combined["_has_ps_AAA"] = False

        st.markdown("### Coverage by Org")

        if ORG_COL not in combined.columns:
            st.error(f"Missing required column: {ORG_COL}")
        else:
            name_col = _pick_col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

            cov = (
                combined.groupby(ORG_COL)
                .agg(
                    players=(name_col, "count"),
                    hitters=("_kind", lambda s: (s == "HITTERS").sum()),
                    pitchers=("_kind", lambda s: (s == "PITCHERS").sum()),
                    mlb=("_has_mlb", "sum"),
                    pl=("_has_pl", "sum"),
                    p1500=("_has_p1500", "sum"),
                    ibw=("_has_ibw", "sum"),
                    ps_A=("_has_ps_A", "sum"),
                    ps_AAA=("_has_ps_AAA", "sum"),
                )
                .reset_index()
            )

            for c in ["mlb", "pl", "p1500", "ibw", "ps_A", "ps_AAA"]:
                cov[f"{c}_pct"] = (cov[c] / cov["players"] * 100).round(1)

            st.dataframe(cov.sort_values("players", ascending=False), width="stretch", hide_index=True)

        st.markdown("### Missing source coverage (high SkillScore)")
        if "SkillScore" not in combined.columns:
            st.info("SkillScore column not found; cannot build missing-coverage suspects list.")
        else:
            min_skill = st.slider("Min SkillScore", 0, 100, 75, key="diag_min_skill")
            source_choice = st.selectbox("Source to check", ["MLB", "Pitcher List", "Prospects1500", "IBW"], index=0, key="diag_source_choice")

            source_flag = {"MLB": "_has_mlb", "Pitcher List": "_has_pl", "Prospects1500": "_has_p1500", "IBW": "_has_ibw"}[source_choice]

            name_col = _pick_col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

            suspects = combined[(combined["SkillScore"].notna()) & (combined["SkillScore"] >= min_skill) & (~combined[source_flag])].copy()
            suspects = suspects.sort_values("SkillScore", ascending=False).head(200)
            show_cols = [name_col, ORG_COL, LEVEL_COL, "_kind", "SkillScore", "RankScore", "DynastyScore_65_35"]
            st.dataframe(suspects[[c for c in show_cols if c in suspects.columns]], width="stretch", hide_index=True)

        st.markdown("### Fuzzy match review (from processed CSV)")

        min_fuzzy = st.slider("Min fuzzy score to show", 0, 100, 0, key="diag_fuzzy_min_score")
        max_fuzzy = st.slider("Max fuzzy score to show", 0, 100, 95, key="diag_fuzzy_max_score")
        st.caption("Tip: Start with Max=92–95 to focus on questionable matches.")

        name_col = _pick_col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

        def _fuzzy_table(label: str, mm_col: str | None, ms_col: str | None, rank_col: str | None):
            if not mm_col or not ms_col:
                st.info(f"{label}: no match_method/match_score columns found in processed CSV.")
                return

            tmp = combined.copy()
            mm = tmp[mm_col].astype(str).str.lower()
            is_fuzzy = mm.str.contains("fuzzy") | mm.str.contains("global_fuzzy")

            scores = tmp[ms_col].apply(_to_intish)
            tmp["_ms_int"] = scores

            tmp = tmp[is_fuzzy & tmp["_ms_int"].notna() & (tmp["_ms_int"] >= min_fuzzy) & (tmp["_ms_int"] <= max_fuzzy)].copy()
            if tmp.empty:
                st.success(f"{label}: no fuzzy matches in the selected score range.")
                return

            cols = [name_col, ORG_COL, LEVEL_COL, "_kind", "SkillScore", "RankScore", "DynastyScore_65_35", mm_col, ms_col]
            if rank_col and rank_col in tmp.columns:
                cols.append(rank_col)

            tmp = tmp.sort_values("_ms_int", ascending=True).head(300)
            st.write(f"**{label}** — showing up to 300 lowest-confidence fuzzy matches")
            st.dataframe(tmp[[c for c in cols if c in tmp.columns]], width="stretch", hide_index=True)

        _fuzzy_table("MLB Top 30", mlb_mm_col, mlb_ms_col, mlb_rank_col)
        _fuzzy_table("Pitcher List", pl_mm_col, pl_ms_col, pl_rank_col)
        _fuzzy_table("Prospects1500", p1500_mm_col, p1500_ms_col, p1500_rank_col)
        _fuzzy_table("IBW", ibw_mm_col, ibw_ms_col, ibw_rank_col)

        st.markdown("### Match audit viewer")

        audit_files = {
            "MLB": Path("data/processed/mlb_top30_match_audit.csv"),
            "Pitcher List": Path("data/processed/pitcherlist_match_audit.csv"),
            "Prospects1500": Path("data/processed/prospects1500_match_audit.csv"),
            "IBW": Path("data/processed/ibw_match_audit.csv"),
        }

        which = st.selectbox("Audit file", list(audit_files.keys()), index=0, key="diag_audit_pick")
        p = audit_files[which]

        if not p.exists():
            st.warning(f"Audit file not found: {p}")
        else:
            adf = pd.read_csv(p)
            st.caption(str(p))
            # -------------------------
            # Needs-overrides filter (helps you focus on the rows that likely need manual fixes)
            # -------------------------
            needs_only = st.checkbox(
                "Needs overrides only",
                value=True,
                key=f"diag_needs_only_{which}",
                help="Filters the audit to rows that are missing ranks or have low-confidence fuzzy matches.",
            )
            cutoff = st.slider(
                "Needs overrides score cutoff",
                0,
                100,
                90,
                key=f"diag_needs_cutoff_{which}",
                help="Rows below this score (for fuzzy matches) are treated as needing overrides.",
            )

            if needs_only:
                try:
                    if which in {"MLB", "Pitcher List"}:
                        # MLB/PL audit schema: Org, PlayerName, best_score, second_score, accepted
                        accepted = adf["accepted"] if "accepted" in adf.columns else False
                        try:
                            acc_bool = accepted.fillna(False).astype(bool)
                        except Exception:
                            acc_bool = accepted.astype(str).str.lower().isin({"true", "1", "yes", "y"})
                        best = adf["best_score"] if "best_score" in adf.columns else pd.Series([0] * len(adf))
                        best_num = pd.to_numeric(best, errors="coerce").fillna(0)
                        # Needs override if not accepted OR low best score
                        adf = adf[(~acc_bool) | (best_num < cutoff)]
                    elif which == "Prospects1500":
                        # Needs override if rank missing OR (fuzzy + low score)
                        rank_missing = adf["p1500_org_rank"].isna() if "p1500_org_rank" in adf.columns else True
                        mm = adf["p1500_match_method"].astype(str).str.lower() if "p1500_match_method" in adf.columns else pd.Series([""] * len(adf))
                        sc = pd.to_numeric(adf["p1500_match_score"], errors="coerce").fillna(0) if "p1500_match_score" in adf.columns else pd.Series([0] * len(adf))
                        is_fuzzy = mm.str.contains("fuzzy")
                        adf = adf[rank_missing | (is_fuzzy & (sc < cutoff))]
                    else:  # IBW
                        # Needs override if rank missing OR (fuzzy + low score)
                        rank_missing = adf["ibw_rank"].isna() if "ibw_rank" in adf.columns else True
                        mm = adf["match_method"].astype(str).str.lower() if "match_method" in adf.columns else pd.Series([""] * len(adf))
                        sc = pd.to_numeric(adf["match_score"], errors="coerce").fillna(0) if "match_score" in adf.columns else pd.Series([0] * len(adf))
                        is_fuzzy = mm.str.contains("fuzzy")
                        adf = adf[rank_missing | (is_fuzzy & (sc < cutoff))]
                except Exception:
                    # If the audit schema isn't what we expect, fall back to unfiltered view
                    pass


            fuzzy_only = st.checkbox("Show fuzzy only", value=False, key="diag_audit_fuzzy_only")
            if fuzzy_only and "match_method" in adf.columns:
                mm = adf["match_method"].astype(str).str.lower()
                adf = adf[mm.str.contains("fuzzy") | mm.str.contains("global_fuzzy")]

            if "match_method" in adf.columns:
                mm_vals = sorted(adf["match_method"].dropna().astype(str).unique().tolist())
                pick = st.multiselect("Match method", mm_vals, default=[], key="diag_audit_mm")
                if pick:
                    adf = adf[adf["match_method"].isin(pick)]

            if "match_score" in adf.columns:
                min_score = st.slider("Min match score", 0, 100, 0, key="diag_audit_min_score")
                adf = adf[adf["match_score"].fillna(0) >= min_score]

            st.dataframe(adf.head(700), width="stretch", hide_index=True)
            st.caption("Showing first 700 rows. Use filters to narrow down.")


            # -------------------------
            # Approve match UI (writes to global overrides CSV)
            # Supports: MLB Top 30, Pitcher List, Prospects1500, IBW
            # -------------------------
            if which in {"MLB", "Pitcher List", "Prospects1500", "IBW"}:
                st.markdown("### Approve match (write override)")

                if append_override_row is None or make_name_org_key is None:
                    st.info("Override writer not available (failed to import pipeline.match_overrides).")
                else:
                    st.caption("Appends one row to data/cache/source_match_overrides.csv. Close that CSV in Excel before approving.")
                    st.caption("Workflow: Approve → rerun `python -m pipeline.run_pipeline` → refresh/restart Streamlit.")

                    # Pick schema by audit type
                    if which in {"MLB", "Pitcher List"}:
                        # Your MLB/PL audit schema
                        base_cols = ["Org", "PlayerName", "name_norm", "best_candidate_norm", "best_score", "second_score", "accepted"]
                        show_cols = [c for c in base_cols if c in adf.columns]
                        approve_view = adf[show_cols].copy() if show_cols else adf.copy()

                        sel_obj = st.dataframe(
                            approve_view.head(700),
                            width="stretch",
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key=f"approve_table_{which}",
                        )

                        selected = None
                        try:
                            s = getattr(sel_obj, "selection", None)
                            if s and getattr(s, "rows", None):
                                pos = s.rows[0]
                                selected = approve_view.head(700).iloc[pos].to_dict()
                        except Exception:
                            selected = None

                        if not selected:
                            st.info("Select a row above to approve a match.")
                            selected = None
                        else:
                            org = _safe_str(selected.get("Org"))
                            src_name = _safe_str(selected.get("PlayerName"))
                            suggested = _safe_str(selected.get("best_candidate_norm")) or src_name
                            best_score = selected.get("best_score")
                            second_score = selected.get("second_score")

                            cA, cB, cC = st.columns([2, 1, 1])
                            with cA:
                                st.write(f"**Source player:** `{org}` — **{src_name}**")
                                if suggested and suggested != src_name:
                                    st.caption(f"Fuzzy suggestion: {suggested}")
                            with cB:
                                st.write("Best score")
                                st.code(_safe_str(best_score))
                            with cC:
                                st.write("Second score")
                                st.code(_safe_str(second_score))

                            override_source = "mlb_top30" if which == "MLB" else "pitcherlist"
                            override_org = org
                            default_search = suggested

                    elif which == "Prospects1500":
                        # Prospects1500 audit schema (post-merge audit)
                        base_cols = [
                            "Org", "PlayerName",
                            "p1500_org_rank", "p1500_match_method", "p1500_match_score", "p1500_in_list",
                            "p1500_org_slug", "p1500_source_url", "p1500_scraped_date",
                        ]
                        show_cols = [c for c in base_cols if c in adf.columns]
                        approve_view = adf[show_cols].copy() if show_cols else adf.copy()

                        sel_obj = st.dataframe(
                            approve_view.head(700),
                            width="stretch",
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key="approve_table_Prospects1500",
                        )

                        selected = None
                        try:
                            s = getattr(sel_obj, "selection", None)
                            if s and getattr(s, "rows", None):
                                pos = s.rows[0]
                                selected = approve_view.head(700).iloc[pos].to_dict()
                        except Exception:
                            selected = None

                        if not selected:
                            st.info("Select a row above to approve a match.")
                            selected = None
                        else:
                            org = _safe_str(selected.get("Org"))
                            src_name = _safe_str(selected.get("PlayerName"))
                            method = _safe_str(selected.get("p1500_match_method"))
                            score = _safe_str(selected.get("p1500_match_score"))
                            rk = _safe_str(selected.get("p1500_org_rank"))

                            st.write(f"**Prospects1500 source player:** `{org}` — **{src_name}**")
                            st.caption(f"Current method: {method or '—'} • score: {score or '—'} • rank: {rk or '—'}")

                            override_source = "prospects1500"
                            override_org = org
                            default_search = src_name

                    else:  # IBW
                        # IBW audit schema is FG-centric; allow user to set the IBW name key explicitly.
                        base_cols = ["year", "fg_name", "fg_name_norm", "match_method", "match_score", "ibw_rank", "ibw_name", "ibw_name_norm", "ibw_url"]
                        show_cols = [c for c in base_cols if c in adf.columns]
                        approve_view = adf[show_cols].copy() if show_cols else adf.copy()

                        sel_obj = st.dataframe(
                            approve_view.head(700),
                            width="stretch",
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key="approve_table_IBW",
                        )

                        selected = None
                        try:
                            s = getattr(sel_obj, "selection", None)
                            if s and getattr(s, "rows", None):
                                pos = s.rows[0]
                                selected = approve_view.head(700).iloc[pos].to_dict()
                        except Exception:
                            selected = None

                        if not selected:
                            st.info("Select a row above to approve a match.")
                            selected = None
                        else:
                            year = _safe_str(selected.get("year"))
                            fg_name = _safe_str(selected.get("fg_name"))
                            method = _safe_str(selected.get("match_method"))
                            score = _safe_str(selected.get("match_score"))
                            cur_ibw_name = _safe_str(selected.get("ibw_name"))
                            cur_ibw_rank = _safe_str(selected.get("ibw_rank"))

                            st.write(f"**IBW audit row:** year {year or '—'} • FG: **{fg_name}**")
                            st.caption(f"Current match: {cur_ibw_name or '—'} • rank: {cur_ibw_rank or '—'} • method: {method or '—'} • score: {score or '—'}")

                            # Allow user to approve a specific IBW name key (default to current ibw_name if present)
                            src_name = st.text_input(
                                "IBW player name (this is what gets keyed into overrides)",
                                value=cur_ibw_name or fg_name,
                                key="approve_ibw_source_name",
                            ).strip()

                            override_source = "ibw"
                            override_org = ""  # IBW is global (not org-scoped)
                            default_search = fg_name or src_name

                    # If we have a selected source row (or in IBW case, a filled name), show the FG picker + approve button
                    if selected is not None:
                        fg_id_col = _col(combined, ["playerId", "fg_playerId", "PlayerId", "player_id"])
                        fg_name_col = _col(combined, ["PlayerName", "player_name", "Name", "player", "Player"]) or "PlayerName"

                        if fg_id_col is None or fg_name_col not in combined.columns:
                            st.error("Could not find required columns in processed data (need playerId + name).")
                        else:
                            cand = combined.copy()
                            # For org-scoped sources, filter candidates to org first
                            if override_org and ORG_COL in cand.columns:
                                cand = cand[cand[ORG_COL].astype(str).str.upper() == override_org.upper()]

                            q = st.text_input(
                                "Search processed players (filters to org when possible)",
                                value=default_search,
                                key=f"approve_search_{which}",
                            ).strip()
                            if q:
                                cand = cand[cand[fg_name_col].astype(str).str.contains(q, case=False, na=False)]

                            keep = [fg_name_col, fg_id_col]
                            for extra in [ORG_COL, LEVEL_COL, "_kind", "SkillScore", "RankScore", "DynastyScore_65_35"]:
                                if extra in cand.columns and extra not in keep:
                                    keep.append(extra)

                            cand_show = cand[keep].copy()
                            if "SkillScore" in cand_show.columns:
                                cand_show = cand_show.sort_values("SkillScore", ascending=False)
                            cand_show = cand_show.head(50)

                            if cand_show.empty:
                                st.warning("No candidates found. Try clearing the search box or selecting a different row.")
                            else:
                                records = cand_show.to_dict(orient="records")

                                def _label(r: dict) -> str:
                                    parts = [str(r.get(fg_name_col, ""))]
                                    if ORG_COL in r and r.get(ORG_COL) is not None and _safe_str(r.get(ORG_COL)):
                                        parts.append(str(r.get(ORG_COL)))
                                    if LEVEL_COL in r and r.get(LEVEL_COL) is not None and _safe_str(r.get(LEVEL_COL)):
                                        parts.append(str(r.get(LEVEL_COL)))
                                    if "_kind" in r and r.get("_kind") is not None and _safe_str(r.get("_kind")):
                                        parts.append(str(r.get("_kind")))
                                    if "SkillScore" in r and _safe_str(r.get("SkillScore")):
                                        parts.append(f"Skill {r.get('SkillScore')}")
                                    parts.append(f"FGID {r.get(fg_id_col)}")
                                    return " • ".join([p for p in parts if p and p.lower() not in {"nan", "none"}])

                                labels = [_label(r) for r in records]
                                pick = st.selectbox(
                                    "Select the correct processed player (FanGraphs playerId)",
                                    labels,
                                    index=0,
                                    key=f"approve_pick_{which}",
                                )
                                picked = records[labels.index(pick)]
                                fg_playerId = picked.get(fg_id_col)

                                note = st.text_input(
                                    "Note (optional)",
                                    value="approved via dashboard",
                                    key=f"approve_note_{which}",
                                )

                                confirm = st.checkbox(
                                    "Confirm: write override row (manual CSV append) and rerun pipeline later",
                                    value=False,
                                    key=f"approve_confirm_{which}",
                                )

                                if st.button(
                                    "✅ Approve match (write override)",
                                    key=f"approve_button_{which}",
                                    disabled=not confirm,
                                ):
                                    try:
                                        source_id = make_name_org_key(src_name, override_org)
                                        if CLOUD_MODE:
                                            st.info("Cloud mode: copy this override row and paste it into data/cache/source_match_overrides.csv on your PC, then rerun the pipeline.")
                                            now_str = datetime.now().isoformat(timespec="seconds")
                                            st.text_area(
                                                "Override row (copy/paste)",
                                                value=f"{source},org_name_norm,{source_id},{org},{int(float(fg_playerId))},{note},{now_str}",
                                                height=80,
                                            )
                                        else:
                                            append_override_row(
                                                source=source,
                                                source_id_type="org_name_norm",
                                                source_id=source_id,
                                                org=org,
                                                fg_playerId=int(float(fg_playerId)),
                                                note=note,
                                            )
                                            st.success("Override saved.")
                                        st.code(
                                            f"{override_source},org_name_norm,{source_id},{override_org},{int(float(fg_playerId))},{note}",
                                            language="text",
                                        )
                                        st.info("Next: rerun `python -m pipeline.run_pipeline` then refresh/restart the dashboard.")
                                        try:
                                            st.cache_data.clear()
                                        except Exception:
                                            pass
                                    except PermissionError as e:
                                        st.error(f"Could not write overrides CSV (is it open in Excel?): {e}")
                                    except Exception as e:
                                        st.error(f"Failed to write override: {e}")
