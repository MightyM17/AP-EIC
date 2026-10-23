# unified_journal_app.py
# ================================================================
# Unified EIC-AIssist + AI Scientist journal simulation
# ================================================================
# This app:
# 1) uses the original EIC-AIssist v12 workflow dataset as the
#    empirical template for realistic journal timing/outcome behavior;
# 2) uses the 103 AI Scientist papers and persona reviews as the
#    manuscript/review corpus;
# 3) builds one coherent simulated journal;
# 4) exposes Paper, Reviewer, AE and EIC POVs;
# 5) runs the existing rule-based + Random Forest recommendation idea;
# 6) creates mock reviewer communications in multiple tones;
# 7) optionally loads the AI Scientist vs Stanford source-comparison outputs.
#
# Run:
#   pip install streamlit pandas numpy plotly openpyxl scikit-learn
#   streamlit run unified_journal_app.py
# ================================================================

from __future__ import annotations

import ast
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ================================================================
# CONFIG
# ================================================================

RANDOM_SEED = 42
BUILD_VERSION = "2.0"
SIMULATION_REFERENCE_DATE = pd.Timestamp(os.environ.get("SIMULATION_REFERENCE_DATE", "2026-10-15"))
APP_DIR = Path(__file__).resolve().parent
AI_SCIENTIST_DIR = Path(
    os.environ.get(
        "AI_SCIENTIST_DIR",
        "/Users/mithilbavishi/Desktop/Mithil/AI_Scientist",
    )
)

UNIFIED_DIR = APP_DIR / "Unified_Journal"

WORKFLOW_XLSX_CANDIDATES = [
    APP_DIR / "EIC-AIssist_peer_review_synth_v12.xlsx",
    APP_DIR / "EIC-AIssist_peer_review_synth_v11.xlsx",
]
WORKFLOW_PAPER_CSV_CANDIDATES = [
    APP_DIR / "EIC-AIssist_PaperHeader_v12.csv",
    APP_DIR / "EIC-AIssist_PaperHeader_v11.csv",
]
WORKFLOW_REVIEWER_CSV_CANDIDATES = [
    APP_DIR / "EIC-AIssist_ReviewerRows_v12.csv",
    APP_DIR / "EIC-AIssist_ReviewerRows_v11.csv",
]
PAPER_REVIEWS_CANDIDATES = [
    APP_DIR / "PaperReviews.csv",
    AI_SCIENTIST_DIR / "PaperReviews.csv",
]
REVIEWERS_CANDIDATES = [
    APP_DIR / "Reviewers.csv",
    AI_SCIENTIST_DIR / "Reviewers.csv",
]
SOURCE_COMPARISON_DIRS = [
    APP_DIR,
    AI_SCIENTIST_DIR / "source_comparison_experiment" / "outputs",
]

PAPERS_FILE = UNIFIED_DIR / "Papers.csv"
EDITORS_FILE = UNIFIED_DIR / "Editors.csv"
REVIEWERS_FILE = UNIFIED_DIR / "Reviewers.csv"
ASSIGNMENTS_FILE = UNIFIED_DIR / "ReviewerAssignments.csv"
REVIEWS_FILE = UNIFIED_DIR / "Reviews.csv"
DECISIONS_FILE = UNIFIED_DIR / "PaperDecisions.csv"
RECOMMENDATIONS_FILE = UNIFIED_DIR / "Recommendations.csv"
MESSAGES_FILE = UNIFIED_DIR / "Messages.csv"
ACTION_LOG_FILE = UNIFIED_DIR / "ActionLog.csv"
SCENARIO_FILE = UNIFIED_DIR / "ScenarioCatalog.csv"
BUILD_META_FILE = UNIFIED_DIR / "build_metadata.json"

DATE_COLS_PAPER = [
    "DatePaperSubmitted",
    "DateReviewersFullyAssigned",
    "DateFirstReviewReceived",
    "DateAllReviewsReceived",
    "ReviewPhaseClosedDateAtEnd",
    "AE_RecommendationDate",
    "EIC_DecisionDate",
    "DateDecisionLetterSent",
]

DATE_COLS_ASSIGNMENT = [
    "DateReviewerInvited",
    "DateInvitationAccepted",
    "DateInvitationResolved",
    "DateNoResponseCensor",
    "DateNoResponseTerminal",
    "DateReviewDue",
    "DateReviewSubmitted",
    "DateFirstReminderSent",
    "DateLastReminderSent",
    "ReviewerAssignmentTerminalDateAtEnd",
]

DECISION_ORDER = ["Reject", "Major Revision", "Minor Revision", "Accept"]
STAGE_ORDER = [
    "Waiting: reviewer assignment",
    "In review: waiting for first review",
    "In review: partial reviews received",
    "Waiting: AE recommendation",
    "Waiting: EIC decision",
    "Waiting: decision letter",
    "Resolved: decision sent",
]


# ================================================================
# HELPERS
# ================================================================

def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def to_datetime_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def parse_listish(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [text]


def consensus_decision(series: pd.Series) -> str:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return "Unknown"
    winners = counts[counts == counts.max()].index.tolist()
    return "Split" if len(winners) > 1 else str(winners[0])


def mode_or_default(series: pd.Series, default):
    s = series.dropna()
    if s.empty:
        return default
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else s.iloc[0]


def append_csv_row(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False)


# ================================================================
# SOURCE LOADERS
# ================================================================

@st.cache_data(show_spinner=False)
def load_workflow_sources() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    xlsx = first_existing(WORKFLOW_XLSX_CANDIDATES)
    if xlsx is not None:
        paper = pd.read_excel(xlsx, sheet_name="PaperHeader")
        reviewer = pd.read_excel(xlsx, sheet_name="ReviewerRows")
        return paper, reviewer, str(xlsx)

    paper_csv = first_existing(WORKFLOW_PAPER_CSV_CANDIDATES)
    reviewer_csv = first_existing(WORKFLOW_REVIEWER_CSV_CANDIDATES)
    if paper_csv is not None and reviewer_csv is not None:
        return pd.read_csv(paper_csv), pd.read_csv(reviewer_csv), f"{paper_csv} + {reviewer_csv}"

    raise FileNotFoundError(
        "Could not find EIC-AIssist_peer_review_synth_v12.xlsx or the v12 PaperHeader/ReviewerRows CSV files beside this app."
    )


@st.cache_data(show_spinner=False)
def load_ai_sources() -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    reviews_path = first_existing(PAPER_REVIEWS_CANDIDATES)
    if reviews_path is None:
        raise FileNotFoundError("Could not find PaperReviews.csv.")

    reviews = pd.read_csv(reviews_path)
    reviewers_path = first_existing(REVIEWERS_CANDIDATES)

    if reviewers_path is not None:
        personas = pd.read_csv(reviewers_path)
        persona_source = str(reviewers_path)
    else:
        keep = [c for c in ["ReviewerID", "ReviewerPersona"] if c in reviews.columns]
        personas = reviews[keep].drop_duplicates().copy()
        persona_source = "derived from PaperReviews.csv"

    return reviews, personas, str(reviews_path), persona_source


def normalize_personas(personas: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    p = personas.copy()
    rename_candidates = {
        "Persona": "ReviewerPersona",
        "Focus": "ReviewerFocus",
        "PrimaryFocus": "ReviewerFocus",
        "Bias": "ReviewerBias",
        "ScoreBias": "ReviewerBias",
        "Band": "ReviewerBand",
    }
    for old, new in rename_candidates.items():
        if old in p.columns and new not in p.columns:
            p = p.rename(columns={old: new})

    if "ReviewerID" not in p.columns:
        raise ValueError("Reviewers.csv must contain ReviewerID.")

    p["ReviewerID"] = p["ReviewerID"].astype(str)

    if "ReviewerPersona" not in p.columns and "ReviewerPersona" in reviews.columns:
        lookup = reviews[["ReviewerID", "ReviewerPersona"]].drop_duplicates("ReviewerID")
        lookup["ReviewerID"] = lookup["ReviewerID"].astype(str)
        p = p.merge(lookup, on="ReviewerID", how="left")

    defaults = {
        "ReviewerPersona": "General peer reviewer",
        "ReviewerFocus": "Overall scientific quality",
        "ReviewerBand": "Balanced",
    }
    for col, default in defaults.items():
        if col not in p.columns:
            p[col] = default
        p[col] = p[col].fillna(default).astype(str)

    if "ReviewerBias" not in p.columns:
        p["ReviewerBias"] = 0.0
    p["ReviewerBias"] = pd.to_numeric(p["ReviewerBias"], errors="coerce").fillna(0.0)

    # Ensure every reviewer appearing in the review corpus exists.
    missing_ids = sorted(set(reviews["ReviewerID"].astype(str)) - set(p["ReviewerID"]))
    if missing_ids:
        add = (
            reviews[reviews["ReviewerID"].astype(str).isin(missing_ids)]
            [["ReviewerID", "ReviewerPersona"]]
            .drop_duplicates("ReviewerID")
            .copy()
        )
        add["ReviewerID"] = add["ReviewerID"].astype(str)
        add["ReviewerFocus"] = "Overall scientific quality"
        add["ReviewerBand"] = "Balanced"
        add["ReviewerBias"] = 0.0
        p = pd.concat([p, add], ignore_index=True, sort=False)

    return p.drop_duplicates("ReviewerID").reset_index(drop=True)



# ================================================================
# SIMULATION RETIMING + SHOWCASE SCENARIOS
# ================================================================

def _positive_days(series: pd.Series, fallback: int, low: int = 1, high: int = 180) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    vals = vals[(vals > 0) & (vals <= high)]
    if vals.empty:
        return np.array([fallback], dtype=float)
    return vals.to_numpy(dtype=float)


def _empirical_duration_pools(workflow_paper: pd.DataFrame) -> Dict[str, np.ndarray]:
    wp = to_datetime_columns(workflow_paper, DATE_COLS_PAPER)

    def diff(a, b, fallback, high=180):
        if a in wp.columns and b in wp.columns:
            return _positive_days((wp[b] - wp[a]).dt.total_seconds() / 86400.0, fallback, high=high)
        return np.array([fallback], dtype=float)

    return {
        "assign": diff("DatePaperSubmitted", "DateReviewersFullyAssigned", 10, 90),
        "first": diff("DateReviewersFullyAssigned", "DateFirstReviewReceived", 14, 90),
        "review": diff("DateReviewersFullyAssigned", "DateAllReviewsReceived", 32, 180),
        "ae": diff("DateAllReviewsReceived", "AE_RecommendationDate", 7, 60),
        "eic": diff("AE_RecommendationDate", "EIC_DecisionDate", 5, 60),
        "letter": diff("EIC_DecisionDate", "DateDecisionLetterSent", 3, 30),
    }


def retime_journal_for_active_snapshot(
    papers: pd.DataFrame,
    assignments: pd.DataFrame,
    unified_reviews: pd.DataFrame,
    reviewer_pool: pd.DataFrame,
    workflow_paper: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Re-time the 103-paper corpus around one simulation reference date.

    The original EIC-AIssist data still supplies the empirical duration
    distributions, but the snapshot mix is deliberately weighted toward
    active/submitted manuscripts so the dashboard behaves like a live journal.
    """
    papers = papers.copy()
    assignments = assignments.copy()
    unified_reviews = unified_reviews.copy()
    reviewer_pool = reviewer_pool.copy()

    pools = _empirical_duration_pools(workflow_paper)
    ref = SIMULATION_REFERENCE_DATE

    # Target snapshot distribution for 103 papers. Most are already submitted
    # and actively moving through review/editorial processing, with only a small
    # number stuck in invitation/assignment stages.
    target_counts = {
        "Waiting: reviewer assignment": 5,
        "In review: waiting for first review": 10,
        "In review: partial reviews received": 38,
        "Waiting: AE recommendation": 18,
        "Waiting: EIC decision": 10,
        "Waiting: decision letter": 5,
        "Resolved: decision sent": 17,
    }

    ids = papers["PaperID"].astype(int).tolist()
    stage_targets = []
    for stage, count in target_counts.items():
        stage_targets.extend([stage] * count)
    # Adjust automatically if the corpus is not exactly 103.
    if len(stage_targets) < len(ids):
        stage_targets.extend(["In review: partial reviews received"] * (len(ids) - len(stage_targets)))
    stage_targets = stage_targets[: len(ids)]
    rng.shuffle(stage_targets)

    # Put at least one genuinely high-disagreement paper into the AE stage.
    high_disagreement_ids = papers.loc[
        pd.to_numeric(papers.get("ReviewScoreRange", 0), errors="coerce") >= 2.0,
        "PaperID",
    ].astype(int).tolist()
    if high_disagreement_ids:
        hid = high_disagreement_ids[0]
        hpos = ids.index(hid)
        if stage_targets[hpos] != "Waiting: AE recommendation":
            try:
                swap_pos = stage_targets.index("Waiting: AE recommendation")
                stage_targets[hpos], stage_targets[swap_pos] = stage_targets[swap_pos], stage_targets[hpos]
            except ValueError:
                pass

    scenario_rows = []
    stage_seen = {k: 0 for k in target_counts}

    reviewer_ids = reviewer_pool["ReviewerID"].astype(str).tolist()

    for pos, pid in enumerate(ids):
        stage = stage_targets[pos]
        stage_seen[stage] = stage_seen.get(stage, 0) + 1
        rank = stage_seen[stage]

        pidx = papers.index[papers["PaperID"].astype(int) == pid][0]

        assign_d = int(max(3, round(float(rng.choice(pools["assign"])))))
        first_d = int(max(4, round(float(rng.choice(pools["first"])))))
        review_d = int(max(first_d + 4, round(float(rng.choice(pools["review"])))))
        ae_d = int(max(2, round(float(rng.choice(pools["ae"])))))
        eic_d = int(max(2, round(float(rng.choice(pools["eic"])))))
        letter_d = int(max(1, round(float(rng.choice(pools["letter"])))))

        # Solve the timeline so the chosen stage is true at the reference date.
        if stage == "Waiting: reviewer assignment":
            submitted = ref - pd.Timedelta(days=int(rng.integers(2, min(assign_d, 14) + 1)))
            assigned = submitted + pd.Timedelta(days=assign_d)
        elif stage == "In review: waiting for first review":
            assigned = ref - pd.Timedelta(days=int(rng.integers(1, min(first_d, 10) + 1)))
            submitted = assigned - pd.Timedelta(days=assign_d)
        elif stage == "In review: partial reviews received":
            first = ref - pd.Timedelta(days=int(rng.integers(1, 10)))
            assigned = first - pd.Timedelta(days=first_d)
            submitted = assigned - pd.Timedelta(days=assign_d)
        elif stage == "Waiting: AE recommendation":
            all_reviews = ref - pd.Timedelta(days=int(rng.integers(1, min(ae_d, 8) + 1)))
            assigned = all_reviews - pd.Timedelta(days=review_d)
            submitted = assigned - pd.Timedelta(days=assign_d)
        elif stage == "Waiting: EIC decision":
            ae_date = ref - pd.Timedelta(days=int(rng.integers(1, min(eic_d, 7) + 1)))
            all_reviews = ae_date - pd.Timedelta(days=ae_d)
            assigned = all_reviews - pd.Timedelta(days=review_d)
            submitted = assigned - pd.Timedelta(days=assign_d)
        elif stage == "Waiting: decision letter":
            eic_date = ref - pd.Timedelta(days=int(rng.integers(1, min(letter_d, 5) + 1)))
            ae_date = eic_date - pd.Timedelta(days=eic_d)
            all_reviews = ae_date - pd.Timedelta(days=ae_d)
            assigned = all_reviews - pd.Timedelta(days=review_d)
            submitted = assigned - pd.Timedelta(days=assign_d)
        else:  # resolved
            letter_date = ref - pd.Timedelta(days=int(rng.integers(1, 61)))
            eic_date = letter_date - pd.Timedelta(days=letter_d)
            ae_date = eic_date - pd.Timedelta(days=eic_d)
            all_reviews = ae_date - pd.Timedelta(days=ae_d)
            assigned = all_reviews - pd.Timedelta(days=review_d)
            submitted = assigned - pd.Timedelta(days=assign_d)

        first = assigned + pd.Timedelta(days=first_d)
        all_reviews = assigned + pd.Timedelta(days=review_d)
        ae_date = all_reviews + pd.Timedelta(days=ae_d)
        eic_date = ae_date + pd.Timedelta(days=eic_d)
        letter_date = eic_date + pd.Timedelta(days=letter_d)

        # Enforce the intended snapshot stage exactly.
        if stage == "Waiting: reviewer assignment":
            assigned = max(assigned, ref + pd.Timedelta(days=3))
            first = assigned + pd.Timedelta(days=first_d)
            all_reviews = assigned + pd.Timedelta(days=review_d)
            ae_date = all_reviews + pd.Timedelta(days=ae_d)
            eic_date = ae_date + pd.Timedelta(days=eic_d)
            letter_date = eic_date + pd.Timedelta(days=letter_d)
        elif stage == "In review: waiting for first review":
            first = max(first, ref + pd.Timedelta(days=3))
            all_reviews = max(all_reviews, first + pd.Timedelta(days=4))
            ae_date = all_reviews + pd.Timedelta(days=ae_d)
            eic_date = ae_date + pd.Timedelta(days=eic_d)
            letter_date = eic_date + pd.Timedelta(days=letter_d)
        elif stage == "In review: partial reviews received":
            first = min(first, ref - pd.Timedelta(days=1))
            all_reviews = max(all_reviews, ref + pd.Timedelta(days=5))
            ae_date = all_reviews + pd.Timedelta(days=ae_d)
            eic_date = ae_date + pd.Timedelta(days=eic_d)
            letter_date = eic_date + pd.Timedelta(days=letter_d)
        elif stage == "Waiting: AE recommendation":
            all_reviews = min(all_reviews, ref - pd.Timedelta(days=1))
            ae_date = max(ae_date, ref + pd.Timedelta(days=4))
            eic_date = ae_date + pd.Timedelta(days=eic_d)
            letter_date = eic_date + pd.Timedelta(days=letter_d)
        elif stage == "Waiting: EIC decision":
            ae_date = min(ae_date, ref - pd.Timedelta(days=1))
            eic_date = max(eic_date, ref + pd.Timedelta(days=4))
            letter_date = eic_date + pd.Timedelta(days=letter_d)
        elif stage == "Waiting: decision letter":
            eic_date = min(eic_date, ref - pd.Timedelta(days=1))
            letter_date = max(letter_date, ref + pd.Timedelta(days=3))
        elif stage == "Resolved: decision sent":
            letter_date = min(letter_date, ref - pd.Timedelta(days=1))

        papers.at[pidx, "DatePaperSubmitted"] = submitted
        papers.at[pidx, "DateReviewersFullyAssigned"] = assigned
        papers.at[pidx, "DateFirstReviewReceived"] = first
        papers.at[pidx, "DateAllReviewsReceived"] = all_reviews
        papers.at[pidx, "ReviewPhaseClosedDateAtEnd"] = all_reviews
        papers.at[pidx, "AE_RecommendationDate"] = ae_date
        papers.at[pidx, "EIC_DecisionDate"] = eic_date
        papers.at[pidx, "DateDecisionLetterSent"] = letter_date
        papers.at[pidx, "TotalTime_SubmissionToDecision_Days"] = int((letter_date - submitted).days)
        papers.at[pidx, "SimulationReferenceDate"] = ref
        papers.at[pidx, "TargetStageAtReference"] = stage

        pa_idx = assignments.index[assignments["PaperID"].astype(int) == pid].tolist()
        generated_ids = set(
            unified_reviews.loc[unified_reviews["PaperID"].astype(int) == pid, "AssignmentID"]
            .dropna().astype(str).tolist()
        )
        generated_idx = [i for i in pa_idx if str(assignments.at[i, "AssignmentID"]) in generated_ids]
        extra_idx = [i for i in pa_idx if i not in generated_idx]

        # Base reviewer timing.
        for j, aidx in enumerate(generated_idx):
            invite = submitted + pd.Timedelta(days=max(1, int(round(assign_d * (j + 1) / (len(generated_idx) + 1)))))
            accept = invite + pd.Timedelta(days=int(rng.integers(1, 4)))
            if accept > assigned:
                accept = assigned - pd.Timedelta(hours=12)
            due = accept + pd.Timedelta(days=int(rng.integers(21, 36)))

            assignments.at[aidx, "InviteOutcome"] = "accept"
            assignments.at[aidx, "DateReviewerInvited"] = invite
            assignments.at[aidx, "DateInvitationAccepted"] = accept
            assignments.at[aidx, "DateInvitationResolved"] = accept
            assignments.at[aidx, "DateReviewDue"] = due
            assignments.at[aidx, "DateNoResponseTerminal"] = pd.NaT

            if stage == "Waiting: reviewer assignment":
                # Keep most invitations unresolved at the snapshot.
                assignments.at[aidx, "DateInvitationAccepted"] = ref + pd.Timedelta(days=2 + j)
                assignments.at[aidx, "DateInvitationResolved"] = ref + pd.Timedelta(days=2 + j)
                review_submit = ref + pd.Timedelta(days=30 + 3 * j)
            elif stage == "In review: waiting for first review":
                review_submit = ref + pd.Timedelta(days=4 + 4 * j)
            elif stage == "In review: partial reviews received":
                # At least one submitted, at least one still active where possible.
                if j == 0:
                    review_submit = ref - pd.Timedelta(days=int(rng.integers(1, 8)))
                else:
                    review_submit = ref + pd.Timedelta(days=int(rng.integers(3, 20)))
            else:
                # Completed-review stages: all persona reviews are in before all_reviews.
                if len(generated_idx) <= 1:
                    frac = 1.0
                else:
                    frac = j / (len(generated_idx) - 1)
                span = max(2, int((all_reviews - first).days))
                review_submit = first + pd.Timedelta(days=int(round(frac * span)))
                if review_submit > all_reviews:
                    review_submit = all_reviews

            assignments.at[aidx, "DateReviewSubmitted"] = review_submit
            assignments.at[aidx, "LateSubmissionFlag"] = "yes" if pd.notna(due) and review_submit > due else "no"

        # Extra declined/no-response rows remain journal invitations but do not create reviews.
        for j, aidx in enumerate(extra_idx):
            invite = submitted + pd.Timedelta(days=1 + j)
            outcome = str(assignments.at[aidx, "InviteOutcome"]).lower().strip()
            assignments.at[aidx, "DateReviewerInvited"] = invite
            assignments.at[aidx, "DateReviewSubmitted"] = pd.NaT
            assignments.at[aidx, "DateReviewDue"] = pd.NaT
            if outcome == "decline":
                assignments.at[aidx, "DateInvitationResolved"] = invite + pd.Timedelta(days=2)
                assignments.at[aidx, "DateInvitationAccepted"] = pd.NaT
            elif outcome == "no_response":
                terminal = invite + pd.Timedelta(days=7)
                assignments.at[aidx, "DateInvitationResolved"] = terminal
                assignments.at[aidx, "DateNoResponseTerminal"] = terminal
                assignments.at[aidx, "DateInvitationAccepted"] = pd.NaT

        # Showcase scenarios deliberately cover the intervention space.
        scenario = "Normal active workflow"
        expected_action = "Continue monitoring"
        expected_outcome = "Manuscript progresses without intervention"

        if stage == "Waiting: reviewer assignment":
            scenario = "Reviewer shortage / invitations pending"
            expected_action = "Invite or assign additional reviewers"
            expected_outcome = "Target reviewer count is reached"

        elif stage == "In review: waiting for first review":
            if rank == 1:
                scenario = "Upcoming review deadline"
                expected_action = "Send friendly deadline reminder"
                expected_outcome = "Reviewer submits before deadline"
            elif rank == 2:
                scenario = "No response to invitation"
                expected_action = "Close invitation and invite replacement reviewer"
                expected_outcome = "Replacement reviewer accepts"
            else:
                scenario = "Accepted reviewers, no review received yet"
                expected_action = "Monitor or remind if due date approaches"
                expected_outcome = "First review arrives"

        elif stage == "In review: partial reviews received":
            if rank == 1:
                scenario = "Review overdue 1–7 days"
                expected_action = "Send standard reviewer reminder"
                expected_outcome = "Outstanding review is submitted"
            elif rank == 2:
                scenario = "Review overdue 8–14 days"
                expected_action = "Send firm reminder and CC EIC"
                expected_outcome = "Reviewer confirms or submits"
            elif rank == 3:
                scenario = "Review overdue >14 days"
                expected_action = "Final reminder and prepare replacement reviewer"
                expected_outcome = "Replacement prevents further delay"
            elif rank == 4:
                scenario = "Reviewer declined"
                expected_action = "Invite replacement reviewer"
                expected_outcome = "Review panel returns to target size"
            elif rank == 5:
                scenario = "High reviewer workload"
                expected_action = "Avoid overloaded reviewer and assign alternative"
                expected_outcome = "Lower delay risk"
            else:
                scenario = "Partial reviews received"
                expected_action = "Monitor outstanding reviewers"
                expected_outcome = "All required reviews arrive"

        elif stage == "Waiting: AE recommendation":
            if float(papers.at[pidx, "ReviewScoreRange"] or 0) >= 2.0 or rank == 1:
                scenario = "High reviewer disagreement"
                expected_action = "Invite adjudicating reviewer or perform AE synthesis"
                expected_outcome = "Disagreement is resolved before EIC decision"
            elif rank == 2:
                scenario = "Reviews agree and are complete"
                expected_action = "AE prepares recommendation"
                expected_outcome = "Recommendation moves to EIC"
            else:
                scenario = "AE recommendation delayed"
                expected_action = "Prioritize AE recommendation"
                expected_outcome = "Editorial bottleneck is cleared"

        elif stage == "Waiting: EIC decision":
            scenario = "EIC decision pending"
            expected_action = "Prioritize EIC decision"
            expected_outcome = "Decision is recorded"

        elif stage == "Waiting: decision letter":
            scenario = "Decision made, letter pending"
            expected_action = "Finalize and send decision letter"
            expected_outcome = "Author receives final editorial outcome"

        elif stage == "Resolved: decision sent":
            scenario = "Resolved manuscript"
            expected_action = "No action"
            expected_outcome = "Workflow complete"

        # Force concrete reviewer states for the showcase cases.
        if generated_idx:
            chosen = generated_idx[-1]
            if scenario == "Upcoming review deadline":
                assignments.at[chosen, "DateReviewSubmitted"] = ref + pd.Timedelta(days=3)
                assignments.at[chosen, "DateReviewDue"] = ref + pd.Timedelta(days=3)
            elif scenario == "Review overdue 1–7 days":
                assignments.at[chosen, "DateReviewSubmitted"] = ref + pd.Timedelta(days=5)
                assignments.at[chosen, "DateReviewDue"] = ref - pd.Timedelta(days=4)
            elif scenario == "Review overdue 8–14 days":
                assignments.at[chosen, "DateReviewSubmitted"] = ref + pd.Timedelta(days=5)
                assignments.at[chosen, "DateReviewDue"] = ref - pd.Timedelta(days=10)
            elif scenario == "Review overdue >14 days":
                assignments.at[chosen, "DateReviewSubmitted"] = ref + pd.Timedelta(days=12)
                assignments.at[chosen, "DateReviewDue"] = ref - pd.Timedelta(days=18)
            elif scenario == "High reviewer workload":
                assignments.at[chosen, "ReviewerWorkloadAtInvite"] = 12

        # Add a no-response or decline invitation if the chosen scenario needs one.
        needs_extra = scenario in {"No response to invitation", "Reviewer declined"}
        if needs_extra:
            existing_reviewers = set(assignments.loc[assignments["PaperID"].astype(int) == pid, "ReviewerID"].astype(str))
            candidate = next((rid for rid in reviewer_ids if rid not in existing_reviewers), None)
            if candidate is not None:
                new_id = f"X-{pid:03d}-{len(extra_idx)+1:02d}"
                invite = submitted + pd.Timedelta(days=2)
                if scenario == "No response to invitation":
                    outcome = "no_response"
                    terminal = ref - pd.Timedelta(days=1)
                    row = {
                        "AssignmentID": new_id, "PaperID": pid, "ReviewerID": candidate,
                        "InviteOutcome": outcome, "ReviewerType": "External",
                        "ReviewerReliabilityTier": "standard", "ReviewerWorkloadAtInvite": 4,
                        "NumRemindersSent": 1, "LateSubmissionFlag": "",
                        "ReviewerDisagreementScore": np.nan,
                        "DateReviewerInvited": invite, "DateInvitationAccepted": pd.NaT,
                        "DateInvitationResolved": terminal, "DateNoResponseTerminal": terminal,
                        "DateReviewDue": pd.NaT, "DateReviewSubmitted": pd.NaT,
                    }
                else:
                    outcome = "decline"
                    resolved = ref - pd.Timedelta(days=1)
                    row = {
                        "AssignmentID": new_id, "PaperID": pid, "ReviewerID": candidate,
                        "InviteOutcome": outcome, "ReviewerType": "External",
                        "ReviewerReliabilityTier": "standard", "ReviewerWorkloadAtInvite": 3,
                        "NumRemindersSent": 0, "LateSubmissionFlag": "",
                        "ReviewerDisagreementScore": np.nan,
                        "DateReviewerInvited": invite, "DateInvitationAccepted": pd.NaT,
                        "DateInvitationResolved": resolved, "DateNoResponseTerminal": pd.NaT,
                        "DateReviewDue": pd.NaT, "DateReviewSubmitted": pd.NaT,
                    }
                assignments = pd.concat([assignments, pd.DataFrame([row])], ignore_index=True, sort=False)

        papers.at[pidx, "ShowcaseScenario"] = scenario
        papers.at[pidx, "ExpectedAction"] = expected_action
        papers.at[pidx, "ExpectedOutcome"] = expected_outcome

        scenario_rows.append({
            "PaperID": pid,
            "PaperTitle": papers.at[pidx, "PaperTitle"],
            "TargetStageAtReference": stage,
            "ShowcaseScenario": scenario,
            "ExpectedAction": expected_action,
            "ExpectedOutcome": expected_outcome,
        })

    assignments = to_datetime_columns(assignments, DATE_COLS_ASSIGNMENT)

    # Keep review content linked to its assignment timing. Future reviews remain
    # in the corpus but are hidden from as-of Paper/Reviewer POVs until submitted.
    submit_map = assignments.set_index("AssignmentID")["DateReviewSubmitted"].to_dict()
    unified_reviews["DateReviewSubmitted"] = unified_reviews["AssignmentID"].map(submit_map)
    unified_reviews = to_datetime_columns(unified_reviews, ["DateReviewSubmitted"])

    return papers, assignments, unified_reviews, pd.DataFrame(scenario_rows)

# ================================================================
# UNIFIED JOURNAL BUILDER
# ================================================================

def build_unified_journal(force: bool = False) -> Dict:
    required = [
        PAPERS_FILE,
        EDITORS_FILE,
        REVIEWERS_FILE,
        ASSIGNMENTS_FILE,
        REVIEWS_FILE,
        DECISIONS_FILE,
    ]

    if not force and all(p.exists() for p in required):
        try:
            meta = json.loads(BUILD_META_FILE.read_text()) if BUILD_META_FILE.exists() else {}
        except Exception:
            meta = {}
        if meta.get("build_version") == BUILD_VERSION:
            return {"status": "existing", "dir": str(UNIFIED_DIR)}
        # Existing tables from an older app build are regenerated automatically.

    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    workflow_paper, workflow_rev, workflow_source = load_workflow_sources()
    paper_reviews, personas_raw, review_source, persona_source = load_ai_sources()

    workflow_paper = to_datetime_columns(workflow_paper, DATE_COLS_PAPER)
    workflow_rev = to_datetime_columns(workflow_rev, DATE_COLS_ASSIGNMENT)

    if "SubmissionRound" not in workflow_paper.columns:
        workflow_paper["SubmissionRound"] = 1
    if "SubmissionRound" not in workflow_rev.columns:
        workflow_rev["SubmissionRound"] = 1

    paper_reviews = paper_reviews.copy()
    paper_reviews["PaperID"] = pd.to_numeric(paper_reviews["PaperID"], errors="coerce")
    paper_reviews["Score"] = pd.to_numeric(paper_reviews["Score"], errors="coerce")
    paper_reviews = paper_reviews.dropna(subset=["PaperID", "PaperTitle", "ReviewerID"]).copy()
    paper_reviews["PaperID"] = paper_reviews["PaperID"].astype(int)
    paper_reviews["ReviewerID"] = paper_reviews["ReviewerID"].astype(str)

    personas = normalize_personas(personas_raw, paper_reviews)

    content_papers = (
        paper_reviews[["PaperID", "PaperTitle"]]
        .drop_duplicates("PaperID")
        .sort_values("PaperID")
        .reset_index(drop=True)
    )

    # Sample real-looking workflow templates from the original v12 paper rows.
    templates = workflow_paper[workflow_paper["DatePaperSubmitted"].notna()].copy()
    if templates.empty:
        templates = workflow_paper.copy()
    sample_idx = rng.choice(
        np.arange(len(templates)),
        size=len(content_papers),
        replace=len(templates) < len(content_papers),
    )
    sampled = templates.iloc[sample_idx].reset_index(drop=True)

    review_agg = (
        paper_reviews.groupby(["PaperID", "PaperTitle"])
        .agg(
            ReviewCount=("ReviewerID", "count"),
            MeanReviewScore=("Score", "mean"),
            MedianReviewScore=("Score", "median"),
            ReviewScoreStd=("Score", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
            MinReviewScore=("Score", "min"),
            MaxReviewScore=("Score", "max"),
            DecisionDiversity=("Decision", "nunique"),
        )
        .reset_index()
    )
    review_agg["ReviewScoreRange"] = review_agg["MaxReviewScore"] - review_agg["MinReviewScore"]
    consensus = paper_reviews.groupby("PaperID")["Decision"].apply(consensus_decision).to_dict()

    template_cols = [
        "SubmissionRound",
        "JournalSection",
        "PaperStatusOnSubmission",
        "HandlingAssociateEditorID",
        "HandlingEIC_ID",
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "ReviewPhaseClosedDateAtEnd",
        "AE_RecommendationDate",
        "AE_Recommendation",
        "EIC_DecisionDate",
        "EIC_Decision",
        "DateDecisionLetterSent",
        "FinalDecisionOutcome",
        "TotalTime_SubmissionToDecision_Days",
    ]

    paper_rows = []
    for i, cp in content_papers.iterrows():
        t = sampled.iloc[i]
        pid = int(cp["PaperID"])
        agg = review_agg[review_agg["PaperID"] == pid].iloc[0]

        row = {
            "PaperID": pid,
            "PaperTitle": cp["PaperTitle"],
            "SourceWorkflowPaperID": clean_id(t["PaperID"]),
            "SourceWorkflowRound": int(pd.to_numeric(pd.Series([t.get("SubmissionRound", 1)]), errors="coerce").fillna(1).iloc[0]),
        }
        for col in template_cols:
            row[col] = t.get(col, np.nan)

        # The actual AI Scientist persona-review count becomes the journal target.
        row["TargetNumberOfReviewers"] = int(agg["ReviewCount"])
        row["AIReviewCount"] = int(agg["ReviewCount"])
        row["MeanReviewScore"] = float(agg["MeanReviewScore"])
        row["MedianReviewScore"] = float(agg["MedianReviewScore"])
        row["ReviewScoreStd"] = float(agg["ReviewScoreStd"])
        row["ReviewScoreRange"] = float(agg["ReviewScoreRange"])
        row["DecisionDiversity"] = int(agg["DecisionDiversity"])
        row["ConsensusDecision"] = consensus.get(pid, "Unknown")

        if pd.isna(row.get("JournalSection")) or not str(row.get("JournalSection", "")).strip():
            row["JournalSection"] = "AI / Machine Learning"
        if pd.isna(row.get("HandlingAssociateEditorID")) or not str(row.get("HandlingAssociateEditorID", "")).strip():
            row["HandlingAssociateEditorID"] = f"AP-{1 + i % 8:02d}"
        if pd.isna(row.get("HandlingEIC_ID")) or not str(row.get("HandlingEIC_ID", "")).strip():
            row["HandlingEIC_ID"] = f"EIC-{1 + i % 3:02d}"

        paper_rows.append(row)

    papers = to_datetime_columns(pd.DataFrame(paper_rows), DATE_COLS_PAPER)

    # Reviewer pool = persistent personas + realistic workflow attributes.
    reviewer_pool = personas.copy()
    if not workflow_rev.empty:
        sampled_rev = workflow_rev.sample(
            n=len(reviewer_pool),
            replace=len(workflow_rev) < len(reviewer_pool),
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)
        for col in ["ReviewerType", "ReviewerReliabilityTier", "ReviewerWorkloadAtInvite"]:
            if col in sampled_rev.columns:
                reviewer_pool[col] = sampled_rev[col].values

    if "ReviewerType" not in reviewer_pool.columns:
        reviewer_pool["ReviewerType"] = "External"
    if "ReviewerReliabilityTier" not in reviewer_pool.columns:
        reviewer_pool["ReviewerReliabilityTier"] = "standard"
    if "ReviewerWorkloadAtInvite" not in reviewer_pool.columns:
        reviewer_pool["ReviewerWorkloadAtInvite"] = 3
    reviewer_pool["ReviewerRole"] = "Peer Reviewer"

    # Prepare source assignment matching.
    workflow_rev["_SourcePID"] = workflow_rev["PaperID"].map(clean_id)
    workflow_rev["_SourceRound"] = pd.to_numeric(workflow_rev["SubmissionRound"], errors="coerce").fillna(1).astype(int)
    completed_pool = workflow_rev[workflow_rev.get("DateReviewSubmitted", pd.Series(index=workflow_rev.index, dtype="datetime64[ns]")).notna()].copy()

    reviewer_ids = reviewer_pool["ReviewerID"].astype(str).tolist()
    assignment_rows = []
    review_rows = []

    # Lookup source-paper submission date for borrowed assignment timing.
    source_submit_lookup = {}
    for _, row in workflow_paper.iterrows():
        key = (clean_id(row["PaperID"]), int(pd.to_numeric(pd.Series([row.get("SubmissionRound", 1)]), errors="coerce").fillna(1).iloc[0]))
        source_submit_lookup[key] = row.get("DatePaperSubmitted", pd.NaT)

    for _, p in papers.iterrows():
        pid = int(p["PaperID"])
        src_pid = clean_id(p["SourceWorkflowPaperID"])
        src_round = int(p["SourceWorkflowRound"])
        unified_submit = p.get("DatePaperSubmitted", pd.NaT)

        src_assign = workflow_rev[
            (workflow_rev["_SourcePID"] == src_pid)
            & (workflow_rev["_SourceRound"] == src_round)
        ].copy()

        generated = (
            paper_reviews[paper_reviews["PaperID"] == pid]
            .sort_values("ReviewerID")
            .reset_index(drop=True)
        )

        submitted_templates = src_assign.copy()
        if "DateReviewSubmitted" in submitted_templates.columns:
            submitted_templates = submitted_templates[submitted_templates["DateReviewSubmitted"].notna()].copy()
        else:
            submitted_templates = submitted_templates.iloc[0:0].copy()

        used = set()
        counter = 0

        # Every generated persona review gets a real-looking accepted assignment.
        for j, review in generated.iterrows():
            rid = str(review["ReviewerID"])
            used.add(rid)
            counter += 1
            assignment_id = f"A-{pid:03d}-{counter:02d}"

            if j < len(submitted_templates):
                template = submitted_templates.iloc[j]
                same_source = True
            elif not completed_pool.empty:
                template = completed_pool.sample(1, random_state=RANDOM_SEED + pid * 100 + j).iloc[0]
                same_source = False
            else:
                template = pd.Series(dtype=object)
                same_source = False

            a = {
                "AssignmentID": assignment_id,
                "PaperID": pid,
                "ReviewerID": rid,
                "InviteOutcome": "accept",
                "ReviewerType": template.get("ReviewerType", "External"),
                "ReviewerReliabilityTier": template.get("ReviewerReliabilityTier", "standard"),
                "ReviewerWorkloadAtInvite": template.get("ReviewerWorkloadAtInvite", 3),
                "NumRemindersSent": template.get("NumRemindersSent", 0),
                "LateSubmissionFlag": template.get("LateSubmissionFlag", ""),
                "ReviewerDisagreementScore": template.get("ReviewerDisagreementScore", np.nan),
            }

            if same_source:
                for col in DATE_COLS_ASSIGNMENT:
                    a[col] = template.get(col, pd.NaT)
            else:
                borrowed_key = (
                    clean_id(template.get("PaperID", "")),
                    int(pd.to_numeric(pd.Series([template.get("SubmissionRound", 1)]), errors="coerce").fillna(1).iloc[0]),
                )
                borrowed_submit = source_submit_lookup.get(borrowed_key, pd.NaT)
                for col in DATE_COLS_ASSIGNMENT:
                    dt = template.get(col, pd.NaT)
                    if pd.notna(dt) and pd.notna(borrowed_submit) and pd.notna(unified_submit):
                        a[col] = unified_submit + (dt - borrowed_submit)
                    else:
                        a[col] = pd.NaT

                # Safe fallbacks if a borrowed source row is incomplete.
                if pd.isna(a.get("DateReviewerInvited")) and pd.notna(unified_submit):
                    a["DateReviewerInvited"] = unified_submit + pd.Timedelta(days=3 + j)
                if pd.isna(a.get("DateInvitationAccepted")) and pd.notna(a.get("DateReviewerInvited")):
                    a["DateInvitationAccepted"] = a["DateReviewerInvited"] + pd.Timedelta(days=2)
                if pd.isna(a.get("DateReviewDue")) and pd.notna(a.get("DateInvitationAccepted")):
                    a["DateReviewDue"] = a["DateInvitationAccepted"] + pd.Timedelta(days=28)
                if pd.isna(a.get("DateReviewSubmitted")) and pd.notna(a.get("DateInvitationAccepted")):
                    a["DateReviewSubmitted"] = a["DateInvitationAccepted"] + pd.Timedelta(days=24 + 3 * j)
                if pd.isna(a.get("DateInvitationResolved")):
                    a["DateInvitationResolved"] = a.get("DateInvitationAccepted", pd.NaT)

            assignment_rows.append(a)

            rr = review.to_dict()
            rr.update(
                {
                    "ReviewID": f"REV-{pid:03d}-{j+1:02d}",
                    "AssignmentID": assignment_id,
                    "PaperID": pid,
                    "DateReviewSubmitted": a.get("DateReviewSubmitted", pd.NaT),
                    "HandlingAssociateEditorID": p["HandlingAssociateEditorID"],
                    "HandlingEIC_ID": p["HandlingEIC_ID"],
                }
            )
            review_rows.append(rr)

        # Preserve declined/no-response invitation behavior as extra assignments.
        extras = src_assign.copy()
        if "DateReviewSubmitted" in extras.columns:
            extras = extras[extras["DateReviewSubmitted"].isna()].copy()

        available = [rid for rid in reviewer_ids if rid not in used]
        for _, template in extras.iterrows():
            if not available:
                break
            rid = available.pop(0)
            counter += 1
            outcome = str(template.get("InviteOutcome", "")).strip().lower()
            if outcome not in {"accept", "decline", "no_response"}:
                outcome = "no_response"

            a = {
                "AssignmentID": f"A-{pid:03d}-{counter:02d}",
                "PaperID": pid,
                "ReviewerID": rid,
                "InviteOutcome": outcome,
                "ReviewerType": template.get("ReviewerType", "External"),
                "ReviewerReliabilityTier": template.get("ReviewerReliabilityTier", "standard"),
                "ReviewerWorkloadAtInvite": template.get("ReviewerWorkloadAtInvite", 3),
                "NumRemindersSent": template.get("NumRemindersSent", 0),
                "LateSubmissionFlag": template.get("LateSubmissionFlag", ""),
                "ReviewerDisagreementScore": template.get("ReviewerDisagreementScore", np.nan),
            }
            for col in DATE_COLS_ASSIGNMENT:
                a[col] = template.get(col, pd.NaT)
            assignment_rows.append(a)

    assignments = to_datetime_columns(pd.DataFrame(assignment_rows), DATE_COLS_ASSIGNMENT)
    unified_reviews = to_datetime_columns(pd.DataFrame(review_rows), ["DateReviewSubmitted"])

    # Reconcile first/all review dates and preserve the original downstream gaps.
    for idx, p in papers.iterrows():
        pid = int(p["PaperID"])
        submitted = assignments[
            (assignments["PaperID"] == pid)
            & assignments["DateReviewSubmitted"].notna()
        ]["DateReviewSubmitted"]
        if submitted.empty:
            continue

        new_first = submitted.min()
        new_all = submitted.max()
        old_all = p.get("DateAllReviewsReceived", pd.NaT)
        old_ae = p.get("AE_RecommendationDate", pd.NaT)
        old_eic = p.get("EIC_DecisionDate", pd.NaT)
        old_letter = p.get("DateDecisionLetterSent", pd.NaT)

        ae_gap = old_ae - old_all if pd.notna(old_ae) and pd.notna(old_all) else pd.NaT
        eic_gap = old_eic - old_ae if pd.notna(old_eic) and pd.notna(old_ae) else pd.NaT
        letter_gap = old_letter - old_eic if pd.notna(old_letter) and pd.notna(old_eic) else pd.NaT

        papers.at[idx, "DateFirstReviewReceived"] = new_first
        papers.at[idx, "DateAllReviewsReceived"] = new_all
        papers.at[idx, "ReviewPhaseClosedDateAtEnd"] = new_all

        if pd.notna(ae_gap):
            papers.at[idx, "AE_RecommendationDate"] = new_all + ae_gap
        if pd.notna(eic_gap) and pd.notna(papers.at[idx, "AE_RecommendationDate"]):
            papers.at[idx, "EIC_DecisionDate"] = papers.at[idx, "AE_RecommendationDate"] + eic_gap
        if pd.notna(letter_gap) and pd.notna(papers.at[idx, "EIC_DecisionDate"]):
            papers.at[idx, "DateDecisionLetterSent"] = papers.at[idx, "EIC_DecisionDate"] + letter_gap

        if pd.notna(papers.at[idx, "DateDecisionLetterSent"]) and pd.notna(papers.at[idx, "DatePaperSubmitted"]):
            papers.at[idx, "TotalTime_SubmissionToDecision_Days"] = (
                papers.at[idx, "DateDecisionLetterSent"] - papers.at[idx, "DatePaperSubmitted"]
            ).days

    # Re-time the unified corpus around a live-journal snapshot and inject
    # representative intervention scenarios while preserving empirical duration shapes.
    papers, assignments, unified_reviews, scenario_catalog = retime_journal_for_active_snapshot(
        papers, assignments, unified_reviews, reviewer_pool, workflow_paper, rng
    )

    # Reviewer summary stats.
    if not unified_reviews.empty:
        reviewer_stats = (
            unified_reviews.groupby("ReviewerID")
            .agg(
                ReviewsSubmitted=("ReviewID", "count"),
                MeanScore=("Score", "mean"),
                MedianScore=("Score", "median"),
                ScoreStd=("Score", lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)),
            )
            .reset_index()
        )
        reviewer_pool = reviewer_pool.merge(reviewer_stats, on="ReviewerID", how="left")

    # Editors.
    editor_rows = []
    for eid in sorted(papers["HandlingAssociateEditorID"].dropna().astype(str).unique()):
        sub = papers[papers["HandlingAssociateEditorID"].astype(str) == eid]
        editor_rows.append(
            {
                "EditorID": eid,
                "Role": "AE",
                "JournalSection": mode_or_default(sub["JournalSection"], "AI / Machine Learning"),
                "PaperCount": len(sub),
            }
        )
    for eid in sorted(papers["HandlingEIC_ID"].dropna().astype(str).unique()):
        sub = papers[papers["HandlingEIC_ID"].astype(str) == eid]
        editor_rows.append(
            {
                "EditorID": eid,
                "Role": "EIC",
                "JournalSection": mode_or_default(sub["JournalSection"], "AI / Machine Learning"),
                "PaperCount": len(sub),
            }
        )
    editors = pd.DataFrame(editor_rows)

    decision_cols = [
        "PaperID",
        "HandlingAssociateEditorID",
        "HandlingEIC_ID",
        "AE_RecommendationDate",
        "AE_Recommendation",
        "EIC_DecisionDate",
        "EIC_Decision",
        "DateDecisionLetterSent",
        "FinalDecisionOutcome",
        "ConsensusDecision",
        "MeanReviewScore",
        "ReviewScoreStd",
        "ReviewScoreRange",
    ]
    decisions = papers[[c for c in decision_cols if c in papers.columns]].copy()

    # Persist core unified tables.
    papers.to_csv(PAPERS_FILE, index=False)
    editors.to_csv(EDITORS_FILE, index=False)
    reviewer_pool.to_csv(REVIEWERS_FILE, index=False)
    assignments.to_csv(ASSIGNMENTS_FILE, index=False)
    unified_reviews.to_csv(REVIEWS_FILE, index=False)
    decisions.to_csv(DECISIONS_FILE, index=False)
    scenario_catalog.to_csv(SCENARIO_FILE, index=False)

    if not MESSAGES_FILE.exists():
        pd.DataFrame(
            columns=[
                "MessageID", "CreatedAt", "PaperID", "AssignmentID",
                "FromEditorID", "ToReviewerID", "CCEditorID", "Tone",
                "Scenario", "Subject", "Body",
            ]
        ).to_csv(MESSAGES_FILE, index=False)

    if not ACTION_LOG_FILE.exists():
        pd.DataFrame(
            columns=[
                "ActionID", "CreatedAt", "PaperID", "ActorID",
                "Action", "Reason", "PriorityScore",
            ]
        ).to_csv(ACTION_LOG_FILE, index=False)

    metadata = {
        "build_version": BUILD_VERSION,
        "workflow_source": workflow_source,
        "paper_reviews_source": review_source,
        "personas_source": persona_source,
        "papers_built": int(len(papers)),
        "assignments_built": int(len(assignments)),
        "reviews_built": int(len(unified_reviews)),
        "random_seed": RANDOM_SEED,
        "simulation_reference_date": str(SIMULATION_REFERENCE_DATE.date()),
        "scenario_rows": int(len(scenario_catalog)),
    }
    BUILD_META_FILE.write_text(json.dumps(metadata, indent=2))
    return metadata


@st.cache_data(show_spinner=False)
def load_unified_tables():
    papers = to_datetime_columns(pd.read_csv(PAPERS_FILE), DATE_COLS_PAPER)
    editors = pd.read_csv(EDITORS_FILE)
    reviewers = pd.read_csv(REVIEWERS_FILE)
    assignments = to_datetime_columns(pd.read_csv(ASSIGNMENTS_FILE), DATE_COLS_ASSIGNMENT)
    reviews = to_datetime_columns(pd.read_csv(REVIEWS_FILE), ["DateReviewSubmitted"])
    decisions = to_datetime_columns(
        pd.read_csv(DECISIONS_FILE),
        ["AE_RecommendationDate", "EIC_DecisionDate", "DateDecisionLetterSent"],
    )
    return papers, editors, reviewers, assignments, reviews, decisions


def clear_unified_cache():
    try:
        load_unified_tables.clear()
    except Exception:
        pass

# ================================================================
# STATE / SNAPSHOT LOGIC
# ================================================================

def default_snapshot_date(papers: pd.DataFrame) -> pd.Timestamp:
    date_series = []
    for col in DATE_COLS_PAPER:
        if col in papers.columns:
            date_series.append(pd.to_datetime(papers[col], errors="coerce").dropna())
    if not date_series:
        return SIMULATION_REFERENCE_DATE

    all_dates = pd.concat(date_series)
    if all_dates.empty:
        return SIMULATION_REFERENCE_DATE

    return min(max(SIMULATION_REFERENCE_DATE, all_dates.min()), all_dates.max())

def stage_as_of_row(row: pd.Series, snapshot: pd.Timestamp) -> Tuple[str, pd.Timestamp]:
    submitted = row.get("DatePaperSubmitted", pd.NaT)
    if pd.isna(submitted) or submitted > snapshot:
        return "Not yet submitted", pd.NaT

    def happened(col):
        dt = row.get(col, pd.NaT)
        return pd.notna(dt) and dt <= snapshot

    if happened("DateDecisionLetterSent"):
        return "Resolved: decision sent", row.get("DateDecisionLetterSent")
    if happened("EIC_DecisionDate"):
        return "Waiting: decision letter", row.get("EIC_DecisionDate")
    if happened("AE_RecommendationDate"):
        return "Waiting: EIC decision", row.get("AE_RecommendationDate")
    if happened("DateAllReviewsReceived") or happened("ReviewPhaseClosedDateAtEnd"):
        start = row.get("DateAllReviewsReceived", row.get("ReviewPhaseClosedDateAtEnd", pd.NaT))
        return "Waiting: AE recommendation", start
    if happened("DateFirstReviewReceived"):
        return "In review: partial reviews received", row.get("DateFirstReviewReceived")
    if happened("DateReviewersFullyAssigned"):
        return "In review: waiting for first review", row.get("DateReviewersFullyAssigned")
    return "Waiting: reviewer assignment", submitted


def build_paper_state(papers: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    out = papers.copy()
    state = out.apply(stage_as_of_row, axis=1, snapshot=snapshot, result_type="expand")
    out["StageAsOf"] = state[0]
    out["StageStartDateAsOf"] = pd.to_datetime(state[1], errors="coerce")
    out["DaysSinceSubmission"] = (
        snapshot - pd.to_datetime(out["DatePaperSubmitted"], errors="coerce")
    ).dt.days.clip(lower=0)
    out["DaysInCurrentStage"] = (
        snapshot - out["StageStartDateAsOf"]
    ).dt.days.clip(lower=0)
    out["ResolutionStateAsOf"] = np.where(
        out["StageAsOf"].eq("Resolved: decision sent"), "Resolved", "Active"
    )
    out.loc[out["StageAsOf"].eq("Not yet submitted"), "ResolutionStateAsOf"] = "Future"
    return out


def assignment_status_as_of(assignments: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    out = to_datetime_columns(assignments, DATE_COLS_ASSIGNMENT)

    def classify(row):
        invited = row.get("DateReviewerInvited", pd.NaT)
        accepted = row.get("DateInvitationAccepted", pd.NaT)
        resolved = row.get("DateInvitationResolved", pd.NaT)
        due = row.get("DateReviewDue", pd.NaT)
        submitted = row.get("DateReviewSubmitted", pd.NaT)
        noresp_terminal = row.get("DateNoResponseTerminal", pd.NaT)
        outcome = str(row.get("InviteOutcome", "")).strip().lower()

        if pd.isna(invited) or invited > snapshot:
            return "Not yet invited"
        if pd.notna(submitted) and submitted <= snapshot:
            return "Submitted"
        if outcome == "decline" and pd.notna(resolved) and resolved <= snapshot:
            return "Declined"

        terminal = noresp_terminal if pd.notna(noresp_terminal) else resolved
        if outcome == "no_response" and pd.notna(terminal) and terminal <= snapshot:
            return "No response"

        if pd.notna(accepted) and accepted <= snapshot:
            if pd.notna(due) and due < snapshot:
                return "Overdue"
            return "In review"

        return "Invitation pending"

    out["AssignmentStatusAsOf"] = out.apply(classify, axis=1)
    out["DaysOverdueAsOf"] = 0
    mask = out["AssignmentStatusAsOf"].eq("Overdue") & out["DateReviewDue"].notna()
    out.loc[mask, "DaysOverdueAsOf"] = (
        snapshot - out.loc[mask, "DateReviewDue"]
    ).dt.days.clip(lower=0)

    out["DaysToDueAsOf"] = np.nan
    due_mask = out["DateReviewDue"].notna()
    out.loc[due_mask, "DaysToDueAsOf"] = (
        out.loc[due_mask, "DateReviewDue"] - snapshot
    ).dt.days
    out["DueSoonAsOfFlag"] = (
        out["AssignmentStatusAsOf"].eq("In review")
        & out["DaysToDueAsOf"].between(0, 5, inclusive="both")
    )
    return out


# ================================================================
# RECOMMENDATION ENGINE
# ================================================================

def aggregate_assignment_features(
    papers: pd.DataFrame,
    assignments: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    pstate = build_paper_state(papers, snapshot)
    astate = assignment_status_as_of(assignments, snapshot)
    astate = astate[
        astate["DateReviewerInvited"].notna()
        & (astate["DateReviewerInvited"] <= snapshot)
    ].copy()

    if not astate.empty:
        astate["ReviewerWorkloadAtInvite"] = pd.to_numeric(
            astate.get("ReviewerWorkloadAtInvite", 0), errors="coerce"
        ).fillna(0)
        astate["NumRemindersSent"] = pd.to_numeric(
            astate.get("NumRemindersSent", 0), errors="coerce"
        ).fillna(0)
        astate["ReviewerDisagreementScore"] = pd.to_numeric(
            astate.get("ReviewerDisagreementScore", 0), errors="coerce"
        ).fillna(0)

        def count_status(name):
            return lambda s: int((s == name).sum())

        grouped = (
            astate.groupby("PaperID")
            .agg(
                InvitesSentAsOf=("AssignmentID", "count"),
                SubmittedReviewsAsOf=("AssignmentStatusAsOf", count_status("Submitted")),
                OverdueReviewsAsOf=("AssignmentStatusAsOf", count_status("Overdue")),
                NoResponseAsOf=("AssignmentStatusAsOf", count_status("No response")),
                DeclinedAsOf=("AssignmentStatusAsOf", count_status("Declined")),
                PendingInvitesAsOf=("AssignmentStatusAsOf", count_status("Invitation pending")),
                ActiveReviewsAsOf=("AssignmentStatusAsOf", count_status("In review")),
                DueSoonReviewsAsOf=("DueSoonAsOfFlag", "sum"),
                MaxDaysOverdueAsOf=("DaysOverdueAsOf", "max"),
                AvgReviewerWorkloadAtInvite=("ReviewerWorkloadAtInvite", "mean"),
                MaxReviewerWorkloadAtInvite=("ReviewerWorkloadAtInvite", "max"),
                AvgRemindersSent=("NumRemindersSent", "mean"),
                MaxDisagreementAsOf=("ReviewerDisagreementScore", "max"),
            )
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(columns=["PaperID"])

    feat = pstate.merge(grouped, on="PaperID", how="left")
    fill_cols = [
        "InvitesSentAsOf",
        "SubmittedReviewsAsOf",
        "OverdueReviewsAsOf",
        "NoResponseAsOf",
        "DeclinedAsOf",
        "PendingInvitesAsOf",
        "ActiveReviewsAsOf",
        "DueSoonReviewsAsOf",
        "MaxDaysOverdueAsOf",
        "AvgReviewerWorkloadAtInvite",
        "MaxReviewerWorkloadAtInvite",
        "AvgRemindersSent",
        "MaxDisagreementAsOf",
    ]
    for col in fill_cols:
        if col not in feat.columns:
            feat[col] = 0
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0)

    feat["TargetReviewers"] = pd.to_numeric(
        feat.get("TargetNumberOfReviewers", 3), errors="coerce"
    ).fillna(3)
    feat["MissingSubmittedReviews"] = (
        feat["TargetReviewers"] - feat["SubmittedReviewsAsOf"]
    ).clip(lower=0)
    return feat


def apply_rule_engine(feat: pd.DataFrame) -> pd.DataFrame:
    def recommend(row):
        stage = str(row.get("StageAsOf", ""))
        days_stage = float(row.get("DaysInCurrentStage", 0) or 0)
        overdue = int(row.get("OverdueReviewsAsOf", 0) or 0)
        due_soon = int(row.get("DueSoonReviewsAsOf", 0) or 0)
        noresp = int(row.get("NoResponseAsOf", 0) or 0)
        declined = int(row.get("DeclinedAsOf", 0) or 0)
        pending = int(row.get("PendingInvitesAsOf", 0) or 0)
        missing = int(row.get("MissingSubmittedReviews", 0) or 0)
        score_range = float(row.get("ReviewScoreRange", 0) or 0)
        score_std = float(row.get("ReviewScoreStd", 0) or 0)
        avg_workload = float(row.get("AvgReviewerWorkloadAtInvite", 0) or 0)
        max_workload = float(row.get("MaxReviewerWorkloadAtInvite", 0) or 0)

        action = "No action needed"
        outcome = "Continue normal workflow"
        score = 10
        reasons = []

        if stage == "Resolved: decision sent":
            return pd.Series(
                {
                    "RuleRecommendedAction": "No action — resolved",
                    "RecommendedOutcome": "Workflow complete",
                    "RulePriorityScore": 0,
                    "RulePriorityBand": "Resolved",
                    "RuleExplanation": "Paper already has a decision letter.",
                }
            )

        if stage == "Waiting: reviewer assignment":
            score += min(days_stage * 2, 35)
            action = "Invite / assign additional reviewers"
            outcome = "Reach target reviewer panel and begin review"
            reasons += [
                f"{missing} required review slot(s) remain",
                f"{int(days_stage)} days since submission / assignment start",
            ]

        elif stage in {"In review: waiting for first review", "In review: partial reviews received"}:
            score += min(days_stage * 1.5, 35)

            if noresp > 0 and missing > 0:
                action = "Close no-response invitation and invite replacement reviewer"
                outcome = "Restore reviewer panel without additional delay"
                score += 45 + 8 * noresp
                reasons += [
                    f"{noresp} invitation(s) ended in no response",
                    f"{missing} required review(s) remain",
                ]

            elif declined > 0 and missing > 0:
                action = "Invite replacement reviewer"
                outcome = "Replace declined reviewer and restore panel size"
                score += 38 + 6 * declined
                reasons += [
                    f"{declined} reviewer(s) declined",
                    f"{missing} required review(s) remain",
                ]

            elif overdue > 0:
                max_overdue = float(row.get("MaxDaysOverdueAsOf", 0) or 0)
                if max_overdue > 14:
                    action = "Final reminder + prepare replacement reviewer"
                    outcome = "Either obtain the review quickly or reassign it"
                    score += 55
                elif max_overdue > 7:
                    action = "Send firm overdue reminder and CC EIC"
                    outcome = "Reviewer confirms availability or submits promptly"
                    score += 45
                else:
                    action = "Send reviewer reminder"
                    outcome = "Outstanding review is submitted"
                    score += 35
                reasons.append(f"{overdue} active review assignment(s) are overdue")
                if max_overdue > 0:
                    reasons.append(f"longest overdue review is {int(max_overdue)} days late")

            elif due_soon > 0:
                action = "Send friendly upcoming-deadline reminder"
                outcome = "Reviewer submits before due date"
                score += 18
                reasons.append(f"{due_soon} active review(s) are due within 5 days")

            elif max_workload >= 10:
                action = "Avoid overloaded reviewer / assign alternative if needed"
                outcome = "Reduce delay risk from reviewer workload"
                score += 25
                reasons.append(f"reviewer workload is high (max={max_workload:.0f})")

            elif days_stage >= 35 and missing > 0:
                action = "Review outstanding assignments and consider replacement"
                outcome = "Recover delayed reviews and close review phase"
                score += 35
                reasons += [
                    f"paper has spent {int(days_stage)} days in review",
                    f"{missing} required review(s) remain",
                ]

            elif pending > 0:
                action = "Follow up on pending invitations"
                outcome = "Confirm reviewer participation or replace quickly"
                score += 18
                reasons.append(f"{pending} reviewer invitation(s) remain pending")

            else:
                action = "Continue monitoring active reviews"
                outcome = "All required reviews arrive on schedule"
                reasons.append("review process is active with no strong delay signal")

        elif stage == "Waiting: AE recommendation":
            score += min(days_stage * 2, 40)
            if score_range >= 2.0 or score_std >= 1.0:
                action = "Invite adjudicating reviewer / perform AE synthesis"
                outcome = "Resolve reviewer disagreement before EIC decision"
                score += 35
                reasons.append(
                    f"reviewer disagreement is high (range={score_range:.2f}, std={score_std:.2f})"
                )
            elif days_stage >= 10:
                action = "Prioritize AE recommendation"
                outcome = "Move complete review package to EIC"
                score += 35
                reasons.append(f"AE recommendation has been pending {int(days_stage)} days")
            else:
                action = "Prepare AE recommendation"
                outcome = "Move manuscript to EIC decision stage"
                reasons.append("all required reviews are available")

        elif stage == "Waiting: EIC decision":
            action = "Prioritize for EIC decision"
            outcome = "Record editorial decision and move to decision letter"
            score += min(days_stage * 2, 40)
            reasons.append(f"EIC decision has been pending {int(days_stage)} days")
            if days_stage >= 7:
                score += 35

        elif stage == "Waiting: decision letter":
            action = "Finalize / send decision letter"
            outcome = "Author receives editorial outcome"
            score += 45
            reasons.append(f"decision letter has been pending {int(days_stage)} days")

        if score_range >= 2.0 and "disagreement" not in " ".join(reasons).lower():
            score += 15
            reasons.append(f"reviewer score range is high ({score_range:.2f})")

        if avg_workload >= 8 and "workload" not in " ".join(reasons).lower():
            score += 8
            reasons.append(f"average reviewer workload is elevated ({avg_workload:.1f})")

        score = int(np.clip(score, 0, 100))
        band = "High" if score >= 70 else "Medium" if score >= 40 else "Low"

        return pd.Series(
            {
                "RuleRecommendedAction": action,
                "RecommendedOutcome": outcome,
                "RulePriorityScore": score,
                "RulePriorityBand": band,
                "RuleExplanation": "; ".join(reasons),
            }
        )

    rules = feat.apply(recommend, axis=1)
    return pd.concat([feat.reset_index(drop=True), rules.reset_index(drop=True)], axis=1)

def train_original_delay_model(workflow_paper: pd.DataFrame, workflow_rev: pd.DataFrame):
    if not SKLEARN_OK:
        return None, {}, [], []

    paper = to_datetime_columns(workflow_paper, DATE_COLS_PAPER)
    rev = to_datetime_columns(workflow_rev, DATE_COLS_ASSIGNMENT)

    if "SubmissionRound" not in paper.columns:
        paper["SubmissionRound"] = 1
    if "SubmissionRound" not in rev.columns:
        rev["SubmissionRound"] = 1

    invite_outcome = rev["InviteOutcome"].astype(str).str.lower() if "InviteOutcome" in rev.columns else pd.Series("", index=rev.index)
    rev["AcceptedFlag"] = invite_outcome.eq("accept").astype(int)
    rev["DeclinedFlag"] = invite_outcome.eq("decline").astype(int)
    rev["NoResponseFlag"] = invite_outcome.eq("no_response").astype(int)
    rev["SubmittedFlag"] = rev["DateReviewSubmitted"].notna().astype(int) if "DateReviewSubmitted" in rev.columns else 0

    if "LateSubmissionFlag" in rev.columns:
        rev["LateFlag"] = rev["LateSubmissionFlag"].astype(str).str.lower().eq("yes").astype(int)
    elif "DateReviewSubmitted" in rev.columns and "DateReviewDue" in rev.columns:
        rev["LateFlag"] = (
            rev["DateReviewSubmitted"].notna()
            & rev["DateReviewDue"].notna()
            & (rev["DateReviewSubmitted"] > rev["DateReviewDue"])
        ).astype(int)
    else:
        rev["LateFlag"] = 0

    for col in ["ReviewerWorkloadAtInvite", "NumRemindersSent", "ReviewerDisagreementScore"]:
        if col not in rev.columns:
            rev[col] = 0
        rev[col] = pd.to_numeric(rev[col], errors="coerce").fillna(0)

    grouped = (
        rev.groupby(["PaperID", "SubmissionRound"])
        .agg(
            InvitesSent=("ReviewerID", "count"),
            Accepted=("AcceptedFlag", "sum"),
            Declined=("DeclinedFlag", "sum"),
            NoResponse=("NoResponseFlag", "sum"),
            SubmittedReviews=("SubmittedFlag", "sum"),
            LateReviews=("LateFlag", "sum"),
            AvgReviewerWorkload=("ReviewerWorkloadAtInvite", "mean"),
            AvgReminders=("NumRemindersSent", "mean"),
            MaxDisagreement=("ReviewerDisagreementScore", "max"),
        )
        .reset_index()
    )

    data = paper.merge(grouped, on=["PaperID", "SubmissionRound"], how="left")
    if "TotalTime_SubmissionToDecision_Days" in data.columns:
        data["FinalTotalDays"] = pd.to_numeric(data["TotalTime_SubmissionToDecision_Days"], errors="coerce")
    elif "DateDecisionLetterSent" in data.columns:
        data["FinalTotalDays"] = (data["DateDecisionLetterSent"] - data["DatePaperSubmitted"]).dt.days
    else:
        return None, {}, [], []

    data = data.dropna(subset=["FinalTotalDays"]).copy()
    if len(data) < 30:
        return None, {}, [], []

    cutoff = float(data["FinalTotalDays"].quantile(0.75))
    data["DelayedTarget"] = (data["FinalTotalDays"] >= cutoff).astype(int)
    if data["DelayedTarget"].nunique() < 2:
        return None, {}, [], []

    numeric = [
        "SubmissionRound", "TargetNumberOfReviewers", "InvitesSent", "Accepted",
        "Declined", "NoResponse", "SubmittedReviews", "LateReviews",
        "AvgReviewerWorkload", "AvgReminders", "MaxDisagreement",
    ]
    categorical = [
        "JournalSection", "PaperStatusOnSubmission",
        "HandlingAssociateEditorID", "HandlingEIC_ID",
    ]
    numeric = [c for c in numeric if c in data.columns]
    categorical = [c for c in categorical if c in data.columns]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=7,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    X = data[numeric + categorical]
    y = data["DelayedTarget"]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "cutoff_days": cutoff,
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else np.nan,
        "train_rows": int(len(data)),
    }
    return model, metrics, numeric, categorical


def build_recommendations(
    papers: pd.DataFrame,
    assignments: pd.DataFrame,
    snapshot: pd.Timestamp,
    workflow_paper: pd.DataFrame,
    workflow_rev: pd.DataFrame,
):
    feat = aggregate_assignment_features(papers, assignments, snapshot)
    recs = apply_rule_engine(feat)

    model, metrics, numeric, categorical = train_original_delay_model(workflow_paper, workflow_rev)

    model_input = recs.copy()
    mapping = {
        "InvitesSent": "InvitesSentAsOf",
        "Accepted": "SubmittedReviewsAsOf",
        "Declined": "DeclinedAsOf",
        "NoResponse": "NoResponseAsOf",
        "SubmittedReviews": "SubmittedReviewsAsOf",
        "LateReviews": "OverdueReviewsAsOf",
        "AvgReviewerWorkload": "AvgReviewerWorkloadAtInvite",
        "AvgReminders": "AvgRemindersSent",
        "MaxDisagreement": "ReviewScoreStd",
    }
    for target, source in mapping.items():
        if target not in model_input.columns:
            model_input[target] = model_input[source] if source in model_input.columns else 0

    if model is not None:
        for col in numeric:
            if col not in model_input.columns:
                model_input[col] = 0
        for col in categorical:
            if col not in model_input.columns:
                model_input[col] = "Unknown"
        recs["MLDelayRisk"] = model.predict_proba(model_input[numeric + categorical])[:, 1]
    else:
        recs["MLDelayRisk"] = recs["RulePriorityScore"] / 100.0

    recs["CombinedPriorityScore"] = (
        0.55 * recs["MLDelayRisk"] * 100
        + 0.45 * recs["RulePriorityScore"]
    ).round(1)
    recs["CombinedPriorityBand"] = pd.cut(
        recs["CombinedPriorityScore"],
        bins=[-1, 39.99, 69.99, 100.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return recs, metrics


# ================================================================
# REVIEWER SUGGESTION + MOCK COMMUNICATIONS
# ================================================================

def recommend_reviewer(
    paper_id: int,
    reviewers: pd.DataFrame,
    assignments: pd.DataFrame,
    snapshot: pd.Timestamp,
    prefer_balanced: bool = False,
) -> Optional[pd.Series]:
    already_assigned = set(
        assignments.loc[assignments["PaperID"] == paper_id, "ReviewerID"].astype(str)
    )
    state = assignment_status_as_of(assignments, snapshot)
    active = state[state["AssignmentStatusAsOf"].isin(["In review", "Overdue", "Invitation pending"])]
    workload = active.groupby("ReviewerID").size().to_dict()

    candidates = reviewers[~reviewers["ReviewerID"].astype(str).isin(already_assigned)].copy()
    if candidates.empty:
        return None

    candidates["ActiveWorkload"] = candidates["ReviewerID"].astype(str).map(workload).fillna(0)
    if prefer_balanced and "ReviewerBand" in candidates.columns:
        candidates["BandPenalty"] = np.where(
            candidates["ReviewerBand"].astype(str).str.lower().str.contains("balanced"), 0, 1
        )
    else:
        candidates["BandPenalty"] = 0

    candidates = candidates.sort_values(["BandPenalty", "ActiveWorkload", "ReviewerID"])
    return candidates.iloc[0]


TONE_OPENERS = {
    "Professional": "Dear Reviewer {reviewer},",
    "Warm": "Dear Reviewer {reviewer},\n\nI hope you are doing well.",
    "Concise": "Hello Reviewer {reviewer},",
    "Firm": "Dear Reviewer {reviewer},",
    "Final reminder": "Dear Reviewer {reviewer},",
}
TONE_CLOSERS = {
    "Professional": "Best regards,\n{ap}",
    "Warm": "Thank you again for your time and contribution.\n\nWarm regards,\n{ap}",
    "Concise": "Regards,\n{ap}",
    "Firm": "Thank you for your prompt attention to this matter.\n\nRegards,\n{ap}",
    "Final reminder": "Please reply as soon as possible so that we can keep the manuscript moving.\n\nRegards,\n{ap}",
}


def communication_scenario(assignment: pd.Series, snapshot: pd.Timestamp) -> str:
    status = str(assignment.get("AssignmentStatusAsOf", ""))
    if status == "No response":
        return "No response — replacement needed"
    if status == "Declined":
        return "Reviewer declined — replacement needed"
    if status == "Overdue":
        days = int(assignment.get("DaysOverdueAsOf", 0) or 0)
        if days > 14:
            return "Final overdue reminder + replacement"
        if days > 7:
            return "Firm overdue reminder"
        return "Overdue reminder"
    if status == "Invitation pending":
        return "Invitation follow-up"
    if status == "In review":
        due = assignment.get("DateReviewDue", pd.NaT)
        if pd.notna(due) and 0 <= (due - snapshot).days <= 5:
            return "Upcoming deadline reminder"
        return "Review in progress"
    if status == "Submitted":
        return "Thank reviewer"
    return "Status follow-up"


def _tone_sentence(tone: str, neutral: str, warm: str, firm: str, concise: str) -> str:
    if tone == "Warm":
        return warm
    if tone in {"Firm", "Final reminder"}:
        return firm
    if tone == "Concise":
        return concise
    return neutral


def generate_mock_message(
    paper: pd.Series,
    assignment: pd.Series,
    tone: str,
    snapshot: pd.Timestamp,
    suggested_reviewer: Optional[pd.Series] = None,
    custom_instruction: str = "",
) -> Tuple[str, str, str]:
    """Create a complete editable reviewer message from the live journal state."""
    reviewer = str(assignment.get("ReviewerID", "Reviewer"))
    ae = str(paper.get("HandlingAssociateEditorID", "AE"))
    eic = str(paper.get("HandlingEIC_ID", "EIC"))
    title = str(paper.get("PaperTitle", "the manuscript"))
    pid = int(paper["PaperID"])
    scenario = communication_scenario(assignment, snapshot)

    opener = TONE_OPENERS[tone].format(reviewer=reviewer)
    closer = TONE_CLOSERS[tone].format(ap=ae)
    due = assignment.get("DateReviewDue", pd.NaT)
    due_text = pd.Timestamp(due).strftime("%d %b %Y") if pd.notna(due) else "the requested deadline"
    days_overdue = int(assignment.get("DaysOverdueAsOf", 0) or 0)

    if scenario == "Invitation follow-up":
        subject = f"Follow-up: review invitation for Manuscript {pid}"
        core = _tone_sentence(
            tone,
            f"We are following up on our invitation to review “{title}”. Please let us know whether you are able to accept this review.",
            f"I wanted to follow up on our invitation to review “{title}”. We would be grateful if you could let us know whether the assignment fits your availability.",
            f"Please confirm whether you can accept the review invitation for “{title}”. If we do not hear back shortly, we may need to invite another reviewer to avoid delay.",
            f"Please confirm whether you can review “{title}”.",
        )

    elif scenario == "Upcoming deadline reminder":
        subject = f"Reminder: upcoming review deadline for Manuscript {pid}"
        core = _tone_sentence(
            tone,
            f"Thank you for reviewing “{title}”. This is a reminder that the requested review date is {due_text}.",
            f"Thank you again for taking on the review of “{title}”. Just a friendly reminder that the requested review date is {due_text}.",
            f"The requested review date for “{title}” is {due_text}. Please submit by that date or let us know immediately if there is a problem.",
            f"Reminder: the review of “{title}” is due {due_text}.",
        )

    elif scenario == "Overdue reminder":
        subject = f"Reminder: review overdue for Manuscript {pid}"
        core = _tone_sentence(
            tone,
            f"Thank you for agreeing to review “{title}”. The review is currently {days_overdue} day(s) past the requested date of {due_text}. We would appreciate submission at your earliest convenience.",
            f"Thank you for your time on “{title}”. The review is now {days_overdue} day(s) overdue. We would greatly appreciate receiving it when possible, and please let us know if you need a brief extension.",
            f"The review of “{title}” is {days_overdue} day(s) overdue. Please submit it promptly or confirm today if you cannot complete the assignment.",
            f"The review of “{title}” is {days_overdue} day(s) overdue. Please submit or confirm availability.",
        )

    elif scenario == "Firm overdue reminder":
        subject = f"Action requested: overdue review for Manuscript {pid}"
        core = _tone_sentence(
            tone,
            f"The review for “{title}” is now {days_overdue} day(s) overdue. Please submit the review promptly, or let us know if you are no longer able to complete it so that we can make alternative arrangements.",
            f"We appreciate your help with “{title}”. The review is now {days_overdue} day(s) overdue, so please let us know whether you can complete it shortly; otherwise we will arrange an alternative reviewer.",
            f"The review for “{title}” is {days_overdue} day(s) overdue. Please submit it promptly or confirm that we should reassign the manuscript.",
            f"Review overdue by {days_overdue} day(s). Please submit or confirm reassignment.",
        )

    elif scenario == "Final overdue reminder + replacement":
        replacement = (
            f" We are preparing to invite {suggested_reviewer['ReviewerID']} as a replacement reviewer."
            if suggested_reviewer is not None
            else " We are preparing to assign a replacement reviewer."
        )
        subject = f"Final reminder: review for Manuscript {pid}"
        core = (
            f"The review for “{title}” is now {days_overdue} day(s) overdue. "
            "If you can still complete the review, please confirm immediately. Otherwise, "
            f"we will reassign the review to prevent further delay.{replacement}"
        )

    elif scenario == "No response — replacement needed":
        replacement = (
            f" We plan to invite {suggested_reviewer['ReviewerID']} as a replacement reviewer."
            if suggested_reviewer is not None
            else " We will now invite a replacement reviewer."
        )
        subject = f"Review invitation closed: Manuscript {pid}"
        core = (
            f"We have not received a response to the review invitation for “{title}”. "
            f"To keep the editorial process moving, we are closing this invitation.{replacement}"
        )

    elif scenario == "Reviewer declined — replacement needed":
        subject = f"Thank you for responding: Manuscript {pid}"
        core = (
            f"Thank you for letting us know that you are unable to review “{title}”. "
            "We appreciate the prompt response and will arrange another reviewer."
        )

    elif scenario == "Thank reviewer":
        subject = f"Thank you for your review: Manuscript {pid}"
        core = (
            f"Thank you for submitting your review of “{title}”. "
            "Your assessment has been added to the editorial record and will inform the next decision stage."
        )

    else:
        subject = f"Review status: Manuscript {pid}"
        core = (
            f"We are checking in regarding your review of “{title}”. "
            "Please let us know if you need clarification about the requested timeline."
        )

    if custom_instruction.strip():
        core += f"\n\nAdditional note: {custom_instruction.strip()}"

    body = (
        f"{opener}\n\n"
        f"{core}\n\n"
        f"Manuscript ID: {pid}\n"
        f"Manuscript title: {title}\n\n"
        f"CC: {eic}\n\n"
        f"{closer}"
    )
    return scenario, subject, body


def generate_reviewer_invitation(
    paper: pd.Series,
    reviewer: pd.Series,
    tone: str,
    purpose: str = "Replacement review",
    custom_instruction: str = "",
) -> Tuple[str, str]:
    rid = str(reviewer["ReviewerID"])
    ae = str(paper["HandlingAssociateEditorID"])
    eic = str(paper["HandlingEIC_ID"])
    pid = int(paper["PaperID"])
    title = str(paper["PaperTitle"])
    persona = str(reviewer.get("ReviewerPersona", ""))
    focus = str(reviewer.get("ReviewerFocus", ""))

    opener = TONE_OPENERS[tone].format(reviewer=rid)
    closer = TONE_CLOSERS[tone].format(ap=ae)

    if purpose == "Adjudicating review":
        subject = f"Review invitation: adjudication review for Manuscript {pid}"
        core = (
            f"We would like to invite you to provide an additional independent review of “{title}”. "
            "The existing reports show meaningful disagreement, and your assessment would help the editorial team resolve the remaining uncertainty."
        )
    else:
        subject = f"Review invitation: Manuscript {pid}"
        core = (
            f"We would like to invite you to review “{title}”. A previous reviewer is unavailable or delayed, "
            "and your assessment would help us keep the manuscript on schedule."
        )

    if persona or focus:
        core += f"\n\nReviewer profile: {persona}. Focus: {focus}."
    if custom_instruction.strip():
        core += f"\n\nAdditional note: {custom_instruction.strip()}"

    body = (
        f"{opener}\n\n{core}\n\n"
        f"Manuscript ID: {pid}\n"
        f"Manuscript title: {title}\n\n"
        f"CC: {eic}\n\n{closer}"
    )
    return subject, body


def generate_tiebreak_invitation(paper: pd.Series, reviewer: pd.Series, tone: str) -> Tuple[str, str]:
    return generate_reviewer_invitation(
        paper,
        reviewer,
        tone,
        purpose="Adjudicating review",
    )

def load_source_comparison():
    names = {
        "reviews": "SourceComparisonReviews.csv",
        "pairs": "SourceComparisonPairs.csv",
        "summary": "SourceComparisonPaperSummary.csv",
        "persona": "PersonaSourceSensitivity.csv",
    }
    found = {}
    for key, name in names.items():
        for directory in SOURCE_COMPARISON_DIRS:
            path = directory / name
            if path.exists():
                found[key] = path
                break
    if len(found) != len(names):
        return None
    return {key: pd.read_csv(path) for key, path in found.items()}


# ================================================================
# CLEAN PAPER TIMELINE VISUALIZATION
# ================================================================

def render_clean_paper_timeline(paper: pd.Series, paper_assignments: pd.DataFrame, snapshot: pd.Timestamp):
    st.markdown("#### Journal timeline")

    # Editorial phase bars: a compact single-lane Gantt rather than clustered labels.
    phase_specs = [
        ("Reviewer assignment", "DatePaperSubmitted", "DateReviewersFullyAssigned"),
        ("Peer review", "DateReviewersFullyAssigned", "DateAllReviewsReceived"),
        ("AE assessment", "DateAllReviewsReceived", "AE_RecommendationDate"),
        ("EIC decision", "AE_RecommendationDate", "EIC_DecisionDate"),
        ("Decision letter", "EIC_DecisionDate", "DateDecisionLetterSent"),
    ]

    phase_rows = []
    for phase, start_col, end_col in phase_specs:
        start = paper.get(start_col, pd.NaT)
        end = paper.get(end_col, pd.NaT)
        if pd.notna(start) and pd.notna(end) and end >= start:
            phase_rows.append({"Phase": phase, "Start": start, "Finish": end, "Lane": "Editorial workflow"})

    if phase_rows:
        phase_df = pd.DataFrame(phase_rows)
        fig = px.timeline(
            phase_df,
            x_start="Start",
            x_end="Finish",
            y="Lane",
            color="Phase",
            hover_data=["Phase", "Start", "Finish"],
            title="Editorial workflow phases",
        )
        snapshot_marker = (
            snapshot.to_pydatetime()
            if hasattr(snapshot, "to_pydatetime")
            else snapshot
        )

        fig.add_vline(
            x=snapshot_marker,
            line_dash="dash",
            line_width=2,
        )

        fig.add_annotation(
            x=snapshot_marker,
            y=1.02,
            xref="x",
            yref="paper",
            text="As of",
            showarrow=False,
        )
        fig.update_yaxes(title="", showticklabels=False)
        fig.update_layout(height=260, bargap=0.45, legend_title_text="Phase")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Reviewer swimlanes")

    lane_rows = []
    marker_rows = []
    pa = to_datetime_columns(paper_assignments.copy(), DATE_COLS_ASSIGNMENT)

    for _, a in pa.iterrows():
        rid = str(a.get("ReviewerID", ""))
        invited = a.get("DateReviewerInvited", pd.NaT)
        accepted = a.get("DateInvitationAccepted", pd.NaT)
        resolved = a.get("DateInvitationResolved", pd.NaT)
        due = a.get("DateReviewDue", pd.NaT)
        submitted = a.get("DateReviewSubmitted", pd.NaT)
        terminal = a.get("DateNoResponseTerminal", pd.NaT)
        outcome = str(a.get("InviteOutcome", "")).lower().strip()

        if pd.notna(invited):
            if outcome == "decline" and pd.notna(resolved):
                lane_rows.append({"Reviewer": rid, "Stage": "Invitation → decline", "Start": invited, "Finish": resolved})
            elif outcome == "no_response":
                end = terminal if pd.notna(terminal) else resolved
                if pd.notna(end):
                    lane_rows.append({"Reviewer": rid, "Stage": "Invitation → no response", "Start": invited, "Finish": end})
            else:
                invite_end = accepted if pd.notna(accepted) else min(snapshot, invited + pd.Timedelta(days=7))
                if pd.notna(invite_end) and invite_end > invited:
                    lane_rows.append({"Reviewer": rid, "Stage": "Invitation", "Start": invited, "Finish": invite_end})

        if pd.notna(accepted):
            review_end = submitted if pd.notna(submitted) else max(snapshot, due) if pd.notna(due) else snapshot
            ontime_end = min(review_end, due) if pd.notna(due) else review_end
            if pd.notna(ontime_end) and ontime_end > accepted:
                lane_rows.append({"Reviewer": rid, "Stage": "Review", "Start": accepted, "Finish": ontime_end})
            if pd.notna(due) and review_end > due:
                lane_rows.append({"Reviewer": rid, "Stage": "Overdue", "Start": due, "Finish": review_end})

        if pd.notna(due):
            marker_rows.append({"Reviewer": rid, "Date": due, "Marker": "Due"})
        if pd.notna(submitted):
            marker_rows.append({"Reviewer": rid, "Date": submitted, "Marker": "Submitted"})

    if lane_rows:
        lane_df = pd.DataFrame(lane_rows)
        fig = px.timeline(
            lane_df,
            x_start="Start",
            x_end="Finish",
            y="Reviewer",
            color="Stage",
            title="Reviewer invitation and review progress",
        )
        fig.update_yaxes(autorange="reversed")
        snapshot_marker = (
            snapshot.to_pydatetime()
            if hasattr(snapshot, "to_pydatetime")
            else snapshot
        )

        fig.add_shape(
            type="line",
            x0=snapshot_marker,
            x1=snapshot_marker,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                dash="dash",
                width=2,
            ),
        )

        fig.add_annotation(
            x=snapshot_marker,
            y=1.02,
            xref="x",
            yref="paper",
            text="As of",
            showarrow=False,
        )

        if marker_rows:
            md = pd.DataFrame(marker_rows)
            symbol_map = {"Due": "diamond-open", "Submitted": "circle"}
            for marker in md["Marker"].unique():
                sub = md[md["Marker"] == marker]
                fig.add_trace(
                    go.Scatter(
                        x=sub["Date"],
                        y=sub["Reviewer"],
                        mode="markers",
                        name=marker,
                        marker=dict(symbol=symbol_map.get(marker, "circle"), size=10),
                        hovertemplate=f"{marker}: %{{x}}<extra></extra>",
                    )
                )

        fig.update_layout(height=max(420, 110 + 48 * pa["ReviewerID"].nunique()), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reviewer timeline events are available for this manuscript.")

    # Compact milestones table remains useful for exact dates.
    milestones = []
    for label, col in [
        ("Submitted", "DatePaperSubmitted"),
        ("Reviewers fully assigned", "DateReviewersFullyAssigned"),
        ("First review received", "DateFirstReviewReceived"),
        ("All reviews received", "DateAllReviewsReceived"),
        ("AE recommendation", "AE_RecommendationDate"),
        ("EIC decision", "EIC_DecisionDate"),
        ("Decision letter", "DateDecisionLetterSent"),
    ]:
        dt = paper.get(col, pd.NaT)
        if pd.notna(dt):
            milestones.append({"Milestone": label, "Date": pd.Timestamp(dt).strftime("%d %b %Y")})
    if milestones:
        st.dataframe(pd.DataFrame(milestones), use_container_width=True, hide_index=True)

# ================================================================
# APP STARTUP
# ================================================================

st.set_page_config(
    page_title="Unified AI Journal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Unified AI Journal")
st.caption(
    "A coherent journal simulation built from the 103 AI Scientist papers, persona reviews, "
    "and the empirical workflow behavior from EIC-AIssist."
)

try:
    build_unified_journal(force=False)
except Exception as exc:
    st.error(f"Unable to build the unified journal: {exc}")
    st.stop()

st.sidebar.header("Journal simulation")

if st.sidebar.button("Rebuild unified journal", type="primary"):
    try:
        with st.spinner("Rebuilding unified journal from the source datasets..."):
            build_unified_journal(force=True)
            clear_unified_cache()
        st.sidebar.success("Unified journal rebuilt.")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(str(exc))

try:
    papers, editors, reviewers, assignments, reviews, decisions = load_unified_tables()
    workflow_paper, workflow_rev, workflow_source = load_workflow_sources()
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Snapshot range.
date_series = []
for col in DATE_COLS_PAPER:
    if col in papers.columns:
        date_series.append(pd.to_datetime(papers[col], errors="coerce").dropna())
all_dates = pd.concat(date_series) if date_series else pd.Series(dtype="datetime64[ns]")
min_snapshot = all_dates.min() if not all_dates.empty else pd.Timestamp.today() - pd.Timedelta(days=365)
max_snapshot = all_dates.max() if not all_dates.empty else pd.Timestamp.today()
default_snapshot = default_snapshot_date(papers)

snapshot_date = st.sidebar.date_input(
    "Simulation as-of date",
    value=default_snapshot.date(),
    min_value=min_snapshot.date(),
    max_value=max_snapshot.date(),
)
snapshot = pd.Timestamp(snapshot_date)

paper_state = build_paper_state(papers, snapshot)
assignment_state = assignment_status_as_of(assignments, snapshot)
recs, model_metrics = build_recommendations(
    papers,
    assignments,
    snapshot,
    workflow_paper,
    workflow_rev,
)

# Persist the current recommendation snapshot for external inspection.
try:
    recs.to_csv(RECOMMENDATIONS_FILE, index=False)
except Exception:
    pass

with st.sidebar.expander("Data provenance"):
    st.write(f"Workflow template: `{workflow_source}`")
    st.write(f"Unified data directory: `{UNIFIED_DIR}`")
    if BUILD_META_FILE.exists():
        try:
            st.json(json.loads(BUILD_META_FILE.read_text()))
        except Exception:
            pass

nav_options = [
    "Journal Overview",
    "Paper POV",
    "Reviewer POV",
    "AE POV",
    "EIC POV",
    "AI Scientist Reviews",
    "Recommendation Center",
    "Communications",
    "Scenario Lab",
    "Source Comparison",
    "Data Explorer",
]

selected_tab = st.segmented_control(
    "Navigation",
    nav_options,
    default="Journal Overview",
    key="unified_nav",
)


# ================================================================
# JOURNAL OVERVIEW
# ================================================================

if selected_tab == "Journal Overview":
    visible = paper_state[paper_state["StageAsOf"] != "Not yet submitted"].copy()
    active = visible[visible["ResolutionStateAsOf"] == "Active"].copy()
    resolved = visible[visible["ResolutionStateAsOf"] == "Resolved"].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AI Scientist papers", len(papers))
    c2.metric("Visible by snapshot", len(visible))
    c3.metric("Active manuscripts", len(active))
    c4.metric("Resolved", len(resolved))
    c5.metric("Persona reviews", len(reviews))

    left, right = st.columns(2)

    with left:
        stage_counts = (
            visible["StageAsOf"]
            .value_counts()
            .reindex(STAGE_ORDER, fill_value=0)
            .reset_index()
        )
        stage_counts.columns = ["Stage", "Papers"]
        fig = px.bar(
            stage_counts,
            x="Stage",
            y="Papers",
            text="Papers",
            title="Manuscripts by journal stage",
            category_orders={"Stage": STAGE_ORDER},
        )
        fig.update_xaxes(tickangle=35)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        priority_counts = (
            recs[recs["ResolutionStateAsOf"].eq("Active")]["CombinedPriorityBand"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .reset_index()
        )
        priority_counts.columns = ["Priority", "Papers"]
        fig = px.bar(
            priority_counts,
            x="Priority",
            y="Papers",
            text="Papers",
            title="Active manuscripts by intervention priority",
            category_orders={"Priority": ["High", "Medium", "Low"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    if "DatePaperSubmitted" in visible.columns:
        monthly = visible.dropna(subset=["DatePaperSubmitted"]).copy()
        monthly["SubmissionMonth"] = monthly["DatePaperSubmitted"].dt.to_period("M").dt.to_timestamp()
        monthly = monthly.groupby(["SubmissionMonth", "StageAsOf"]).size().reset_index(name="Papers")
        fig = px.bar(
            monthly,
            x="SubmissionMonth",
            y="Papers",
            color="StageAsOf",
            title="Submission volume and current stage",
            category_orders={"StageAsOf": STAGE_ORDER},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-priority manuscripts")
    top = recs[recs["ResolutionStateAsOf"].eq("Active")].sort_values("CombinedPriorityScore", ascending=False)
    cols = [
        "PaperID", "PaperTitle", "StageAsOf", "HandlingAssociateEditorID",
        "HandlingEIC_ID", "RuleRecommendedAction", "CombinedPriorityBand",
        "CombinedPriorityScore", "MLDelayRisk", "DaysSinceSubmission",
        "DaysInCurrentStage", "RuleExplanation",
    ]
    st.dataframe(top[[c for c in cols if c in top.columns]].head(20), use_container_width=True, hide_index=True)


# ================================================================
# PAPER POV
# ================================================================

if selected_tab == "Paper POV":
    st.subheader("Paper POV")

    paper_options = {
        f"{int(row.PaperID)} — {row.PaperTitle}": int(row.PaperID)
        for row in papers.sort_values("PaperID").itertuples()
    }
    selected_label = st.selectbox("Select manuscript", list(paper_options.keys()), key="paper_pov_select")
    pid = paper_options[selected_label]

    p = paper_state[paper_state["PaperID"] == pid].iloc[0]
    pr = recs[recs["PaperID"] == pid].iloc[0]
    pa = assignment_state[assignment_state["PaperID"] == pid].copy()
    rv = reviews[
        (reviews["PaperID"] == pid)
        & pd.to_datetime(reviews["DateReviewSubmitted"], errors="coerce").notna()
        & (pd.to_datetime(reviews["DateReviewSubmitted"], errors="coerce") <= snapshot)
    ].copy()

    st.markdown(f"### {p['PaperTitle']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Stage", p["StageAsOf"])
    m2.metric("Days since submission", int(p["DaysSinceSubmission"]) if pd.notna(p["DaysSinceSubmission"]) else 0)
    m3.metric("Reviews", len(rv))
    m4.metric("Mean score", f"{float(p.get('MeanReviewScore', 0)):.2f}")
    m5.metric("Score range", f"{float(p.get('ReviewScoreRange', 0)):.2f}")
    m6.metric("Priority", f"{float(pr['CombinedPriorityScore']):.1f}")

    st.info(
        f"**Recommended action:** {pr['RuleRecommendedAction']}  \n"
        f"**Reason:** {pr['RuleExplanation']}"
    )

    render_clean_paper_timeline(p, pa, snapshot)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Reviewer assignments")
        cols = [
            "AssignmentID", "ReviewerID", "AssignmentStatusAsOf",
            "DateReviewerInvited", "DateInvitationAccepted", "DateReviewDue",
            "DateReviewSubmitted", "DaysOverdueAsOf", "NumRemindersSent",
        ]
        st.dataframe(pa[[c for c in cols if c in pa.columns]].sort_values("ReviewerID"), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Reviewer scores")
        if not rv.empty:
            fig = px.bar(
                rv,
                x="ReviewerID",
                y="Score",
                color="Decision",
                text="Score",
                title="Submitted review assessments",
                category_orders={"Decision": DECISION_ORDER},
            )
            fig.update_yaxes(range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full review reports")
    for _, row in rv.sort_values("Score").iterrows():
        title = f"{row['ReviewerID']} — {row.get('ReviewerPersona', '')} — {float(row['Score']):.1f}/10 — {row['Decision']}"
        with st.expander(title):
            st.markdown("**Summary**")
            st.write(row.get("Summary", ""))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Strengths**")
                for x in parse_listish(row.get("Strengths", "")):
                    st.markdown(f"- {x}")
            with c2:
                st.markdown("**Weaknesses**")
                for x in parse_listish(row.get("Weaknesses", "")):
                    st.markdown(f"- {x}")
            st.markdown("**Relevance**")
            st.write(row.get("Relevance", ""))
            st.markdown("**Conclusion**")
            st.write(row.get("Conclusion", ""))


# ================================================================
# REVIEWER POV
# ================================================================

if selected_tab == "Reviewer POV":
    st.subheader("Reviewer POV")

    reviewer_options = sorted(reviewers["ReviewerID"].astype(str).unique())
    rid = st.selectbox("Select reviewer", reviewer_options, key="reviewer_pov_select")

    profile = reviewers[reviewers["ReviewerID"].astype(str) == rid].iloc[0]
    ra = assignment_state[assignment_state["ReviewerID"].astype(str) == rid].copy()
    rr = reviews[
        (reviews["ReviewerID"].astype(str) == rid)
        & pd.to_datetime(reviews["DateReviewSubmitted"], errors="coerce").notna()
        & (pd.to_datetime(reviews["DateReviewSubmitted"], errors="coerce") <= snapshot)
    ].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Persona", profile.get("ReviewerPersona", ""))
    c2.metric("Band", profile.get("ReviewerBand", ""))
    c3.metric("Assignments", len(ra))
    c4.metric("Submitted reviews", len(rr))
    active_n = int(ra["AssignmentStatusAsOf"].isin(["In review", "Overdue", "Invitation pending"]).sum())
    c5.metric("Active workload", active_n)

    st.caption(f"Focus: {profile.get('ReviewerFocus', 'Overall scientific quality')}")

    if not ra.empty:
        status_counts = ra["AssignmentStatusAsOf"].value_counts().reset_index()
        status_counts.columns = ["Status", "Assignments"]
        fig = px.bar(status_counts, x="Status", y="Assignments", text="Assignments", title="Assignment status")
        st.plotly_chart(fig, use_container_width=True)

    display = ra.merge(
        papers[["PaperID", "PaperTitle", "HandlingAssociateEditorID", "HandlingEIC_ID"]],
        on="PaperID",
        how="left",
    )
    cols = [
        "PaperID", "PaperTitle", "AssignmentStatusAsOf", "DateReviewerInvited",
        "DateReviewDue", "DateReviewSubmitted", "DaysOverdueAsOf",
        "HandlingAssociateEditorID", "HandlingEIC_ID",
    ]
    st.dataframe(display[[c for c in cols if c in display.columns]], use_container_width=True, hide_index=True)

    if not rr.empty:
        st.markdown("#### Review scoring history")
        fig = px.scatter(
            rr.sort_values("PaperID"),
            x="PaperID",
            y="Score",
            color="Decision",
            hover_name="PaperTitle",
            category_orders={"Decision": DECISION_ORDER},
        )
        fig.update_yaxes(range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# AE POV
# ================================================================

if selected_tab == "AE POV":
    st.subheader("AE POV")
    st.caption("See the AE's full manuscript portfolio, active workload, bottlenecks, and recommended interventions.")

    active_ae_ids = sorted(
        recs.loc[
            recs["ResolutionStateAsOf"].eq("Active"),
            "HandlingAssociateEditorID",
        ].dropna().astype(str).unique().tolist()
    )
    all_ae_ids = sorted(papers["HandlingAssociateEditorID"].dropna().astype(str).unique().tolist())
    ae_options = active_ae_ids if active_ae_ids else all_ae_ids

    if not ae_options:
        st.warning("No AE assignments were found in the unified journal.")
        st.stop()

    ae_id = st.selectbox("Select AE", ae_options, key="ae_pov_select")
    subset = recs[recs["HandlingAssociateEditorID"].astype(str) == ae_id].copy()
    active = subset[subset["ResolutionStateAsOf"] == "Active"].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Assigned manuscripts", len(subset))
    c2.metric("Active", len(active))
    c3.metric("High priority", int((active["CombinedPriorityBand"] == "High").sum()) if not active.empty else 0)
    c4.metric("Waiting on AE", int((active["StageAsOf"] == "Waiting: AE recommendation").sum()) if not active.empty else 0)
    c5.metric("Median priority", f"{active['CombinedPriorityScore'].median():.1f}" if not active.empty else "0")

    portfolio_for_chart = subset[subset["StageAsOf"] != "Not yet submitted"].copy()
    stage_counts = portfolio_for_chart["StageAsOf"].value_counts().reindex(STAGE_ORDER, fill_value=0).reset_index()
    stage_counts.columns = ["Stage", "Papers"]
    fig = px.bar(
        stage_counts,
        x="Stage",
        y="Papers",
        text="Papers",
        title="AE manuscript portfolio by stage",
        category_orders={"Stage": STAGE_ORDER},
    )
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    if active.empty:
        st.info("This AE has no active manuscript at the selected snapshot. Showing the full portfolio below.")
        queue = subset.copy()
    else:
        queue = active.sort_values("CombinedPriorityScore", ascending=False)

    st.markdown("#### AE intervention queue")
    cols = [
        "PaperID", "PaperTitle", "ShowcaseScenario", "StageAsOf",
        "CombinedPriorityBand", "CombinedPriorityScore", "MLDelayRisk",
        "RuleRecommendedAction", "RecommendedOutcome", "DaysInCurrentStage",
        "SubmittedReviewsAsOf", "OverdueReviewsAsOf", "NoResponseAsOf",
        "DeclinedAsOf", "PendingInvitesAsOf", "MissingSubmittedReviews",
        "ReviewScoreRange", "RuleExplanation",
    ]
    st.dataframe(
        queue[[c for c in cols if c in queue.columns]],
        use_container_width=True,
        hide_index=True,
        height=460,
    )


# ================================================================
# EIC POV
# ================================================================

if selected_tab == "EIC POV":
    st.subheader("EIC POV")

    eic_options = sorted(papers["HandlingEIC_ID"].dropna().astype(str).unique())
    eic_id = st.selectbox("Select EIC", eic_options, key="eic_pov_select")
    subset = recs[recs["HandlingEIC_ID"].astype(str) == eic_id].copy()
    active = subset[subset["ResolutionStateAsOf"] == "Active"].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Manuscripts", len(subset))
    c2.metric("Active", len(active))
    c3.metric("High priority", int((active["CombinedPriorityBand"] == "High").sum()))
    c4.metric("High disagreement", int((pd.to_numeric(active.get("ReviewScoreRange", 0), errors="coerce") >= 2.0).sum()))

    if not active.empty:
        fig = px.scatter(
            active,
            x="DaysSinceSubmission",
            y="CombinedPriorityScore",
            color="StageAsOf",
            size="ReviewScoreRange",
            hover_name="PaperTitle",
            hover_data=["PaperID", "RuleRecommendedAction", "ReviewScoreRange", "MeanReviewScore"],
            title="EIC portfolio: age, risk and reviewer disagreement",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### EIC attention queue")
    cols = [
        "PaperID", "PaperTitle", "HandlingAssociateEditorID", "StageAsOf",
        "ConsensusDecision", "MeanReviewScore", "ReviewScoreRange",
        "CombinedPriorityScore", "RuleRecommendedAction", "RuleExplanation",
    ]
    st.dataframe(
        active[[c for c in cols if c in active.columns]].sort_values("CombinedPriorityScore", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ================================================================
# AI SCIENTIST REVIEWS
# ================================================================

if selected_tab == "AI Scientist Reviews":
    st.subheader("AI Scientist Reviews")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", reviews["PaperID"].nunique())
    c2.metric("Reviews", len(reviews))
    c3.metric("Reviewers", reviews["ReviewerID"].nunique())
    c4.metric("Mean score", f"{pd.to_numeric(reviews['Score'], errors='coerce').mean():.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(reviews, x="Score", nbins=18, title="Review score distribution")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = (
            reviews["Decision"]
            .value_counts()
            .reindex(DECISION_ORDER, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Decision", "Reviews"]
        fig = px.bar(counts, x="Decision", y="Reviews", text="Reviews", title="Review decision distribution")
        st.plotly_chart(fig, use_container_width=True)

    paper_review_stats = (
        reviews.groupby(["PaperID", "PaperTitle"])
        .agg(
            MeanScore=("Score", "mean"),
            ScoreStd=("Score", lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)),
            MinScore=("Score", "min"),
            MaxScore=("Score", "max"),
            ReviewerCount=("ReviewerID", "count"),
        )
        .reset_index()
    )
    paper_review_stats["ScoreRange"] = paper_review_stats["MaxScore"] - paper_review_stats["MinScore"]

    fig = px.scatter(
        paper_review_stats,
        x="MeanScore",
        y="ScoreRange",
        size="ReviewerCount",
        hover_name="PaperTitle",
        hover_data=["PaperID", "ScoreStd"],
        title="Paper quality vs reviewer disagreement",
    )
    fig.add_hline(y=2.0, line_dash="dash", annotation_text="High disagreement")
    st.plotly_chart(fig, use_container_width=True)

    if "ReviewerPersona" in reviews.columns:
        fig = px.box(
            reviews,
            x="ReviewerPersona",
            y="Score",
            points="outliers",
            title="Score distribution by reviewer persona",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Review explorer")
    review_paper_options = {
        f"{int(row.PaperID)} — {row.PaperTitle}": int(row.PaperID)
        for row in papers.sort_values("PaperID").itertuples()
    }
    label = st.selectbox("Paper", list(review_paper_options.keys()), key="reviews_paper_select")
    review_pid = review_paper_options[label]
    review_subset = reviews[reviews["PaperID"] == review_pid].copy()

    if not review_subset.empty:
        fig = px.bar(
            review_subset,
            x="ReviewerID",
            y="Score",
            color="Decision",
            text="Score",
            title="Reviewer score comparison",
            category_orders={"Decision": DECISION_ORDER},
        )
        fig.update_yaxes(range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

        for _, row in review_subset.sort_values("Score").iterrows():
            title = f"{row['ReviewerID']} — {row.get('ReviewerPersona', '')} — {float(row['Score']):.1f}/10 — {row['Decision']}"
            with st.expander(title):
                st.write(row.get("Summary", ""))
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    for item in parse_listish(row.get("Strengths", "")):
                        st.markdown(f"- {item}")
                with c2:
                    st.markdown("**Weaknesses**")
                    for item in parse_listish(row.get("Weaknesses", "")):
                        st.markdown(f"- {item}")
                st.markdown("**Relevance**")
                st.write(row.get("Relevance", ""))
                st.markdown("**Conclusion**")
                st.write(row.get("Conclusion", ""))


# ================================================================
# RECOMMENDATION CENTER
# ================================================================

if selected_tab == "Recommendation Center":
    st.subheader("Recommendation Center")
    st.caption(
        "Rule-based editorial interventions combined with a Random Forest delay-risk model "
        "trained on the original EIC-AIssist workflow corpus."
    )

    active = recs[recs["ResolutionStateAsOf"] == "Active"].copy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active manuscripts", len(active))
    c2.metric("High priority", int((active["CombinedPriorityBand"] == "High").sum()))
    c3.metric("Median ML risk", f"{active['MLDelayRisk'].median():.2f}" if not active.empty else "0")
    c4.metric("Median priority", f"{active['CombinedPriorityScore'].median():.1f}" if not active.empty else "0")
    if model_metrics and not np.isnan(model_metrics.get("auc", np.nan)):
        c5.metric("Model AUC", f"{model_metrics['auc']:.2f}")
    else:
        c5.metric("Model AUC", "Fallback")

    if model_metrics:
        st.caption(
            f"Original workflow training rows: {model_metrics.get('train_rows', 0)} · "
            f"Delayed threshold: {model_metrics.get('cutoff_days', 0):.1f} days · "
            f"Test accuracy: {model_metrics.get('accuracy', 0):.2f}"
        )

    action_counts = (
        active.groupby(["RuleRecommendedAction", "CombinedPriorityBand"])
        .size()
        .reset_index(name="Papers")
    )
    fig = px.bar(
        action_counts,
        x="RuleRecommendedAction",
        y="Papers",
        color="CombinedPriorityBand",
        text="Papers",
        title="Recommended actions",
        category_orders={"CombinedPriorityBand": ["High", "Medium", "Low"]},
    )
    fig.update_xaxes(tickangle=35)
    st.plotly_chart(fig, use_container_width=True)

    if not active.empty:
        fig = px.scatter(
            active,
            x="DaysSinceSubmission",
            y="CombinedPriorityScore",
            color="RuleRecommendedAction",
            size="DaysInCurrentStage",
            hover_name="PaperTitle",
            hover_data=["PaperID", "StageAsOf", "MLDelayRisk", "RuleExplanation"],
            title="Active manuscripts by age and intervention priority",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recommended interventions")
    cols = [
        "PaperID", "PaperTitle", "StageAsOf", "HandlingAssociateEditorID",
        "HandlingEIC_ID", "RuleRecommendedAction", "CombinedPriorityBand",
        "CombinedPriorityScore", "MLDelayRisk", "DaysSinceSubmission",
        "DaysInCurrentStage", "SubmittedReviewsAsOf", "OverdueReviewsAsOf",
        "NoResponseAsOf", "PendingInvitesAsOf", "MissingSubmittedReviews",
        "ReviewScoreRange", "RuleExplanation",
    ]
    st.dataframe(
        active[[c for c in cols if c in active.columns]].sort_values("CombinedPriorityScore", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if not active.empty:
        st.markdown("#### Log simulated action")
        rec_choices = {
            f"{int(row.PaperID)} — {row.PaperTitle}": int(row.PaperID)
            for row in active.sort_values("CombinedPriorityScore", ascending=False).itertuples()
        }
        label = st.selectbox("Paper", list(rec_choices.keys()), key="rec_log_paper")
        log_pid = rec_choices[label]
        row = active[active["PaperID"] == log_pid].iloc[0]

        if st.button("Log recommended action", key="rec_log_button"):
            append_csv_row(
                ACTION_LOG_FILE,
                {
                    "ActionID": f"ACT-{uuid.uuid4().hex[:8]}",
                    "CreatedAt": pd.Timestamp.now().isoformat(),
                    "PaperID": log_pid,
                    "ActorID": row.get("HandlingAssociateEditorID", ""),
                    "Action": row["RuleRecommendedAction"],
                    "Reason": row["RuleExplanation"],
                    "PriorityScore": row["CombinedPriorityScore"],
                },
            )
            st.success("Simulated action logged to Unified_Journal/ActionLog.csv.")


# ================================================================
# COMMUNICATIONS
# ================================================================

if selected_tab == "Communications":
    st.subheader("AE Communications")
    st.caption(
        "Drafts are generated automatically from the live manuscript/reviewer state. "
        "They are mock messages only: nothing is sent, and the AE can edit every field before saving."
    )

    active_recs = recs[recs["ResolutionStateAsOf"] == "Active"].copy()

    if active_recs.empty:
        st.info("No active papers at this snapshot.")
    else:
        paper_choices = {
            f"{int(row.PaperID)} — {row.PaperTitle}": int(row.PaperID)
            for row in active_recs.sort_values("CombinedPriorityScore", ascending=False).itertuples()
        }
        paper_label = st.selectbox("Select active manuscript", list(paper_choices.keys()), key="comm_paper")
        pid = paper_choices[paper_label]

        p = paper_state[paper_state["PaperID"] == pid].iloc[0]
        prec = active_recs[active_recs["PaperID"] == pid].iloc[0]
        pa = assignment_state[assignment_state["PaperID"] == pid].copy()

        st.info(
            f"**System recommendation:** {prec['RuleRecommendedAction']}  \n"
            f"**Expected outcome:** {prec.get('RecommendedOutcome', '')}  \n"
            f"**Why:** {prec['RuleExplanation']}"
        )

        c1, c2 = st.columns(2)
        with c1:
            tone = st.selectbox(
                "Tone",
                ["Professional", "Warm", "Concise", "Firm", "Final reminder"],
                key="comm_tone",
            )
        with c2:
            custom_instruction = st.text_input(
                "Optional note to include",
                value="",
                placeholder="e.g., offer a 3-day extension or ask for confirmation today",
                key="comm_custom_instruction",
            )

        disagreement_case = (
            str(prec["RuleRecommendedAction"]).lower().find("adjudicat") >= 0
            or float(prec.get("ReviewScoreRange", 0) or 0) >= 2.0
        )

        suggested = recommend_reviewer(
            pid,
            reviewers,
            assignments,
            snapshot,
            prefer_balanced=disagreement_case,
        )

        if suggested is not None:
            st.caption(
                f"Suggested reviewer: {suggested['ReviewerID']} — "
                f"{suggested.get('ReviewerPersona', '')} · "
                f"active workload {int(suggested.get('ActiveWorkload', 0))}"
            )

        draft_modes = ["Message current reviewer"]
        if suggested is not None:
            draft_modes.append("Invite replacement reviewer")
        if disagreement_case and suggested is not None:
            draft_modes.append("Invite adjudicating reviewer")

        mode = st.radio("Draft purpose", draft_modes, horizontal=True, key="comm_mode")

        assignment_id = ""
        scenario = ""
        to_reviewer = ""

        if mode == "Message current reviewer":
            selectable = pa[
                pa["AssignmentStatusAsOf"].isin(
                    ["Overdue", "No response", "Declined", "Invitation pending", "In review", "Submitted"]
                )
            ].copy()

            if selectable.empty:
                st.info("No reviewer assignment is available for a communication draft.")
                st.stop()

            # Put actionable assignments first.
            status_rank = {
                "Overdue": 0,
                "No response": 1,
                "Declined": 2,
                "Invitation pending": 3,
                "In review": 4,
                "Submitted": 5,
            }
            selectable["_rank"] = selectable["AssignmentStatusAsOf"].map(status_rank).fillna(9)
            selectable = selectable.sort_values(["_rank", "DaysOverdueAsOf"], ascending=[True, False])

            assign_options = {
                f"{row.AssignmentID} — {row.ReviewerID} — {row.AssignmentStatusAsOf}": row.AssignmentID
                for row in selectable.itertuples()
            }
            assign_label = st.selectbox("Reviewer assignment", list(assign_options.keys()), key="comm_assignment")
            assignment_id = assign_options[assign_label]
            a = selectable[selectable["AssignmentID"] == assignment_id].iloc[0]
            scenario, subject, body = generate_mock_message(
                p,
                a,
                tone,
                snapshot,
                suggested_reviewer=suggested,
                custom_instruction=custom_instruction,
            )
            to_reviewer = str(a["ReviewerID"])

        elif mode == "Invite replacement reviewer":
            subject, body = generate_reviewer_invitation(
                p,
                suggested,
                tone,
                purpose="Replacement review",
                custom_instruction=custom_instruction,
            )
            scenario = "Invite replacement reviewer"
            to_reviewer = str(suggested["ReviewerID"])

        else:
            subject, body = generate_reviewer_invitation(
                p,
                suggested,
                tone,
                purpose="Adjudicating review",
                custom_instruction=custom_instruction,
            )
            scenario = "Disagreement — invite adjudicating reviewer"
            to_reviewer = str(suggested["ReviewerID"])

        # Streamlit keeps widget state across reruns. Use a context signature so
        # changing paper/reviewer/tone automatically regenerates a fresh draft,
        # while subsequent typing remains editable.
        signature = "|".join(
            [
                str(pid),
                str(assignment_id),
                str(to_reviewer),
                str(mode),
                str(tone),
                str(snapshot.date()),
                str(custom_instruction),
            ]
        )

        if st.session_state.get("comm_draft_signature") != signature:
            st.session_state["comm_draft_signature"] = signature
            st.session_state["comm_subject_edit"] = subject
            st.session_state["comm_body_edit"] = body

        regen_col, info_col = st.columns([1, 4])
        with regen_col:
            if st.button("Regenerate draft", key="comm_regenerate"):
                st.session_state["comm_subject_edit"] = subject
                st.session_state["comm_body_edit"] = body
                st.rerun()
        with info_col:
            st.caption(f"Detected scenario: {scenario}")

        st.markdown("#### Editable mock message")
        st.text_input("From", value=str(p["HandlingAssociateEditorID"]), disabled=True)
        st.text_input("To", value=str(to_reviewer), disabled=True)
        st.text_input("CC", value=str(p["HandlingEIC_ID"]), disabled=True)
        edited_subject = st.text_input("Subject", key="comm_subject_edit")
        edited_body = st.text_area("Body", height=360, key="comm_body_edit")

        save_col, reset_col = st.columns([1, 1])
        with save_col:
            if st.button("Save mock draft", key="comm_save"):
                append_csv_row(
                    MESSAGES_FILE,
                    {
                        "MessageID": f"MSG-{uuid.uuid4().hex[:8]}",
                        "CreatedAt": pd.Timestamp.now().isoformat(),
                        "PaperID": pid,
                        "AssignmentID": assignment_id,
                        "FromEditorID": p["HandlingAssociateEditorID"],
                        "ToReviewerID": to_reviewer,
                        "CCEditorID": p["HandlingEIC_ID"],
                        "Tone": tone,
                        "Scenario": scenario,
                        "Subject": edited_subject,
                        "Body": edited_body,
                    },
                )
                st.success("Mock draft saved to Unified_Journal/Messages.csv.")

        with reset_col:
            if st.button("Reset to generated draft", key="comm_reset"):
                st.session_state["comm_subject_edit"] = subject
                st.session_state["comm_body_edit"] = body
                st.rerun()

        if MESSAGES_FILE.exists():
            saved = pd.read_csv(MESSAGES_FILE)
            if not saved.empty and "PaperID" in saved.columns:
                saved = saved[pd.to_numeric(saved["PaperID"], errors="coerce") == pid]
                if not saved.empty:
                    st.markdown("#### Saved mock drafts")
                    st.dataframe(
                        saved[[
                            c for c in [
                                "CreatedAt", "ToReviewerID", "CCEditorID", "Tone",
                                "Scenario", "Subject"
                            ] if c in saved.columns
                        ]],
                        use_container_width=True,
                        hide_index=True,
                    )


# ================================================================
# SCENARIO LAB
# ================================================================

if selected_tab == "Scenario Lab":
    st.subheader("Scenario Lab")
    st.caption(
        "A demonstration set of realistic journal conditions seeded into the 103-paper simulation. "
        "Use this view to show the condition, system recommendation, action, and expected outcome."
    )

    scenario_df = pd.read_csv(SCENARIO_FILE) if SCENARIO_FILE.exists() else pd.DataFrame()
    scenario_view = recs.merge(
        scenario_df[[c for c in ["PaperID", "ShowcaseScenario", "ExpectedAction", "ExpectedOutcome"] if c in scenario_df.columns]],
        on="PaperID",
        how="left",
        suffixes=("", "_Scenario"),
    ) if not scenario_df.empty else recs.copy()

    if "ShowcaseScenario" not in scenario_view.columns:
        scenario_view["ShowcaseScenario"] = scenario_view.get("ShowcaseScenario_Scenario", "General workflow")

    scenario_options = sorted(scenario_view["ShowcaseScenario"].dropna().astype(str).unique().tolist())
    selected_scenarios = st.multiselect(
        "Filter scenarios",
        scenario_options,
        default=[],
        key="scenario_lab_filter",
    )
    if selected_scenarios:
        scenario_view = scenario_view[scenario_view["ShowcaseScenario"].astype(str).isin(selected_scenarios)].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenario cases", len(scenario_view))
    c2.metric("Distinct conditions", scenario_view["ShowcaseScenario"].nunique())
    c3.metric("High-priority cases", int((scenario_view["CombinedPriorityBand"] == "High").sum()))
    c4.metric("Active cases", int((scenario_view["ResolutionStateAsOf"] == "Active").sum()))

    scenario_counts = scenario_view["ShowcaseScenario"].value_counts().reset_index()
    scenario_counts.columns = ["Scenario", "Papers"]
    fig = px.bar(
        scenario_counts,
        x="Papers",
        y="Scenario",
        orientation="h",
        text="Papers",
        title="Demonstration conditions represented in the journal",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(500, 34 * len(scenario_counts) + 160))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Condition → recommendation → outcome")
    cols = [
        "PaperID", "PaperTitle", "ShowcaseScenario", "StageAsOf",
        "RuleRecommendedAction", "RecommendedOutcome", "ExpectedAction", "ExpectedOutcome",
        "CombinedPriorityBand", "CombinedPriorityScore", "RuleExplanation",
    ]
    st.dataframe(
        scenario_view[[c for c in cols if c in scenario_view.columns]]
        .sort_values(["CombinedPriorityScore", "PaperID"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True,
        height=560,
    )

    st.markdown("#### Inspect one scenario")
    if not scenario_view.empty:
        choices = {
            f"{int(r.PaperID)} — {r.ShowcaseScenario} — {r.PaperTitle}": int(r.PaperID)
            for r in scenario_view.sort_values("PaperID").itertuples()
        }
        label = st.selectbox("Scenario manuscript", list(choices.keys()), key="scenario_case_select")
        spid = choices[label]
        sr = scenario_view[scenario_view["PaperID"] == spid].iloc[0]
        sa = assignment_state[assignment_state["PaperID"] == spid].copy()

        st.info(
            f"**Condition:** {sr.get('ShowcaseScenario', '')}  \n"
            f"**Detected stage:** {sr.get('StageAsOf', '')}  \n"
            f"**Recommended action:** {sr.get('RuleRecommendedAction', '')}  \n"
            f"**Expected outcome:** {sr.get('RecommendedOutcome', sr.get('ExpectedOutcome', ''))}"
        )

        st.dataframe(
            sa[[c for c in [
                "ReviewerID", "AssignmentStatusAsOf", "DateReviewerInvited",
                "DateReviewDue", "DateReviewSubmitted", "DaysOverdueAsOf",
                "DaysToDueAsOf", "ReviewerWorkloadAtInvite"
            ] if c in sa.columns]],
            use_container_width=True,
            hide_index=True,
        )

# ================================================================
# SOURCE COMPARISON
# ================================================================

if selected_tab == "Source Comparison":
    st.subheader("Source Comparison")
    comparison = load_source_comparison()

    if comparison is None:
        st.info(
            "Source-comparison outputs were not found. Place SourceComparisonReviews.csv, "
            "SourceComparisonPairs.csv, SourceComparisonPaperSummary.csv and "
            "PersonaSourceSensitivity.csv either beside this app or in "
            "AI_Scientist/source_comparison_experiment/outputs."
        )
    else:
        comp_reviews = comparison["reviews"]
        pairs = comparison["pairs"]
        summary = comparison["summary"]
        persona = comparison["persona"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Comparison reviews", len(comp_reviews))
        c2.metric("Papers", comp_reviews["PaperID"].nunique() if "PaperID" in comp_reviews.columns else "—")
        c3.metric("Personas", comp_reviews["ReviewerID"].nunique() if "ReviewerID" in comp_reviews.columns else "—")
        c4.metric("Paired comparisons", len(pairs))

        if "SourceCondition" in comp_reviews.columns and "Score" in comp_reviews.columns:
            fig = px.box(
                comp_reviews,
                x="SourceCondition",
                y="Score",
                points="all",
                title="Review score distribution by upstream source",
            )
            st.plotly_chart(fig, use_container_width=True)

        delta_col = next(
            (
                col
                for col in ["ScoreDelta", "Score_Delta", "StanfordMinusAIScientist", "AbsoluteScoreDelta"]
                if col in pairs.columns
            ),
            None,
        )
        if delta_col:
            fig = px.histogram(pairs, x=delta_col, nbins=20, title=f"Paired score shift ({delta_col})")
            st.plotly_chart(fig, use_container_width=True)

        if not persona.empty and "ReviewerID" in persona.columns:
            numeric_cols = [c for c in persona.columns if pd.api.types.is_numeric_dtype(persona[c])]
            if numeric_cols:
                metric = numeric_cols[-1]
                fig = px.bar(
                    persona.sort_values(metric),
                    x=metric,
                    y="ReviewerID",
                    orientation="h",
                    title=f"Reviewer sensitivity by source ({metric})",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Paper-level source comparison")
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ================================================================
# DATA EXPLORER
# ================================================================

if selected_tab == "Data Explorer":
    st.subheader("Unified journal datasets")

    table_map = {
        "Papers": papers,
        "Editors": editors,
        "Reviewers": reviewers,
        "ReviewerAssignments": assignment_state,
        "Reviews": reviews,
        "PaperDecisions": decisions,
        "Recommendations": recs,
        "Messages": pd.read_csv(MESSAGES_FILE) if MESSAGES_FILE.exists() else pd.DataFrame(),
        "ActionLog": pd.read_csv(ACTION_LOG_FILE) if ACTION_LOG_FILE.exists() else pd.DataFrame(),
    }

    table_name = st.selectbox("Dataset", list(table_map.keys()))
    df = table_map[table_name]
    st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")
    st.dataframe(df, use_container_width=True, height=560)

    st.download_button(
        f"Download {table_name}.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_name}.csv",
        mime="text/csv",
    )
