# app.py
# Streamlit dashboard to explore EIC-AIssist synthetic peer-review dataset (v2)
# Works with the files shared in this chat:
#   - EIC-AIssist_peer_review_synth_v2.xlsx  (sheets: PaperHeader, ReviewerRows)
#   - OR: EIC-AIssist_PaperHeader_v2.csv + EIC-AIssist_ReviewerRows_v2.csv
#
# Run:
#   pip install streamlit pandas numpy plotly scipy openpyxl
#   streamlit run app.py

import io
import math
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="EIC-AIssist | Peer Review Workflow Diagnostics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EIC-AIssist — Peer Review Workflow Diagnostics")
st.caption(
    "Interactive dashboard to validate distributions, heavy tails, missingness, and workflow timeline constraints "
    "for PaperHeader + ReviewerRows tables."
)


# -----------------------------
# Helpers
# -----------------------------
def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _duration_days(df: pd.DataFrame, start_col: str, end_col: str, out_col: str) -> None:
    if start_col in df.columns and end_col in df.columns:
        df[out_col] = (df[end_col] - df[start_col]).dt.total_seconds() / 86400.0


def _infer_table_kind(columns: List[str]) -> str:
    # crude inference for user uploads
    cols = set(columns)
    if "InviteOutcome" in cols or "ReviewerID" in cols:
        return "ReviewerRows"
    if "PaperStatusOnSubmission" in cols or "SubmissionRound" in cols:
        return "PaperHeader"
    return "Unknown"


@st.cache_data(show_spinner=False)
def load_from_excel_bytes(xlsx_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bio = io.BytesIO(xlsx_bytes)
    xls = pd.ExcelFile(bio)
    sheet_names = [s.strip() for s in xls.sheet_names]
    # Prefer standard names
    ph_name = "PaperHeader" if "PaperHeader" in sheet_names else sheet_names[0]
    rr_name = "ReviewerRows" if "ReviewerRows" in sheet_names else (sheet_names[1] if len(sheet_names) > 1 else sheet_names[0])
    paper = pd.read_excel(bio, sheet_name=ph_name)
    bio.seek(0)
    rev = pd.read_excel(bio, sheet_name=rr_name)
    return paper, rev


@st.cache_data(show_spinner=False)
def load_from_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(csv_bytes))


@st.cache_data(show_spinner=False)
def load_local_files(xlsx_path: str, paper_csv_path: str, reviewer_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Returns (paper_df, reviewer_df, source_label).
    Tries Excel first, then CSVs.
    """
    import os
    if xlsx_path and os.path.exists(xlsx_path):
        paper = pd.read_excel(xlsx_path, sheet_name="PaperHeader")
        rev = pd.read_excel(xlsx_path, sheet_name="ReviewerRows")
        return paper, rev, f"Loaded Excel: {xlsx_path}"
    if paper_csv_path and reviewer_csv_path and os.path.exists(paper_csv_path) and os.path.exists(reviewer_csv_path):
        paper = pd.read_csv(paper_csv_path)
        rev = pd.read_csv(reviewer_csv_path)
        return paper, rev, f"Loaded CSVs: {paper_csv_path}, {reviewer_csv_path}"
    return pd.DataFrame(), pd.DataFrame(), "No local files found."


def add_derived_columns(paper: pd.DataFrame, rev: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ---- Date parsing (paper)
    paper_date_cols = [
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "EIC_DecisionDate",
        "DateDecisionLetterSent",
    ]
    paper = _to_datetime(paper, paper_date_cols)

    # ---- Date parsing (reviewers)
    rev_date_cols = [
        "DateReviewerInvited",
        "DateInvitationAccepted",
        "DateInvitationResolved",
        "DateReviewDue",
        "DateReviewSubmitted",
        "DateFirstReminderSent",
        "DateLastReminderSent",
    ]
    rev = _to_datetime(rev, rev_date_cols)

    # ---- Numeric fixes
    for c in ["SubmissionRound", "TargetNumberOfReviewers", "TotalTime_SubmissionToDecision_Days"]:
        if c in paper.columns:
            paper[c] = _safe_numeric(paper[c])

    for c in ["ReviewerWorkloadAtInvite", "NumRemindersSent", "ReviewLengthWords", "ReviewerDisagreementScore"]:
        if c in rev.columns:
            rev[c] = _safe_numeric(rev[c])

    # ---- Core derived durations (paper)
    _duration_days(paper, "DatePaperSubmitted", "DateReviewersFullyAssigned", "D_SubmitToFullyAssigned")
    _duration_days(paper, "DatePaperSubmitted", "DateFirstReviewReceived", "D_SubmitToFirstReview")
    _duration_days(paper, "DatePaperSubmitted", "DateAllReviewsReceived", "D_SubmitToAllReviews")
    _duration_days(paper, "DateAllReviewsReceived", "AE_RecommendationDate", "D_AllReviewsToAERec")
    _duration_days(paper, "AE_RecommendationDate", "EIC_DecisionDate", "D_AERecToEICDecision")
    _duration_days(paper, "EIC_DecisionDate", "DateDecisionLetterSent", "D_EICDecisionToLetter")

    # If total time column missing, compute
    if "TotalTime_SubmissionToDecision_Days" not in paper.columns and "DateDecisionLetterSent" in paper.columns and "DatePaperSubmitted" in paper.columns:
        _duration_days(paper, "DatePaperSubmitted", "DateDecisionLetterSent", "TotalTime_SubmissionToDecision_Days")

    # ---- Core derived durations (reviewers)
    _duration_days(rev, "DateReviewerInvited", "DateInvitationResolved", "D_InviteToResolved")
    _duration_days(rev, "DateInvitationAccepted", "DateReviewSubmitted", "D_AcceptToSubmit")
    _duration_days(rev, "DateInvitationAccepted", "DateReviewDue", "D_AcceptToDue")

    if "DateReviewSubmitted" in rev.columns and "DateReviewDue" in rev.columns:
        rev["D_OverdueDays"] = (rev["DateReviewSubmitted"] - rev["DateReviewDue"]).dt.total_seconds() / 86400.0

    # Late flag if missing
    if "LateSubmissionFlag" not in rev.columns and "D_OverdueDays" in rev.columns:
        rev["LateSubmissionFlag"] = np.where(rev["D_OverdueDays"] > 0, "yes", "no")

    # ---- Join keys sanity
    for key in ["PaperID", "SubmissionRound"]:
        if key in paper.columns and key in rev.columns:
            # ok
            pass

    return paper, rev


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "column": miss.index,
        "missing_count": miss.values,
        "missing_pct": (miss.values / max(1, n)) * 100.0,
        "dtype": [str(df[c].dtype) for c in miss.index],
    }).sort_values("missing_pct", ascending=False)
    return out


def workflow_violations(paper: pd.DataFrame) -> pd.DataFrame:
    """
    Checks monotonic timeline constraints per paper-round:
    submitted <= fully assigned <= first review <= all reviews <= AE rec <= EIC decision <= letter sent
    Returns counts and examples.
    """
    cols = [
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "EIC_DecisionDate",
        "DateDecisionLetterSent",
    ]
    present = [c for c in cols if c in paper.columns]
    if len(present) < 3:
        return pd.DataFrame(columns=["rule", "violations", "violation_pct"])

    rules = []
    pairs = [
        ("Submitted <= FullyAssigned", "DatePaperSubmitted", "DateReviewersFullyAssigned"),
        ("FullyAssigned <= FirstReview", "DateReviewersFullyAssigned", "DateFirstReviewReceived"),
        ("FirstReview <= AllReviews", "DateFirstReviewReceived", "DateAllReviewsReceived"),
        ("AllReviews <= AERec", "DateAllReviewsReceived", "AE_RecommendationDate"),
        ("AERec <= EICDecision", "AE_RecommendationDate", "EIC_DecisionDate"),
        ("EICDecision <= LetterSent", "EIC_DecisionDate", "DateDecisionLetterSent"),
    ]
    n = len(paper)
    for label, a, b in pairs:
        if a in paper.columns and b in paper.columns:
            mask = paper[a].notna() & paper[b].notna() & (paper[a] > paper[b])
            rules.append({
                "rule": label,
                "violations": int(mask.sum()),
                "violation_pct": (mask.sum() / max(1, n)) * 100.0,
            })
    return pd.DataFrame(rules).sort_values("violations", ascending=False)


def fit_distribution(series: pd.Series, dist_name: str):
    """
    Fits a distribution to positive values using scipy.
    Returns (params, fitted_dist_obj).
    """
    x = series.dropna().values.astype(float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 50:
        return None, None

    if dist_name == "lognormal":
        # stats.lognorm parameterization: s=shape, loc, scale
        params = stats.lognorm.fit(x, floc=0)
        return params, stats.lognorm(*params)
    if dist_name == "gamma":
        params = stats.gamma.fit(x, floc=0)
        return params, stats.gamma(*params)
    if dist_name == "weibull":
        params = stats.weibull_min.fit(x, floc=0)
        return params, stats.weibull_min(*params)

    return None, None


def plot_numeric_distribution(series: pd.Series, title: str, log_x: bool, dist_overlay: Optional[str] = None):
    x = series.dropna().astype(float)
    x = x[np.isfinite(x)]
    fig = go.Figure()

    # histogram
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=60,
        name="Empirical",
        opacity=0.75,
        histnorm="probability density",
    ))

    # overlay fit curve
    if dist_overlay:
        params, dist = fit_distribution(x, dist_overlay)
        if dist is not None:
            # grid on positive range
            xmin = max(1e-6, float(np.percentile(x[x > 0], 1)))
            xmax = float(np.percentile(x[x > 0], 99))
            grid = np.geomspace(xmin, xmax, 300) if log_x else np.linspace(xmin, xmax, 300)
            pdf = dist.pdf(grid)
            fig.add_trace(go.Scatter(x=grid, y=pdf, mode="lines", name=f"{dist_overlay} fit"))
            fig.update_layout(
                title=f"{title} (overlay: {dist_overlay})",
            )
            st.caption(f"Fit params ({dist_overlay}): {params}")
        else:
            fig.update_layout(title=title)
            st.caption("Not enough positive samples to fit distribution (need ~50+).")
    else:
        fig.update_layout(title=title)

    fig.update_layout(
        bargap=0.02,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def plot_qq(series: pd.Series, dist_name: str, title: str):
    x = series.dropna().astype(float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 50:
        st.warning("Not enough positive samples for Q-Q plot (need ~50+).")
        return None

    params, dist = fit_distribution(pd.Series(x), dist_name)
    if dist is None:
        st.warning("Fit failed.")
        return None

    # empirical quantiles
    x_sorted = np.sort(x)
    p = (np.arange(1, len(x_sorted) + 1) - 0.5) / len(x_sorted)
    q_theory = dist.ppf(p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_theory, y=x_sorted, mode="markers", name="Q-Q"))
    # 45-degree line
    lo = float(min(q_theory.min(), x_sorted.min()))
    hi = float(max(q_theory.max(), x_sorted.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x"))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Theoretical quantiles",
        yaxis_title="Empirical quantiles",
    )
    st.caption(f"Fit params ({dist_name}): {params}")
    return fig


# -----------------------------
# Sidebar: data loading (FIXED with session_state)
# -----------------------------
st.sidebar.header("Data source")

default_xlsx = "EIC-AIssist_peer_review_synth_v6.xlsx"
default_paper_csv = "EIC-AIssist_PaperHeader_v6.csv"
default_reviewer_csv = "EIC-AIssist_ReviewerRows_v6.csv"

# initialize session state keys
if "paper_df" not in st.session_state:
    st.session_state["paper_df"] = pd.DataFrame()
if "rev_df" not in st.session_state:
    st.session_state["rev_df"] = pd.DataFrame()
if "source_label" not in st.session_state:
    st.session_state["source_label"] = ""

source_mode = st.sidebar.radio(
    "Load mode",
    ["Use local files (recommended)", "Upload files"],
    index=0,
)

# helper: store + derive once
def _store_loaded(paper_loaded: pd.DataFrame, rev_loaded: pd.DataFrame, label: str) -> None:
    paper_loaded, rev_loaded = add_derived_columns(paper_loaded, rev_loaded)
    st.session_state["paper_df"] = paper_loaded
    st.session_state["rev_df"] = rev_loaded
    st.session_state["source_label"] = label

# UI + loading logic
if source_mode == "Use local files (recommended)":
    xlsx_path = st.sidebar.text_input("Excel path", value=default_xlsx)
    paper_csv_path = st.sidebar.text_input("PaperHeader CSV path (fallback)", value=default_paper_csv)
    reviewer_csv_path = st.sidebar.text_input("ReviewerRows CSV path (fallback)", value=default_reviewer_csv)

    auto_load = st.sidebar.checkbox("Auto-load on rerun", value=True)

    if st.sidebar.button("Load data") or (auto_load and st.session_state["paper_df"].empty):
        paper_tmp, rev_tmp, label = load_local_files(xlsx_path, paper_csv_path, reviewer_csv_path)
        if not paper_tmp.empty and not rev_tmp.empty:
            _store_loaded(paper_tmp, rev_tmp, label)
        else:
            st.sidebar.error("Could not load local files. Check paths / filenames.")

else:
    st.sidebar.write("Upload either the Excel workbook OR both CSVs.")
    up_xlsx = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    up_paper = st.sidebar.file_uploader("Upload PaperHeader CSV", type=["csv"])
    up_rev = st.sidebar.file_uploader("Upload ReviewerRows CSV", type=["csv"])

    if st.sidebar.button("Load uploads"):
        if up_xlsx is not None:
            paper_tmp, rev_tmp = load_from_excel_bytes(up_xlsx.getvalue())
            _store_loaded(paper_tmp, rev_tmp, f"Loaded uploaded Excel: {up_xlsx.name}")
        elif up_paper is not None and up_rev is not None:
            paper_tmp = load_from_csv_bytes(up_paper.getvalue())
            rev_tmp = load_from_csv_bytes(up_rev.getvalue())
            _store_loaded(paper_tmp, rev_tmp, f"Loaded uploaded CSVs: {up_paper.name}, {up_rev.name}")
        else:
            st.sidebar.error("Upload Excel OR both CSVs.")

# pull from session_state every rerun
paper_df = st.session_state["paper_df"]
rev_df = st.session_state["rev_df"]
source_label = st.session_state["source_label"]

# allow clearing
if st.sidebar.button("Clear loaded data"):
    st.session_state["paper_df"] = pd.DataFrame()
    st.session_state["rev_df"] = pd.DataFrame()
    st.session_state["source_label"] = ""
    st.rerun()

# gate the rest of the app
if paper_df.empty or rev_df.empty:
    st.info("Load the dataset using the sidebar to begin.")
    st.stop()

st.success(source_label)



# -----------------------------
# Global filters
# -----------------------------
st.sidebar.header("Filters")

# Date range filter (paper submission date)
sub_col = "DatePaperSubmitted" if "DatePaperSubmitted" in paper_df.columns else None
if sub_col:
    min_d = pd.to_datetime(paper_df[sub_col].min())
    max_d = pd.to_datetime(paper_df[sub_col].max())
    date_range = st.sidebar.date_input("Submission date range", value=(min_d.date(), max_d.date()))
else:
    date_range = None

# Paper-level categorical filters
section_col = _pick_first_existing(paper_df, ["JournalSection", "Section", "SubjectArea"])
ae_col = _pick_first_existing(paper_df, ["HandlingAssociateEditorID", "AE_ID", "AssociateEditorID"])
eic_col = _pick_first_existing(paper_df, ["HandlingEIC_ID", "EIC_ID", "EditorInChiefID"])
round_col = _pick_first_existing(paper_df, ["SubmissionRound", "SubmissionRounds", "Round"])
scenario_col = _pick_first_existing(paper_df, ["ScenarioLabel", "Scenario", "HolidaySurgeLabel"])

def multiselect_filter(df: pd.DataFrame, col: Optional[str], label: str) -> pd.DataFrame:
    if col and col in df.columns:
        values = sorted([v for v in df[col].dropna().astype(str).unique().tolist()])
        selected = st.sidebar.multiselect(label, values, default=[])
        if selected:
            return df[df[col].astype(str).isin(selected)]
    return df

# apply paper filters
paper_f = paper_df.copy()
if sub_col and date_range and len(date_range) == 2:
    d0 = pd.to_datetime(date_range[0])
    d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    paper_f = paper_f[(paper_f[sub_col] >= d0) & (paper_f[sub_col] <= d1)]

paper_f = multiselect_filter(paper_f, section_col, "Journal section")
paper_f = multiselect_filter(paper_f, ae_col, "Handling AE")
paper_f = multiselect_filter(paper_f, eic_col, "Handling EIC")
paper_f = multiselect_filter(paper_f, scenario_col, "Scenario label")

if round_col and round_col in paper_f.columns:
    rounds = sorted([int(x) for x in paper_f[round_col].dropna().unique().tolist() if float(x).is_integer()])
    sel_rounds = st.sidebar.multiselect("Submission rounds", rounds, default=[])
    if sel_rounds:
        paper_f = paper_f[paper_f[round_col].isin(sel_rounds)]

# join reviewer rows to filtered papers using keys
join_keys = []
if "PaperID" in paper_f.columns and "PaperID" in rev_df.columns:
    join_keys.append("PaperID")
if "SubmissionRound" in paper_f.columns and "SubmissionRound" in rev_df.columns:
    join_keys.append("SubmissionRound")

rev_f = rev_df.copy()
if join_keys:
    keyset = paper_f[join_keys].drop_duplicates()
    rev_f = rev_f.merge(keyset, on=join_keys, how="inner")

# Reviewer filters
tier_col = _pick_first_existing(rev_f, ["ReviewerReliabilityTier", "ReviewerRatingLevel", "ReliabilityTier"])
outcome_col = _pick_first_existing(rev_f, ["InviteOutcome", "Invite_Status", "InvitationOutcome"])
rev_f = multiselect_filter(rev_f, tier_col, "Reviewer reliability tier")
rev_f = multiselect_filter(rev_f, outcome_col, "Invite outcome")


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_focus, tab_distributions, tab_fits, tab_sanity, tab_tables = st.tabs(
    ["Overview", "Your focus columns", "Explore distributions", "Heavy-tail fits", "Sanity checks", "Data tables"]
)


# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper-round rows", f"{len(paper_f):,}")
    c2.metric("Reviewer invite rows", f"{len(rev_f):,}")
    c3.metric("Unique papers", f"{paper_f['PaperID'].nunique():,}" if "PaperID" in paper_f.columns else "—")
    c4.metric("Unique reviewers", f"{rev_f['ReviewerID'].nunique():,}" if "ReviewerID" in rev_f.columns else "—")

    st.subheader("Missingness snapshot")
    miss_p = missingness_summary(paper_f)
    miss_r = missingness_summary(rev_f)

    colA, colB = st.columns(2)
    with colA:
        st.caption("PaperHeader missingness (%)")
        fig = px.bar(miss_p.head(20), x="missing_pct", y="column", orientation="h", hover_data=["missing_count", "dtype"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.caption("ReviewerRows missingness (%)")
        fig = px.bar(miss_r.head(20), x="missing_pct", y="column", orientation="h", hover_data=["missing_count", "dtype"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick timeline bottleneck overview (paper-level)")
    dur_cols = [
        "D_SubmitToFullyAssigned",
        "D_SubmitToFirstReview",
        "D_SubmitToAllReviews",
        "D_AllReviewsToAERec",
        "D_AERecToEICDecision",
        "D_EICDecisionToLetter",
        "TotalTime_SubmissionToDecision_Days",
    ]
    dur_cols = [c for c in dur_cols if c in paper_f.columns]
    if dur_cols:
        stats_df = pd.DataFrame({
            "duration": dur_cols,
            "median": [paper_f[c].median(skipna=True) for c in dur_cols],
            "p90": [paper_f[c].quantile(0.9) for c in dur_cols],
            "p95": [paper_f[c].quantile(0.95) for c in dur_cols],
            "max": [paper_f[c].max(skipna=True) for c in dur_cols],
        }).round(2)
        st.dataframe(stats_df, use_container_width=True)


# -----------------------------
# Your requested focus columns
# -----------------------------
with tab_focus:
    st.subheader("1) SubmissionRound")
    left, right = st.columns([1, 1])
    with left:
        if "SubmissionRound" in paper_f.columns:
            fig = px.bar(
                paper_f["SubmissionRound"].value_counts().sort_index().reset_index(),
                x="SubmissionRound",
                y="count",
                labels={"index": "SubmissionRound", "count": "Count"},
                title="SubmissionRound distribution (PaperHeader)",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SubmissionRound column not found in PaperHeader.")
    with right:
        if "PaperID" in paper_f.columns and "SubmissionRound" in paper_f.columns:
            max_round = paper_f.groupby("PaperID")["SubmissionRound"].max().reset_index(name="MaxRound")
            fig = px.bar(
                max_round["MaxRound"].value_counts().sort_index().reset_index(),
                x="MaxRound",
                y="count",
                title="Max submission round per PaperID",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("2) ReviewerWorkloadAtInvite")
    left, right = st.columns([1, 1])
    workload_col = _pick_first_existing(rev_f, ["ReviewerWorkloadAtInvite", "ReviewerWorkload", "WorkloadAtInvite"])
    if workload_col:
        with left:
            fig = px.histogram(
                rev_f,
                x=workload_col,
                nbins=50,
                title=f"{workload_col} histogram",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(
                rev_f,
                y=workload_col,
                points="outliers",
                title=f"{workload_col} box (outliers visible)",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Reviewer workload column not found.")

    st.subheader("3) InviteOutcome")
    left, right = st.columns([1, 1])
    if outcome_col:
        with left:
            vc = (
                rev_f[outcome_col]
                .astype(str)
                .value_counts(dropna=False)
                .reset_index()
            )
            vc.columns = ["InviteOutcome", "count"]

            fig = px.bar(
                vc,
                x="InviteOutcome",
                y="count",
                title=f"{outcome_col} distribution",
            )

            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=50, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

            fig.update_layout(height=420, xaxis_title="InviteOutcome", yaxis_title="Count", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            # acceptance rate vs workload (binned)
            if workload_col:
                tmp = rev_f[[workload_col, outcome_col]].dropna()
                if len(tmp) > 0:
                    tmp["workload_bin"] = pd.cut(tmp[workload_col], bins=10, duplicates="drop").astype(str)
                    acc = tmp.groupby("workload_bin")[outcome_col].apply(lambda s: (s.astype(str) == "accept").mean()).reset_index(name="accept_rate")
                    fig = px.bar(acc, x="workload_bin", y="accept_rate", title="Acceptance rate by workload bin")
                    fig.update_layout(height=420, yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("InviteOutcome column not found.")


# -----------------------------
# Explore distributions (any column)
# -----------------------------
with tab_distributions:
    st.subheader("Explore any column (interactive)")

    table_choice = st.radio("Choose table", ["PaperHeader", "ReviewerRows"], horizontal=True)
    df = paper_f if table_choice == "PaperHeader" else rev_f

    # Identify numeric/categorical columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        col_type = st.selectbox("Column type", ["numeric", "categorical", "datetime"], index=0)
    with c2:
        if col_type == "numeric":
            col_sel = st.selectbox("Select numeric column", numeric_cols, index=0 if numeric_cols else None)
        elif col_type == "categorical":
            col_sel = st.selectbox("Select categorical column", cat_cols, index=0 if cat_cols else None)
        else:
            col_sel = st.selectbox("Select datetime column", dt_cols, index=0 if dt_cols else None)
    with c3:
        if col_type == "numeric":
            log_x = st.checkbox("Log x-axis", value=False)
        else:
            log_x = False

    if col_sel:
        if col_type == "numeric":
            nbins = st.slider("Histogram bins", 10, 120, 50, 5)
            fig = px.histogram(df, x=col_sel, nbins=nbins, title=f"Histogram: {col_sel}")
            if log_x:
                fig.update_xaxes(type="log")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # ECDF
            fig2 = px.ecdf(df, x=col_sel, title=f"ECDF: {col_sel}")
            if log_x:
                fig2.update_xaxes(type="log")
            fig2.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        elif col_type == "categorical":
            top_k = st.slider("Top-K categories", 5, 60, 20, 1)
            vc = df[col_sel].astype(str).value_counts(dropna=False).head(top_k).reset_index()
            vc.columns = ["category", "count"]
            fig = px.bar(vc, x="category", y="count", title=f"Top {top_k}: {col_sel}")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        else:
            # datetime: counts by week/month
            gran = st.selectbox("Time granularity", ["D", "W", "M"], index=1)
            tmp = df[[col_sel]].dropna().copy()
            tmp["bucket"] = tmp[col_sel].dt.to_period(gran).dt.to_timestamp()
            agg = tmp.groupby("bucket").size().reset_index(name="count")
            fig = px.line(agg, x="bucket", y="count", title=f"Event volume over time: {col_sel} ({gran})")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Descriptive stats
        st.subheader("Quick stats")
        st.write(df[[col_sel]].describe(include="all").T)


# -----------------------------
# Heavy-tail fits (lognormal/gamma/weibull + Q-Q)
# -----------------------------
with tab_fits:
    st.subheader("Heavy-tail diagnostics (fit + overlay + Q-Q)")

    fit_table = st.radio("Choose table for fitting", ["PaperHeader", "ReviewerRows"], horizontal=True)
    df_fit = paper_f if fit_table == "PaperHeader" else rev_f

    numeric_cols_fit = [c for c in df_fit.columns if pd.api.types.is_numeric_dtype(df_fit[c])]
    # Recommend typical heavy-tail columns
    recommended = []
    if fit_table == "PaperHeader":
        recommended = [c for c in [
            "D_SubmitToFullyAssigned",
            "D_SubmitToFirstReview",
            "D_SubmitToAllReviews",
            "D_AllReviewsToAERec",
            "D_AERecToEICDecision",
            "TotalTime_SubmissionToDecision_Days",
        ] if c in numeric_cols_fit]
    else:
        recommended = [c for c in [
            "ReviewerWorkloadAtInvite",
            "D_InviteToResolved",
            "D_AcceptToSubmit",
            "D_OverdueDays",
            "ReviewLengthWords",
            "NumRemindersSent",
        ] if c in numeric_cols_fit]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        fit_col = st.selectbox("Numeric column to fit", recommended + [c for c in numeric_cols_fit if c not in recommended])
    with col2:
        dist_name = st.selectbox("Distribution", ["lognormal", "gamma", "weibull"], index=0)
    with col3:
        log_x = st.checkbox("Log x-axis (plot)", value=True)

    if fit_col:
        fig = plot_numeric_distribution(df_fit[fit_col], title=fit_col, log_x=log_x, dist_overlay=dist_name)
        st.plotly_chart(fig, use_container_width=True)

        qq = plot_qq(df_fit[fit_col], dist_name=dist_name, title=f"Q-Q plot: {fit_col} vs {dist_name}")
        if qq is not None:
            st.plotly_chart(qq, use_container_width=True)

        st.subheader("Outliers (top 20)")
        s = df_fit[fit_col].dropna()
        if len(s) > 0:
            top = df_fit.loc[s.sort_values(ascending=False).head(20).index]
            st.dataframe(top, use_container_width=True)


# -----------------------------
# Sanity checks (timeline constraints + logical consistency)
# -----------------------------
with tab_sanity:
    st.subheader("Timeline monotonicity checks (PaperHeader)")
    vio = workflow_violations(paper_f)
    if len(vio) == 0:
        st.info("Not enough timeline columns found to run monotonicity checks.")
    else:
        st.dataframe(vio, use_container_width=True)
        fig = px.bar(vio, x="violations", y="rule", orientation="h", title="Violation counts by rule")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reviewer consistency checks (ReviewerRows)")
    checks = []
    if "InviteOutcome" in rev_f.columns and "DateInvitationAccepted" in rev_f.columns:
        mask = (rev_f["InviteOutcome"].astype(str) != "accept") & (rev_f["DateInvitationAccepted"].notna())
        checks.append(("Non-accept outcome but has acceptance date", int(mask.sum()), float(mask.mean() * 100)))
    if "InviteOutcome" in rev_f.columns and "DateReviewSubmitted" in rev_f.columns:
        mask = (rev_f["InviteOutcome"].astype(str) != "accept") & (rev_f["DateReviewSubmitted"].notna())
        checks.append(("Non-accept outcome but has submitted date", int(mask.sum()), float(mask.mean() * 100)))
    if "DateReviewDue" in rev_f.columns and "DateReviewSubmitted" in rev_f.columns:
        mask = rev_f["DateReviewDue"].notna() & rev_f["DateReviewSubmitted"].notna() & (rev_f["DateReviewSubmitted"] < rev_f["DateReviewerInvited"])
        checks.append(("Submitted before invited (impossible)", int(mask.sum()), float(mask.mean() * 100)))

    chk_df = pd.DataFrame(checks, columns=["check", "count", "pct_of_rows"]).sort_values("count", ascending=False)
    if len(chk_df) == 0:
        st.info("Not enough columns to run reviewer checks.")
    else:
        st.dataframe(chk_df, use_container_width=True)


# -----------------------------
# Data tables
# -----------------------------
with tab_tables:
    st.subheader("Filtered data preview")

    left, right = st.columns(2)
    with left:
        st.caption("PaperHeader (filtered)")
        st.dataframe(paper_f.head(200), use_container_width=True, height=420)
    with right:
        st.caption("ReviewerRows (filtered)")
        st.dataframe(rev_f.head(200), use_container_width=True, height=420)

    st.subheader("Download filtered data")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download PaperHeader (filtered) as CSV",
            data=paper_f.to_csv(index=False).encode("utf-8"),
            file_name="PaperHeader_filtered.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download ReviewerRows (filtered) as CSV",
            data=rev_f.to_csv(index=False).encode("utf-8"),
            file_name="ReviewerRows_filtered.csv",
            mime="text/csv",
        )


st.caption("Tip: Use the 'Heavy-tail fits' tab to see whether your lognormal tails look right (overlay + Q-Q).")
