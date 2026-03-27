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

default_xlsx = "EIC-AIssist_peer_review_synth_v8.xlsx"
default_paper_csv = "EIC-AIssist_PaperHeader_v8.csv"
default_reviewer_csv = "EIC-AIssist_ReviewerRows_v8.csv"

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
tab_overview, tab_focus, tab_distributions, tab_fits, tab_sanity, tab_tables, tab_eic, tab_paper_timeline = st.tabs(
    ["Overview", "Your focus columns", "Explore distributions", "Heavy-tail fits", "Sanity checks", "Data tables", "EIC", "Paper timeline"]
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


# ==========================================================
# NEW TAB (DOES NOT TOUCH YOUR OLD CODE):
# EIC POV — “How many days a paper is in review?”
#
# Why this design (user-friendly + reliable):
# - Click-on-bars is flaky in Streamlit unless you add custom components.
# - So we use a RANGE SLIDER (acts like “click-to-filter” but always works),
#   plus a simple paper picker to drill down into reviewer status.
#
# Requirements: only pandas + plotly + streamlit (already in your app).
# Put this AFTER your data is loaded (paper_df, rev_df exist) and AFTER
# your existing tabs are created. Just add this as an extra tab.
# ==========================================================

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Helper: safe datetime parsing
# -----------------------------
def _to_dt(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# -----------------------------
# Helper: compute EIC-phase durations (days)
# -----------------------------
def _add_eic_durations(paper):
    paper = paper.copy()
    paper = _to_dt(
        paper,
        [
            "DatePaperSubmitted",
            "DateReviewersFullyAssigned",
            "DateFirstReviewReceived",
            "DateAllReviewsReceived",
            "AE_RecommendationDate",
            "EIC_DecisionDate",
            "DateDecisionLetterSent",
        ],
    )

    # “In review” (EIC POV): from “fully assigned” to “all reviews received”
    if "D_ReviewPhase" not in paper.columns:
        if "DateAllReviewsReceived" in paper.columns and "DateReviewersFullyAssigned" in paper.columns:
            paper["D_ReviewPhase"] = (paper["DateAllReviewsReceived"] - paper["DateReviewersFullyAssigned"]).dt.days

    # Optional extra phases (helpful context)
    if "D_AssignPhase" not in paper.columns:
        if "DateReviewersFullyAssigned" in paper.columns and "DatePaperSubmitted" in paper.columns:
            paper["D_AssignPhase"] = (paper["DateReviewersFullyAssigned"] - paper["DatePaperSubmitted"]).dt.days

    if "D_AEPhase" not in paper.columns:
        if "AE_RecommendationDate" in paper.columns and "DateAllReviewsReceived" in paper.columns:
            paper["D_AEPhase"] = (paper["AE_RecommendationDate"] - paper["DateAllReviewsReceived"]).dt.days

    if "D_EICPhase" not in paper.columns:
        if "EIC_DecisionDate" in paper.columns and "AE_RecommendationDate" in paper.columns:
            paper["D_EICPhase"] = (paper["EIC_DecisionDate"] - paper["AE_RecommendationDate"]).dt.days

    if "TotalTime_SubmissionToDecision_Days" not in paper.columns:
        if "DateDecisionLetterSent" in paper.columns and "DatePaperSubmitted" in paper.columns:
            paper["TotalTime_SubmissionToDecision_Days"] = (
                (paper["DateDecisionLetterSent"] - paper["DatePaperSubmitted"]).dt.days
            )

    return paper


# -----------------------------
# Helper: reviewer status (simple + readable)
# -----------------------------
def _add_reviewer_status(rev):
    rev = rev.copy()
    # ensure expected cols exist
    if "InviteOutcome" in rev.columns:
        rev["InviteOutcome"] = rev["InviteOutcome"].astype(str)
    else:
        rev["InviteOutcome"] = ""

    rev = _to_dt(rev, ["DateInvitationAccepted", "DateReviewSubmitted", "DateReviewDue", "DateReviewerInvited"])

    if "LateSubmissionFlag" in rev.columns:
        rev["LateSubmissionFlag"] = rev["LateSubmissionFlag"].astype(str)
    else:
        rev["LateSubmissionFlag"] = ""

    def status_row(r):
        if r["InviteOutcome"] != "accept":
            return r["InviteOutcome"]  # decline / no_response
        if pd.isna(r["DateReviewSubmitted"]):
            return "accepted_not_submitted"
        if r["LateSubmissionFlag"].lower() == "yes":
            return "submitted_late"
        return "submitted_on_time"

    rev["ReviewerStatus"] = rev.apply(status_row, axis=1)
    return rev


# ==========================================================
# Add a NEW TAB (keep your old tabs unchanged)
# ==========================================================
# If you already have tabs like: tab_overview, tab_focus, tab_distributions, ...
# just add one more tab name to your st.tabs([...]) list.
#
# Example:
# tab_overview, tab_focus, tab_distributions, tab_fits, tab_sanity, tab_tables, tab_eic = st.tabs([...,"EIC POV"])
#
# Below assumes you created `tab_eic` as the new tab object.
# ==========================================================

# --- NEW TAB CONTENT ---
with tab_eic:
    st.subheader("EIC POV: How many days a paper is in review")

    # Build clean working copies
    paper = _add_eic_durations(paper_df)
    rev = _add_reviewer_status(rev_df)

    # Pick the main metric
    metric = st.radio(
        "Metric",
        options=[
            "D_ReviewPhase",                   # EIC POV: in review
            "D_AssignPhase",                   # submit -> reviewers assigned
            "D_AEPhase",                       # all reviews -> AE rec
            "D_EICPhase",                      # AE rec -> EIC decision
            "TotalTime_SubmissionToDecision_Days",
        ],
        index=0,
        horizontal=True,
    )

    if metric not in paper.columns:
        st.error(f"Column '{metric}' not found / cannot be computed from available dates.")
        st.stop()

    # Clean values
    x = pd.to_numeric(paper[metric], errors="coerce")
    x = x[(x.notna()) & np.isfinite(x)]
    x = x[x >= 0]

    if len(x) == 0:
        st.info("No valid values for this metric (check missing dates).")
        st.stop()

    # Friendly range selector (reliable alternative to click-a-bar)
    p99 = int(np.nanpercentile(x, 99))
    cap = st.slider("Cap max days (for visualization)", min_value=14, max_value=max(30, p99), value=max(30, min(120, p99)))
    x_cap = x[x <= cap]

    # Bin width selector
    bin_width = st.selectbox("Bin width (days)", [3, 5, 7, 10, 14], index=2)

    # Range slider = “click-to-filter”
    low_default = 0
    high_default = min(21, cap)
    low_high = st.slider("Filter range (days)", min_value=0, max_value=int(cap), value=(low_default, high_default), step=1)
    low, high = int(low_high[0]), int(low_high[1])

    # Histogram
    fig = px.histogram(
        paper.assign(_metric=pd.to_numeric(paper[metric], errors="coerce")),
        x="_metric",
        nbins=max(5, int(cap / bin_width)),
        title=f"{metric} distribution (use the range slider to drill down)",
    )
    # Highlight selected range
    fig.add_vrect(x0=low, x1=high, opacity=0.15, line_width=0)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(range=[0, cap], title="Days")
    st.plotly_chart(fig, use_container_width=True)

    # Filtered subset of papers in selected range
    paper["_metric"] = pd.to_numeric(paper[metric], errors="coerce")
    subset = paper[(paper["_metric"].notna()) & (paper["_metric"] >= low) & (paper["_metric"] <= high)].copy()

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper-rounds in range", f"{len(subset):,}")
    c2.metric("Median days", f"{subset['_metric'].median():.0f}")
    c3.metric("P90 days", f"{subset['_metric'].quantile(0.90):.0f}")
    c4.metric("Max days", f"{subset['_metric'].max():.0f}")

    st.divider()

    # EIC backlog snapshot (helps EIC understand load)
    if "HandlingEIC_ID" in subset.columns:
        eic_summary = (
            subset.groupby("HandlingEIC_ID")["_metric"]
            .agg(count="count", median="median", p90=lambda s: s.quantile(0.90))
            .reset_index()
            .sort_values(["p90", "median", "count"], ascending=False)
        )
        st.markdown("#### EIC backlog summary (within selected range)")
        st.dataframe(eic_summary, use_container_width=True, height=220)

    st.markdown("#### Drilldown: pick a paper-round and see reviewer statuses")

    # Pick paper + round (simple & reliable)
    if "PaperID" not in subset.columns or "SubmissionRound" not in subset.columns:
        st.warning("Missing PaperID/SubmissionRound for drilldown.")
        st.stop()

    # Build a compact identifier list
    subset["paper_round_key"] = subset["PaperID"].astype(str) + " | round " + subset["SubmissionRound"].astype(int).astype(str)
    choices = subset["paper_round_key"].drop_duplicates().tolist()

    if len(choices) == 0:
        st.info("No paper-rounds found in this range.")
        st.stop()

    picked = st.selectbox("Select a paper-round", choices)
    picked_pid = picked.split("|")[0].strip()
    picked_round = int(picked.split("round")[1].strip())

    # Show paper row (timeline)
    show_cols = [
        "PaperID","SubmissionRound","JournalSection","PaperStatusOnSubmission",
        "HandlingAssociateEditorID","HandlingEIC_ID",
        "DatePaperSubmitted","DateReviewersFullyAssigned","DateAllReviewsReceived",
        "AE_RecommendationDate","AE_Recommendation",
        "EIC_DecisionDate","EIC_Decision",
        "DateDecisionLetterSent",
        "D_AssignPhase","D_ReviewPhase","D_AEPhase","D_EICPhase",
        "TotalTime_SubmissionToDecision_Days"
    ]
    show_cols = [c for c in show_cols if c in paper.columns]

    one = paper[(paper["PaperID"] == picked_pid) & (paper["SubmissionRound"] == picked_round)].copy()
    st.dataframe(one[show_cols], use_container_width=True)

    # Reviewer rows for that paper-round
    rr = rev[(rev["PaperID"] == picked_pid) & (rev["SubmissionRound"] == picked_round)].copy()
    if len(rr) == 0:
        st.info("No reviewer rows for this paper-round.")
        st.stop()

    # Status breakdown
    status_counts = rr["ReviewerStatus"].value_counts().reset_index()
    status_counts.columns = ["ReviewerStatus", "count"]

    fig2 = px.bar(status_counts, x="ReviewerStatus", y="count", title="Reviewer status breakdown")
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # Show reviewers table (EIC-friendly: who is blocking)
    rr_cols = [
        "ReviewerID","ReviewerType","ReviewerReliabilityTier","ReviewerWorkloadAtInvite",
        "InviteOutcome","DateReviewerInvited","DateInvitationAccepted","DateReviewDue","DateReviewSubmitted",
        "LateSubmissionFlag","NumRemindersSent","ReviewerStatus"
    ]
    rr_cols = [c for c in rr_cols if c in rr.columns]
    st.dataframe(rr[rr_cols].sort_values(["ReviewerStatus","InviteOutcome"], ascending=[True, True]),
                 use_container_width=True, height=420)

def _to_dt(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _abbr_decision(d: str) -> str:
    d = str(d).strip().lower()
    m = {
        "accept": "ACC",
        "minor revision": "MIN",
        "major revision": "MAJ",
        "submit as new": "SNEW",
        "reject": "REJ",
    }
    return m.get(d, "")


def _build_segments_and_markers(rr: pd.DataFrame, end_anchor: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      seg_df: bars for invite/review/overdue for accepted + decline
      noresp_df: markers for no_response only (single timestamp)
      decision_df: markers at review submission date (with decision/sentiment/etc.)
    """
    segments = []
    noresp = []
    decisions = []

    for _, r in rr.iterrows():
        rid = str(r.get("ReviewerID", ""))
        outcome = str(r.get("InviteOutcome", "")).strip()

        inv = r.get("DateReviewerInvited", pd.NaT)
        res = r.get("DateInvitationResolved", pd.NaT)
        acc = r.get("DateInvitationAccepted", pd.NaT)
        due = r.get("DateReviewDue", pd.NaT)
        sub = r.get("DateReviewSubmitted", pd.NaT)

        rating = str(r.get("ReviewerPaperRating", "")).strip()
        sent = r.get("ReviewSentiment_1to5", "")
        wc = r.get("ReviewLengthWords", "")
        rem = r.get("NumRemindersSent", "")

        if pd.isna(inv):
            continue

        # ---- NO RESPONSE: marker only (no finish date)
        if outcome == "no_response":
            noresp.append({
                "ReviewerID": rid,
                "At": inv,
                "Label": "NO RESPONSE",
            })
            continue

        # Resolve fallback (if missing)
        if pd.isna(res):
            res = inv + pd.Timedelta(days=7)

        # ---- DECLINE: can show a short bar from invite -> resolved
        if outcome == "decline":
            segments.append({
                "ReviewerID": rid,
                "Stage": "Declined",
                "Start": inv,
                "Finish": res,
            })
            continue

        # ---- ACCEPT path
        if pd.isna(acc):
            acc = res

        # Invite pending
        if acc >= inv:
            segments.append({
                "ReviewerID": rid,
                "Stage": "Invite pending",
                "Start": inv,
                "Finish": acc,
            })

        # If never submitted
        if pd.isna(sub):
            if not pd.isna(due) and due >= acc:
                # on-time window
                segments.append({
                    "ReviewerID": rid,
                    "Stage": "Review (on-time window)",
                    "Start": acc,
                    "Finish": due,
                })
                # overdue window until anchor (or due+14)
                end2 = min(end_anchor, due + pd.Timedelta(days=14)) if pd.notna(end_anchor) else due + pd.Timedelta(days=14)
                if end2 > due:
                    segments.append({
                        "ReviewerID": rid,
                        "Stage": "Overdue (no submission)",
                        "Start": due,
                        "Finish": end2,
                    })
            else:
                # no due -> review in progress until anchor
                if end_anchor > acc:
                    segments.append({
                        "ReviewerID": rid,
                        "Stage": "Review (in progress)",
                        "Start": acc,
                        "Finish": end_anchor,
                    })
            # no decision marker because no submission
            continue

        # Submitted: on-time vs overdue parts
        if not pd.isna(due) and due >= acc:
            end1 = min(due, sub)
            if end1 > acc:
                segments.append({
                    "ReviewerID": rid,
                    "Stage": "Review (on-time window)",
                    "Start": acc,
                    "Finish": end1,
                })
            if sub > due:
                segments.append({
                    "ReviewerID": rid,
                    "Stage": "Overdue",
                    "Start": due,
                    "Finish": sub,
                })
        else:
            if sub > acc:
                segments.append({
                    "ReviewerID": rid,
                    "Stage": "Review (in progress)",
                    "Start": acc,
                    "Finish": sub,
                })

        # Decision marker at submission date
        decisions.append({
            "ReviewerID": rid,
            "At": sub,
            "Decision": rating,
            "DecisionAbbr": _abbr_decision(rating),
            "Sentiment": sent,
            "Words": wc,
            "Reminders": rem,
        })

    seg_df = pd.DataFrame(segments)
    noresp_df = pd.DataFrame(noresp)
    decision_df = pd.DataFrame(decisions)
    return seg_df, noresp_df, decision_df




def _to_dt(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _rating_to_level(r):
    r = str(r).strip().lower()
    # 1(low) -> 5(high)
    m = {
        "reject": 1,
        "submit as new": 2,
        "major revision": 3,
        "minor revision": 4,
        "accept": 5,
    }
    return m.get(r, None)


def _abbr_rating(r):
    r = str(r).strip().lower()
    m = {
        "reject": "REJ",
        "submit as new": "SNEW",
        "major revision": "MAJ",
        "minor revision": "MIN",
        "accept": "ACC",
    }
    return m.get(r, "")


def _days_since(base, dt):
    if pd.isna(base) or pd.isna(dt):
        return np.nan
    return (dt - base).total_seconds() / 86400.0


#PAPER TIMELINE TAB ONLY (KEEP ONLY THIS BLOCK).
with tab_paper_timeline:
#IMPORTS.
    import numpy as np
#IMPORTS.
    import pandas as pd
#IMPORTS.
    import plotly.graph_objects as go
#IMPORTS.
    import streamlit as st
#TITLE.
    st.subheader("Paper timeline (reviewers + phases + reminders + decisions)")
#HELPER: datetime parse without mutating global dfs.
    def _pt_to_dt(df,cols):
#COPY.
        df=df.copy()
#PARSE.
        for c in cols:
#CHECK.
            if c in df.columns:
#CONVERT.
                df[c]=pd.to_datetime(df[c],errors="coerce")
#RETURN.
        return df
#HELPER: days since submission.
    def _pt_days(base,dt):
#MISSING.
        if pd.isna(base) or pd.isna(dt):
#NAN.
            return np.nan
#RETURN.
        return (dt-base).total_seconds()/86400.0
#HELPER: decision level.
    def _pt_level(dec):
#NORMALIZE.
        dec=str(dec).strip().lower()
#MAP.
        m={"reject":1,"submit as new":2,"major revision":3,"minor revision":4,"accept":5}
#RETURN.
        return m.get(dec,None)
#HELPER: decision abbr.
    def _pt_abbr(dec):
#NORMALIZE.
        dec=str(dec).strip().lower()
#MAP.
        m={"reject":"REJ","submit as new":"SNEW","major revision":"MAJ","minor revision":"MIN","accept":"ACC"}
#RETURN.
        return m.get(dec,"")
#USE FILTERED DFS IF PRESENT, ELSE FALLBACK.
    try:
#SOURCE.
        _paper_src=paper_f
#FALLBACK.
    except NameError:
#SOURCE.
        _paper_src=paper_df
#USE FILTERED DFS IF PRESENT, ELSE FALLBACK.
    try:
#SOURCE.
        _rev_src=rev_f
#FALLBACK.
    except NameError:
#SOURCE.
        _rev_src=rev_df
#PARSE PAPER DATES.
    paper=_pt_to_dt(_paper_src,["DatePaperSubmitted","DateReviewersFullyAssigned","DateAllReviewsReceived","AE_RecommendationDate","EIC_DecisionDate","DateDecisionLetterSent"])
#PARSE REVIEWER DATES.
    rev=_pt_to_dt(_rev_src,["DateReviewerInvited","DateInvitationAccepted","DateInvitationResolved","DateNoResponseCensor","DateReviewDue","DateReviewSubmitted"])
#GUARD PAPER COLS.
    if ("PaperID" not in paper.columns) or ("SubmissionRound" not in paper.columns) or ("DatePaperSubmitted" not in paper.columns):
#ERROR.
        st.error("PaperHeader must contain PaperID, SubmissionRound, DatePaperSubmitted.")
#STOP.
        st.stop()
#GUARD REVIEWER COLS.
    if ("PaperID" not in rev.columns) or ("SubmissionRound" not in rev.columns) or ("ReviewerID" not in rev.columns) or ("InviteOutcome" not in rev.columns) or ("DateReviewerInvited" not in rev.columns):
#ERROR.
        st.error("ReviewerRows must contain PaperID, SubmissionRound, ReviewerID, InviteOutcome, DateReviewerInvited.")
#STOP.
        st.stop()
#BUILD KEY.
    paper=paper.copy()
#KEY.
    paper["paper_round_key"]=paper["PaperID"].astype(str)+" | round "+paper["SubmissionRound"].astype(int).astype(str)
#GUARD EMPTY.
    if paper.empty:
#INFO.
        st.info("No papers match current filters. Relax filters to view a timeline.")
#STOP.
        st.stop()
#SELECT.
    picked=st.selectbox("Select a paper-round",paper["paper_round_key"].drop_duplicates().tolist())
#PID.
    pid=picked.split("|")[0].strip()
#ROUND.
    rnd=int(picked.split("round")[1].strip())
#GET PAPER ROW.
    one=paper[(paper["PaperID"]==pid)&(paper["SubmissionRound"].astype(int)==rnd)].copy()
#GUARD.
    if one.empty:
#WARN.
        st.warning("Paper-round not found after filtering.")
#STOP.
        st.stop()
#ROW.
    p=one.iloc[0]
#BASE.
    base_date=p["DatePaperSubmitted"]
#GUARD.
    if pd.isna(base_date):
#ERROR.
        st.error("Selected paper-round has no DatePaperSubmitted.")
#STOP.
        st.stop()
#CAPTION.
    st.caption(f"Base date (submission): {base_date.date().isoformat()} — x-axis shows calendar dates every 21 days")
#SHOW PAPER MOMENTS (DATES ONLY).
    show_cols=["PaperID","SubmissionRound","JournalSection","PaperStatusOnSubmission","DatePaperSubmitted","DateReviewersFullyAssigned","DateAllReviewsReceived","AE_RecommendationDate","EIC_DecisionDate","DateDecisionLetterSent","TotalTime_SubmissionToDecision_Days"]
#FILTER.
    show_cols=[c for c in show_cols if c in one.columns]
#DISPLAY.
    st.dataframe(one[show_cols],use_container_width=True)
#GET REVIEWERS.
    rr=rev[(rev["PaperID"]==pid)&(rev["SubmissionRound"].astype(int)==rnd)].copy()
#GUARD.
    if rr.empty:
#INFO.
        st.info("No reviewer rows for this paper-round.")
#STOP.
        st.stop()
#SORT.
    rr=rr.sort_values(["InviteOutcome","ReviewerID"],ascending=[True,True]).reset_index(drop=True)
#REVIEWER ORDER.
    reviewers=rr["ReviewerID"].astype(str).tolist()
#PAPER LINES (DAYS).
    ae_line=_pt_days(base_date,p.get("AE_RecommendationDate",pd.NaT))
#PAPER LINES (DAYS).
    eic_line=_pt_days(base_date,p.get("EIC_DecisionDate",pd.NaT))
#PAPER LINES (DAYS).
    letter_line=_pt_days(base_date,p.get("DateDecisionLetterSent",pd.NaT))
#END ANCHOR.
    end_anchor=letter_line
#FALLBACK.
    if np.isnan(end_anchor):
#FALLBACK.
        end_anchor=eic_line
#FALLBACK.
    if np.isnan(end_anchor):
#FALLBACK.
        end_anchor=_pt_days(base_date,p.get("DateAllReviewsReceived",pd.NaT))
#FALLBACK.
    if np.isnan(end_anchor):
#DEFAULT.
        end_anchor=60.0
#COLORS.
    stage_colors={"Invite phase":"#4C78A8","Review phase":"#54A24B","Review submitted phase":"#F58518","Declined":"#9D9D9D"}
#STAGE DATA.
    stage_data={"Invite phase":[],"Review phase":[],"Review submitted phase":[],"Declined":[]}
#REMINDERS.
    rem_x=[]
#REMINDERS.
    rem_y=[]
#NO RESP.
    nr_x=[]
#NO RESP.
    nr_y=[]
#NO RESP.
    nr_hover=[]
#DECISIONS.
    dec_x=[]
#DECISIONS.
    dec_y=[]
#DECISIONS.
    dec_size=[]
#DECISIONS.
    dec_text=[]
#DECISIONS.
    dec_hover=[]
#REMINDER POLICY.
    REM1=21
#REMINDER POLICY.
    REM2=42
#XMAX.
    xmax=0.0
#BUILD.
    for _,r in rr.iterrows():
#RID.
        rid=str(r.get("ReviewerID",""))
#OUTCOME.
        outcome=str(r.get("InviteOutcome","")).strip()
#RELATIVE.
        inv=_pt_days(base_date,r.get("DateReviewerInvited",pd.NaT))
#RELATIVE.
        acc=_pt_days(base_date,r.get("DateInvitationAccepted",pd.NaT))
#RELATIVE.
        res=_pt_days(base_date,r.get("DateInvitationResolved",pd.NaT))
#RELATIVE.
        censor=_pt_days(base_date,r.get("DateNoResponseCensor",pd.NaT))
#RELATIVE.
        due=_pt_days(base_date,r.get("DateReviewDue",pd.NaT))
#RELATIVE.
        sub=_pt_days(base_date,r.get("DateReviewSubmitted",pd.NaT))
#NO RESPONSE (MARKER ONLY).
        if outcome=="no_response":
#CHECK.
            if not np.isnan(inv):
#ADD.
                nr_x.append(inv)
#ADD.
                nr_y.append(rid)
#HOVER.
                nr_hover.append(f"Reviewer={rid}<br>Status=NO RESPONSE<br>Invite day={inv:.1f}<br>Censor day={(censor if not np.isnan(censor) else 'NA')}")
#XMAX.
                xmax=max(xmax,inv)
#NEXT.
            continue
#DECLINE (INVITE->RESOLVED).
        if outcome=="decline":
#FALLBACK.
            if np.isnan(res) and not np.isnan(inv):
#SET.
                res=inv+2.0
#BAR.
            if not np.isnan(inv) and not np.isnan(res) and res>inv:
#ADD.
                stage_data["Declined"].append((rid,inv,res-inv))
#XMAX.
                xmax=max(xmax,res)
#NEXT.
            continue
#ACCEPT (PHASES).
        if outcome=="accept":
#FALLBACK.
            if np.isnan(acc) and not np.isnan(res):
#SET.
                acc=res
#FALLBACK.
            if np.isnan(acc) and not np.isnan(inv):
#SET.
                acc=inv+1.0
#INVITE PHASE.
            if not np.isnan(inv) and not np.isnan(acc) and acc>inv:
#ADD.
                stage_data["Invite phase"].append((rid,inv,acc-inv))
#XMAX.
                xmax=max(xmax,acc)
#REVIEW END.
            if not np.isnan(sub):
#SET.
                review_end=sub
#ELIF.
            elif not np.isnan(due):
#SET.
                review_end=min(due,end_anchor)
#ELSE.
            else:
#SET.
                review_end=end_anchor
#REVIEW PHASE.
            if not np.isnan(acc) and review_end>acc:
#ADD.
                stage_data["Review phase"].append((rid,acc,review_end-acc))
#XMAX.
                xmax=max(xmax,review_end)
#REM COUNT.
            n_rem=r.get("NumRemindersSent",0)
#CAST.
            try:
#CAST.
                n_rem=int(n_rem)
#EXCEPT.
            except Exception:
#ZERO.
                n_rem=0
#REM1.
            if n_rem>0 and not np.isnan(acc):
#X.
                x1=acc+REM1
#WITHIN.
                if x1<=review_end:
#ADD.
                    rem_x.append(x1)
#ADD.
                    rem_y.append(rid)
#REM2.
            if n_rem>1 and not np.isnan(acc):
#X.
                x2=acc+REM2
#WITHIN.
                if x2<=review_end:
#ADD.
                    rem_x.append(x2)
#ADD.
                    rem_y.append(rid)
#ORANGE PHASE.
            if not np.isnan(sub):
#END.
                orange_end=eic_line
#FALLBACK.
                if np.isnan(orange_end):
#FALLBACK.
                    orange_end=end_anchor
#ADD.
                if not np.isnan(orange_end) and orange_end>sub:
#ADD.
                    stage_data["Review submitted phase"].append((rid,sub,orange_end-sub))
#XMAX.
                    xmax=max(xmax,orange_end)
#DECISION MARKER.
                lvl=_pt_level(r.get("ReviewerPaperRating",""))
#CHECK.
                if lvl is not None:
#SIZE.
                    size=10+(lvl*4)
#ADD.
                    dec_x.append(sub)
#ADD.
                    dec_y.append(rid)
#ADD.
                    dec_size.append(size)
#ADD.
                    dec_text.append(_pt_abbr(r.get("ReviewerPaperRating","")))
#HOVER.
                    dec_hover.append(f"Reviewer={rid}<br>Submitted day={sub:.1f}<br>Decision={r.get('ReviewerPaperRating','')}<br>Sentiment={r.get('ReviewSentiment_1to5','')}<br>Words={r.get('ReviewLengthWords','')}")
#XMAX.
                    xmax=max(xmax,sub)
#NEXT.
            continue
#XMAX.
    xmax=max(xmax,end_anchor,0.0)
#GUARD.
    if sum(len(v) for v in stage_data.values())==0:
#WARN.
        st.warning("No phase bars were generated. Check missing dates in reviewer rows.")
#SHOW.
        st.dataframe(rr[["ReviewerID","InviteOutcome","DateReviewerInvited","DateInvitationAccepted","DateInvitationResolved","DateReviewSubmitted","DateNoResponseCensor"]].head(80),use_container_width=True)
#STOP.
        st.stop()
#FIGURE.
    fig=go.Figure()
#ADD STAGES.
    for stage in ["Invite phase","Review phase","Review submitted phase","Declined"]:
#ROWS.
        rows=stage_data.get(stage,[])
#SKIP.
        if not rows:
#NEXT.
            continue
#Y.
        y=[t[0] for t in rows]
#BASE.
        base=[t[1] for t in rows]
#LEN.
        x=[t[2] for t in rows]
#BAR.
        fig.add_trace(go.Bar(x=x,y=y,base=base,orientation="h",name=stage if stage!="Declined" else "Invite phase (declined)",marker=dict(color=stage_colors.get(stage,"#999999")),opacity=0.95,hovertemplate="Reviewer=%{y}<br>Start day=%{base:.1f}<br>Duration=%{x:.1f}<extra></extra>"))
#REMINDERS.
    if len(rem_x)>0:
#ADD.
        fig.add_trace(go.Scatter(x=rem_x,y=rem_y,mode="markers",name="Reminder",marker=dict(symbol="line-ns-open",size=18,color="#FFD166",line=dict(width=2,color="#FFD166")),hovertemplate="Reviewer=%{y}<br>Reminder date=%{x:.1f} days<extra></extra>"))
#NO RESP.
    if len(nr_x)>0:
#ADD.
        fig.add_trace(go.Scatter(x=nr_x,y=nr_y,mode="markers+text",text=["NO RESP"]*len(nr_x),textposition="middle right",name="No response",marker=dict(symbol="x",size=10,color="#B279A2"),hovertext=nr_hover,hovertemplate="%{hovertext}<extra></extra>"))
#DECISIONS.
    if len(dec_x)>0:
#ADD.
        fig.add_trace(go.Scatter(x=dec_x,y=dec_y,mode="markers+text",text=dec_text,textposition="middle right",name="Reviewer decision",marker=dict(symbol="line-ns-open",size=dec_size,color="black",line=dict(width=3,color="black")),hovertext=dec_hover,hovertemplate="%{hovertext}<extra></extra>"))
#SUBMISSION LINE.
    fig.add_vline(x=0.0,line_width=1,line_dash="solid")
#ANNOTATE.
    fig.add_annotation(x=0.0,y=1.02,xref="x",yref="paper",text="Submitted",showarrow=False,font=dict(size=10))
#AE LINE.
    if not np.isnan(ae_line):
#ADD.
        fig.add_vline(x=float(ae_line),line_width=1,line_dash="dot")
#ANNOTATE.
        fig.add_annotation(x=float(ae_line),y=1.02,xref="x",yref="paper",text="AE rec",showarrow=False,font=dict(size=10))
#EIC LINE.
    if not np.isnan(eic_line):
#ADD.
        fig.add_vline(x=float(eic_line),line_width=1,line_dash="solid")
#ANNOTATE.
        fig.add_annotation(x=float(eic_line),y=1.02,xref="x",yref="paper",text="EIC decision",showarrow=False,font=dict(size=10))
#LAYOUT.
    fig.update_layout(barmode="overlay",bargap=0.70,title="Reviewer phases (slimmer bars) — x = calendar dates (21-day ticks)",height=max(520,160+len(reviewers)*38),margin=dict(l=10,r=10,t=60,b=65),legend_title_text="Phase")
#TICKS: every 21 days (display as DD Mon YYYY).
    tick_step=21.0
#RANGE.
    x_max_plot=max(5.0,float(xmax)+5.0)
#TICK VALUES.
    tickvals=list(np.arange(0.0,x_max_plot+0.0001,tick_step))
#TICK LABELS AS DATES.
    ticktext=[(base_date+pd.Timedelta(days=float(v))).strftime("%d %b %Y") for v in tickvals]
#APPLY AXIS (LINEAR BUT DATE LABELS).
    fig.update_xaxes(type="linear",title="Date",range=[0.0,x_max_plot],tickmode="array",tickvals=tickvals,ticktext=ticktext,tickangle=45)
#Y ORDER.
    fig.update_yaxes(title="Reviewer",categoryorder="array",categoryarray=reviewers[::-1])
#SHOW.
    st.plotly_chart(fig,use_container_width=True)
#DETAILS.
    st.markdown("#### Reviewer details (durations + decision)")
#COPY.
    rr2=rr.copy()
#DUR.
    rr2["InviteToAccept_days"]=(rr2["DateInvitationAccepted"]-rr2["DateReviewerInvited"]).dt.days
#DUR.
    rr2["AcceptToSubmit_days"]=(rr2["DateReviewSubmitted"]-rr2["DateInvitationAccepted"]).dt.days
#DUR.
    rr2["Overdue_days"]=(rr2["DateReviewSubmitted"]-rr2["DateReviewDue"]).dt.days
#COLS.
    cols=["ReviewerID","InviteOutcome","ReviewerType","ReviewerReliabilityTier","ReviewerWorkloadAtInvite","DateReviewerInvited","DateInvitationAccepted","DateReviewDue","DateReviewSubmitted","NumRemindersSent","InviteToAccept_days","AcceptToSubmit_days","Overdue_days","ReviewSentiment_1to5","ReviewerPaperRating","ReviewLengthWords","DateNoResponseCensor"]
#FILTER.
    cols=[c for c in cols if c in rr2.columns]
#DISPLAY (THIS MUST BE THE LAST LINE IN THIS TAB BLOCK).
    st.dataframe(rr2[cols],use_container_width=True,height=420)