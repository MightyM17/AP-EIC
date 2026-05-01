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

default_xlsx = "EIC-AIssist_peer_review_synth_v12.xlsx"
default_paper_csv = "EIC-AIssist_PaperHeader_v12.csv"
default_reviewer_csv = "EIC-AIssist_ReviewerRows_v12.csv"

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
# tab_overview, tab_focus, tab_distributions, tab_fits, tab_sanity, tab_tables, tab_eic, tab_paper_timeline, tab_reviewer_timeline, tab_ae_timeline = st.tabs(
#     ["Overview", "Your focus columns", "Explore distributions", "Heavy-tail fits", "Sanity checks", "Data tables", "EIC", "Paper timeline", "Reviewer timeline", "AE timeline"]
# )


nav_options = [
    "Overview",
    "Your focus columns",
    "Explore distributions",
    "Heavy-tail fits",
    "Sanity checks",
    "Data tables",
    "EIC",
    "Paper timeline",
    "Reviewer timeline",
    "AE timeline",
    "Paper status overview"
]

# ✅ ONLY override when coming from AE click
if "active_tab" in st.session_state and st.session_state.get("from_ae_click", False):
    st.session_state["nav_control"] = st.session_state["active_tab"]
    st.session_state["from_ae_click"] = False  # reset

selected_tab = st.segmented_control(
    "Navigation",
    nav_options,
    key="nav_control"
)

st.session_state["active_tab"] = selected_tab

# -----------------------------
# Overview tab
# -----------------------------
# with tab_overview:
if selected_tab == "Overview":
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
        st.plotly_chart(fig, width='stretch')
    with colB:
        st.caption("ReviewerRows missingness (%)")
        fig = px.bar(miss_r.head(20), x="missing_pct", y="column", orientation="h", hover_data=["missing_count", "dtype"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width='stretch')

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
        st.dataframe(stats_df, width='stretch')


# -----------------------------
# Your requested focus columns
# -----------------------------
if selected_tab == "Your focus columns":
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
            st.plotly_chart(fig, width='stretch')
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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')
        with right:
            fig = px.box(
                rev_f,
                y=workload_col,
                points="outliers",
                title=f"{workload_col} box (outliers visible)",
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width='stretch')
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

            st.plotly_chart(fig, width='stretch')

            fig.update_layout(height=420, xaxis_title="InviteOutcome", yaxis_title="Count", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width='stretch')
        with right:
            # acceptance rate vs workload (binned)
            if workload_col:
                tmp = rev_f[[workload_col, outcome_col]].dropna()
                if len(tmp) > 0:
                    tmp["workload_bin"] = pd.cut(tmp[workload_col], bins=10, duplicates="drop").astype(str)
                    acc = tmp.groupby("workload_bin")[outcome_col].apply(lambda s: (s.astype(str) == "accept").mean()).reset_index(name="accept_rate")
                    fig = px.bar(acc, x="workload_bin", y="accept_rate", title="Acceptance rate by workload bin")
                    fig.update_layout(height=420, yaxis=dict(range=[0, 1]), margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, width='stretch')
    else:
        st.warning("InviteOutcome column not found.")


# -----------------------------
# Explore distributions (any column)
# -----------------------------
if selected_tab == "Explore distributions":
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
            st.plotly_chart(fig, width='stretch')

            # ECDF
            fig2 = px.ecdf(df, x=col_sel, title=f"ECDF: {col_sel}")
            if log_x:
                fig2.update_xaxes(type="log")
            fig2.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, width='stretch')

        elif col_type == "categorical":
            top_k = st.slider("Top-K categories", 5, 60, 20, 1)
            vc = df[col_sel].astype(str).value_counts(dropna=False).head(top_k).reset_index()
            vc.columns = ["category", "count"]
            fig = px.bar(vc, x="category", y="count", title=f"Top {top_k}: {col_sel}")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width='stretch')

        else:
            # datetime: counts by week/month
            gran = st.selectbox("Time granularity", ["D", "W", "M"], index=1)
            tmp = df[[col_sel]].dropna().copy()
            tmp["bucket"] = tmp[col_sel].dt.to_period(gran).dt.to_timestamp()
            agg = tmp.groupby("bucket").size().reset_index(name="count")
            fig = px.line(agg, x="bucket", y="count", title=f"Event volume over time: {col_sel} ({gran})")
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width='stretch')

        # Descriptive stats
        st.subheader("Quick stats")
        st.write(df[[col_sel]].describe(include="all").T)


# -----------------------------
# Heavy-tail fits (lognormal/gamma/weibull + Q-Q)
# -----------------------------
if selected_tab == "Heavy-tail fits":
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
        st.plotly_chart(fig, width='stretch')

        qq = plot_qq(df_fit[fit_col], dist_name=dist_name, title=f"Q-Q plot: {fit_col} vs {dist_name}")
        if qq is not None:
            st.plotly_chart(qq, width='stretch')

        st.subheader("Outliers (top 20)")
        s = df_fit[fit_col].dropna()
        if len(s) > 0:
            top = df_fit.loc[s.sort_values(ascending=False).head(20).index]
            st.dataframe(top, width='stretch')


# -----------------------------
# Sanity checks (timeline constraints + logical consistency)
# -----------------------------
if selected_tab == "Sanity checks":
    st.subheader("Timeline monotonicity checks (PaperHeader)")
    vio = workflow_violations(paper_f)
    if len(vio) == 0:
        st.info("Not enough timeline columns found to run monotonicity checks.")
    else:
        st.dataframe(vio, width='stretch')
        fig = px.bar(vio, x="violations", y="rule", orientation="h", title="Violation counts by rule")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width='stretch')

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
        st.dataframe(chk_df, width='stretch')


# -----------------------------
# Data tables
# -----------------------------
if selected_tab == "Data tables":
    st.subheader("Filtered data preview")

    left, right = st.columns(2)
    with left:
        st.caption("PaperHeader (filtered)")
        st.dataframe(paper_f.head(200), width='stretch', height=420)
    with right:
        st.caption("ReviewerRows (filtered)")
        st.dataframe(rev_f.head(200), width='stretch', height=420)

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
if selected_tab == "EIC":
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
    st.plotly_chart(fig, width='stretch')

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
        st.dataframe(eic_summary, width='stretch', height=220)

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
    st.dataframe(one[show_cols], width='stretch')

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
    st.plotly_chart(fig2, width='stretch')

    # Show reviewers table (EIC-friendly: who is blocking)
    rr_cols = [
        "ReviewerID","ReviewerType","ReviewerReliabilityTier","ReviewerWorkloadAtInvite",
        "InviteOutcome","DateReviewerInvited","DateInvitationAccepted","DateReviewDue","DateReviewSubmitted",
        "LateSubmissionFlag","NumRemindersSent","ReviewerStatus"
    ]
    rr_cols = [c for c in rr_cols if c in rr.columns]
    st.dataframe(rr[rr_cols].sort_values(["ReviewerStatus","InviteOutcome"], ascending=[True, True]),
                 width='stretch', height=420)

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


#PAPER TIMELINE TAB ONLY.
if selected_tab == "Paper timeline":
#LOCAL imports.
    import numpy as np
#LOCAL imports.
    import pandas as pd
#LOCAL imports.
    import plotly.graph_objects as go
#LOCAL imports.
    import streamlit as st

#TITLE.
    st.subheader("Paper timeline (reviewers + phases + reminders + decisions)")

#HELPER: safe datetime parsing.
    def _pt_to_dt(df, cols):
#COPY.
        df = df.copy()
#PARSE requested cols.
        for c in cols:
#CHECK exists.
            if c in df.columns:
#CONVERT.
                df[c] = pd.to_datetime(df[c], errors="coerce")
#RETURN parsed copy.
        return df

#HELPER: numeric days since paper submission.
    def _pt_days(base, dt):
#MISSING guard.
        if pd.isna(base) or pd.isna(dt):
#RETURN nan.
            return np.nan
#RETURN float days.
        return (dt - base).total_seconds() / 86400.0

#HELPER: decision level for visual marker size.
    def _pt_level(dec):
#NORMALIZE.
        dec = str(dec).strip().lower()
#MAP levels.
        m = {"reject": 1, "submit as new": 2, "major revision": 3, "minor revision": 4, "accept": 5}
#RETURN mapped level.
        return m.get(dec, None)

#HELPER: short reviewer decision label.
    def _pt_abbr(dec):
#NORMALIZE.
        dec = str(dec).strip().lower()
#MAP abbreviations.
        m = {"reject": "REJ", "submit as new": "SNEW", "major revision": "MAJ", "minor revision": "MIN", "accept": "ACC"}
#RETURN text.
        return m.get(dec, "")

#USE filtered paper df if available.
    try:
#SOURCE.
        _paper_src = paper_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _paper_src = paper_df

#USE filtered reviewer df if available.
    try:
#SOURCE.
        _rev_src = rev_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _rev_src = rev_df

#PARSE paper dates.
    paper = _pt_to_dt(
        _paper_src,
        [
            "DatePaperSubmitted",
            "DateReviewersFullyAssigned",
            "DateAllReviewsReceived",
            "AE_RecommendationDate",
            "EIC_DecisionDate",
            "DateDecisionLetterSent",
        ],
    )

#PARSE reviewer dates.
    rev = _pt_to_dt(
        _rev_src,
        [
            "DateReviewerInvited",
            "DateInvitationAccepted",
            "DateInvitationResolved",
            "DateNoResponseCensor",
            "DateNoResponseTerminal",
            "DateReviewDue",
            "DateReviewSubmitted",
            "AE_RecommendationDateAtEnd",
            "EIC_DecisionDateAtEnd",
            "DateDecisionLetterSentAtEnd",
        ],
    )

#GUARD required paper cols.
    if ("PaperID" not in paper.columns) or ("SubmissionRound" not in paper.columns) or ("DatePaperSubmitted" not in paper.columns):
#ERROR.
        st.error("PaperHeader must contain PaperID, SubmissionRound, DatePaperSubmitted.")
#STOP.
        st.stop()

#GUARD required reviewer cols.
    if ("PaperID" not in rev.columns) or ("SubmissionRound" not in rev.columns) or ("ReviewerID" not in rev.columns) or ("InviteOutcome" not in rev.columns) or ("DateReviewerInvited" not in rev.columns):
#ERROR.
        st.error("ReviewerRows must contain PaperID, SubmissionRound, ReviewerID, InviteOutcome, DateReviewerInvited.")
#STOP.
        st.stop()

#BUILD selector key.
    paper = paper.copy()
#DISPLAY key.
    paper["paper_round_key"] = paper["PaperID"].astype(str) + " | round " + paper["SubmissionRound"].astype(int).astype(str)

#EMPTY guard.
    if paper.empty:
#INFO.
        st.info("No papers match current filters. Relax filters to view a timeline.")
#STOP.
        st.stop()

    default_val = st.session_state.get("selected_paper_from_ae", None)

    options = paper["paper_round_key"].drop_duplicates().tolist()

    idx = options.index(default_val) if default_val in options else 0

    picked = st.selectbox(
        "Select a paper-round",
        options,
        index=idx,
        key="paper_select"
    )

#PARSE paper id.
    pid = picked.split("|")[0].strip()
#PARSE round.
    rnd = int(picked.split("round")[1].strip())

#GET selected paper row.
    one = paper[(paper["PaperID"] == pid) & (paper["SubmissionRound"].astype(int) == rnd)].copy()

#GUARD not found.
    if one.empty:
#WARN.
        st.warning("Paper-round not found after filtering.")
#STOP.
        st.stop()

#SELECT row.
    p = one.iloc[0]

#BASE date.
    base_date = p["DatePaperSubmitted"]

#GUARD base date.
    if pd.isna(base_date):
#ERROR.
        st.error("Selected paper-round has no DatePaperSubmitted.")
#STOP.
        st.stop()

#CAPTION.
    st.caption(f"Base date (submission): {base_date.date().isoformat()} — x-axis shows calendar dates every 21 days")

#SHOW paper timing fields.
    show_cols = [
        "PaperID",
        "SubmissionRound",
        "JournalSection",
        "PaperStatusOnSubmission",
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "AE_Recommendation",
        "EIC_DecisionDate",
        "EIC_Decision",
        "DateDecisionLetterSent",
        "FinalDecisionOutcome",
        "TotalTime_SubmissionToDecision_Days",
    ]

#KEEP existing cols only.
    show_cols = [c for c in show_cols if c in one.columns]

#DISPLAY paper summary.
    st.dataframe(one[show_cols], width='stretch')

#GET reviewer rows.
    rr = rev[(rev["PaperID"] == pid) & (rev["SubmissionRound"].astype(int) == rnd)].copy()

#GUARD reviewer rows.
    if rr.empty:
#INFO.
        st.info("No reviewer rows for this paper-round.")
#STOP.
        st.stop()

#SORT reviewers.
    rr = rr.sort_values(["InviteOutcome", "ReviewerID"], ascending=[True, True]).reset_index(drop=True)

#Y order.
    reviewers = rr["ReviewerID"].astype(str).tolist()

#PAPER-level relative dates.
    ae_line = _pt_days(base_date, p.get("AE_RecommendationDate", pd.NaT))
#PAPER-level relative dates.
    eic_line = _pt_days(base_date, p.get("EIC_DecisionDate", pd.NaT))
#PAPER-level relative dates.
    letter_line = _pt_days(base_date, p.get("DateDecisionLetterSent", pd.NaT))

#END anchor for open phases.
    end_anchor = letter_line
#FALLBACK.
    if np.isnan(end_anchor):
#SET fallback.
        end_anchor = eic_line
#FALLBACK.
    if np.isnan(end_anchor):
#SET fallback.
        end_anchor = _pt_days(base_date, p.get("DateAllReviewsReceived", pd.NaT))
#FINAL fallback.
    if np.isnan(end_anchor):
#DEFAULT.
        end_anchor = 60.0

#PHASE colors.
    stage_colors = {
        "Invite phase": "#4C78A8",
        "Review phase": "#54A24B",
        "AE to EIC phase": "#B279A2",
        "Declined": "#9D9D9D",
    }

#COLLECT phase bars with custom hover.
    stage_data = {"Invite phase": [], "Review phase": [], "AE to EIC phase": [], "Declined": []}

#COLLECT reminder markers.
    rem_x = []
#COLLECT reminder markers.
    rem_y = []

#COLLECT decision markers.
    dec_x = []
#COLLECT decision markers.
    dec_y = []
#COLLECT decision markers.
    dec_size = []
#COLLECT decision markers.
    dec_text = []
#COLLECT decision markers.
    dec_hover = []

#REMINDER policy offsets.
    REM1 = 21
#REMINDER policy offsets.
    REM2 = 42

#TRACK x max.
    xmax = 0.0

#BUILD timeline row by row.
    for _, r in rr.iterrows():
#REVIEWER id.
        rid = str(r.get("ReviewerID", ""))
#INVITE outcome.
        outcome = str(r.get("InviteOutcome", "")).strip()

#RELATIVE date positions.
        inv = _pt_days(base_date, r.get("DateReviewerInvited", pd.NaT))
#RELATIVE date positions.
        acc = _pt_days(base_date, r.get("DateInvitationAccepted", pd.NaT))
#RELATIVE date positions.
        res = _pt_days(base_date, r.get("DateInvitationResolved", pd.NaT))
#RELATIVE date positions.
        censor = _pt_days(base_date, r.get("DateNoResponseCensor", pd.NaT))
#RELATIVE date positions.
        terminal = _pt_days(base_date, r.get("DateNoResponseTerminal", pd.NaT))
#RELATIVE date positions.
        due = _pt_days(base_date, r.get("DateReviewDue", pd.NaT))
#RELATIVE date positions.
        sub = _pt_days(base_date, r.get("DateReviewSubmitted", pd.NaT))

#NO RESPONSE: extend invite phase until terminal event.
        if outcome == "no_response":
#GET terminal outcome.
            terminal_outcome = str(r.get("NoResponseTerminalOutcome", "")).strip()
#FALLBACK terminal.
            if np.isnan(terminal):
#USE censor if available.
                terminal = censor
#FINAL fallback.
            if np.isnan(terminal):
#USE anchor.
                terminal = end_anchor
#DRAW only if invite exists.
            if not np.isnan(inv):
#ENSURE positive bar length.
                bar_end = max(inv + 0.5, terminal)
#HOVER text.
                hover_txt = (
                    f"Reviewer={rid}<br>"
                    f"Status=no_response<br>"
                    f"Invite sent={r.get('DateReviewerInvited', '')}<br>"
                    f"Terminal outcome={terminal_outcome if terminal_outcome else 'NA'}<br>"
                    f"Terminal date={r.get('DateNoResponseTerminal', '') if pd.notna(r.get('DateNoResponseTerminal', pd.NaT)) else 'NA'}<br>"
                    f"AE suggestion at end={r.get('AE_RecommendationAtEnd', 'NA')}<br>"
                    f"EIC decision at end={r.get('EIC_DecisionAtEnd', 'NA')}"
                )
#APPEND as invite phase.
                stage_data["Invite phase"].append({"rid": rid, "start": inv, "dur": bar_end - inv, "hover": hover_txt})
#UPDATE xmax.
                xmax = max(xmax, bar_end)
#DONE.
            continue

#DECLINE: invite to resolved.
        if outcome == "decline":
#FALLBACK resolved date if missing.
            if np.isnan(res) and not np.isnan(inv):
#SMALL fallback.
                res = inv + 2.0
#DRAW if valid.
            if not np.isnan(inv) and not np.isnan(res) and res > inv:
#HOVER text.
                hover_txt = (
                    f"Reviewer={rid}<br>"
                    f"Status=decline<br>"
                    f"Invite sent={r.get('DateReviewerInvited', '')}<br>"
                    f"Declined/Resolved={r.get('DateInvitationResolved', '')}<br>"
                    f"AE suggestion at end={r.get('AE_RecommendationAtEnd', 'NA')}<br>"
                    f"EIC decision at end={r.get('EIC_DecisionAtEnd', 'NA')}"
                )
#APPEND decline phase.
                stage_data["Declined"].append({"rid": rid, "start": inv, "dur": res - inv, "hover": hover_txt})
#UPDATE xmax.
                xmax = max(xmax, res)
#DONE.
            continue

#ACCEPT path.
        if outcome == "accept":
#FALLBACK accept to resolved.
            if np.isnan(acc) and not np.isnan(res):
#SET fallback.
                acc = res
#FINAL fallback.
            if np.isnan(acc) and not np.isnan(inv):
#SET fallback.
                acc = inv + 1.0

#INVITE phase.
            if not np.isnan(inv) and not np.isnan(acc) and acc > inv:
#HOVER text.
                hover_txt = (
                    f"Reviewer={rid}<br>"
                    f"Status=accept<br>"
                    f"Invite sent={r.get('DateReviewerInvited', '')}<br>"
                    f"Accepted={r.get('DateInvitationAccepted', '')}"
                )
#APPEND invite phase.
                stage_data["Invite phase"].append({"rid": rid, "start": inv, "dur": acc - inv, "hover": hover_txt})
#UPDATE xmax.
                xmax = max(xmax, acc)

#REVIEW phase end.
            if not np.isnan(sub):
#SET review end.
                review_end = sub
#ELIF due exists.
            elif not np.isnan(due):
#SET review end.
                review_end = min(due, end_anchor)
#ELSE fallback.
            else:
#SET review end.
                review_end = end_anchor

#REVIEW phase.
            if not np.isnan(acc) and review_end > acc:
#HOVER text.
                hover_txt = (
                    f"Reviewer={rid}<br>"
                    f"Accepted={r.get('DateInvitationAccepted', '')}<br>"
                    f"Due={r.get('DateReviewDue', '')}<br>"
                    f"Submitted={r.get('DateReviewSubmitted', '') if pd.notna(r.get('DateReviewSubmitted', pd.NaT)) else 'NA'}"
                )
#APPEND review phase.
                stage_data["Review phase"].append({"rid": rid, "start": acc, "dur": review_end - acc, "hover": hover_txt})
#UPDATE xmax.
                xmax = max(xmax, review_end)

#REMINDER count.
            n_rem = r.get("NumRemindersSent", 0)
#SAFE cast.
            try:
#CAST.
                n_rem = int(n_rem)
#EXCEPT.
            except Exception:
#ZERO fallback.
                n_rem = 0

#REMINDER 1.
            if n_rem > 0 and not np.isnan(acc):
#POSITION.
                x1 = acc + REM1
#KEEP inside phase.
                if x1 <= review_end:
#APPEND.
                    rem_x.append(x1)
#APPEND.
                    rem_y.append(rid)

#REMINDER 2.
            if n_rem > 1 and not np.isnan(acc):
#POSITION.
                x2 = acc + REM2
#KEEP inside phase.
                if x2 <= review_end:
#APPEND.
                    rem_x.append(x2)
#APPEND.
                    rem_y.append(rid)

#NEW PHASE: reviewer submitted -> AE recommendation (what AE tells EIC).
            if not np.isnan(sub):
#END at AE recommendation if present, else EIC decision, else anchor.
                ae_phase_end = _pt_days(base_date, r.get("AE_RecommendationDateAtEnd", pd.NaT))
#FALLBACK.
                if np.isnan(ae_phase_end):
#USE EIC decision if AE rec missing.
                    ae_phase_end = _pt_days(base_date, r.get("EIC_DecisionDateAtEnd", pd.NaT))
#FALLBACK.
                if np.isnan(ae_phase_end):
#USE paper anchor.
                    ae_phase_end = end_anchor
#DRAW if valid.
                if not np.isnan(ae_phase_end) and ae_phase_end > sub:
#GET AE/EIC text.
                    ae_suggestion = str(r.get("AE_RecommendationAtEnd", ""))
#GET AE date.
                    ae_date = r.get("AE_RecommendationDateAtEnd", pd.NaT)
#FORMAT date.
                    ae_date_txt = ae_date.date().isoformat() if pd.notna(ae_date) else "NA"
#GET EIC text.
                    eic_dec = str(r.get("EIC_DecisionAtEnd", ""))
#GET EIC date.
                    eic_date = r.get("EIC_DecisionDateAtEnd", pd.NaT)
#FORMAT date.
                    eic_date_txt = eic_date.date().isoformat() if pd.notna(eic_date) else "NA"
#HOVER text.
                    hover_txt = (
                        f"Reviewer={rid}<br>"
                        f"Review submitted={r.get('DateReviewSubmitted', '')}<br>"
                        f"AE suggestion to EIC={ae_suggestion if ae_suggestion else 'NA'}<br>"
                        f"AE recommendation date={ae_date_txt}<br>"
                        f"EIC decision={eic_dec if eic_dec else 'NA'}<br>"
                        f"EIC decision date={eic_date_txt}"
                    )
#APPEND new phase.
                    stage_data["AE to EIC phase"].append({"rid": rid, "start": sub, "dur": ae_phase_end - sub, "hover": hover_txt})
#UPDATE xmax.
                    xmax = max(xmax, ae_phase_end)

#REVIEWER decision marker at submission.
                lvl = _pt_level(r.get("ReviewerPaperRating", ""))

#DRAW marker if level known.
                if lvl is not None:
#SIZE scaling.
                    size = 10 + (lvl * 4)
#APPEND x.
                    dec_x.append(sub)
#APPEND y.
                    dec_y.append(rid)
#APPEND size.
                    dec_size.append(size)
#APPEND label.
                    dec_text.append(_pt_abbr(r.get("ReviewerPaperRating", "")))
#APPEND hover.
                    dec_hover.append(
                        f"Reviewer={rid}<br>"
                        f"Submitted day={sub:.1f}<br>"
                        f"Decision={r.get('ReviewerPaperRating', '')}<br>"
                        f"Sentiment={r.get('ReviewSentiment_1to5', '')}<br>"
                        f"Words={r.get('ReviewLengthWords', '')}"
                    )
#UPDATE xmax.
                    xmax = max(xmax, sub)

#DONE accept path.
            continue

#KEEP xmax sensible.
    xmax = max(xmax, end_anchor, 0.0)

#GUARD if no phase bars.
    if sum(len(v) for v in stage_data.values()) == 0:
#WARN.
        st.warning("No phase bars were generated. Check reviewer dates for this paper-round.")
#SHOW diagnostic sample.
        st.dataframe(
            rr[
                [
                    c
                    for c in [
                        "ReviewerID",
                        "InviteOutcome",
                        "DateReviewerInvited",
                        "DateInvitationAccepted",
                        "DateInvitationResolved",
                        "DateNoResponseCensor",
                        "DateNoResponseTerminal",
                        "NoResponseTerminalOutcome",
                        "DateReviewSubmitted",
                        "AE_RecommendationAtEnd",
                        "AE_RecommendationDateAtEnd",
                        "EIC_DecisionAtEnd",
                        "EIC_DecisionDateAtEnd",
                    ]
                    if c in rr.columns
                ]
            ].head(80),
            width='stretch',
        )
#STOP.
        st.stop()

#CREATE figure.
    fig = go.Figure()

#ADD phase bars in order.
    for stage in ["Invite phase", "Review phase", "AE to EIC phase", "Declined"]:
#GET rows.
        rows = stage_data.get(stage, [])
#SKIP empty.
        if not rows:
#CONTINUE.
            continue
#Y categories.
        y = [t["rid"] for t in rows]
#START positions.
        base = [t["start"] for t in rows]
#BAR lengths.
        x = [t["dur"] for t in rows]
#HOVER strings.
        hover = [t["hover"] for t in rows]
#TRACE label.
        trace_name = stage if stage != "Declined" else "Invite phase (declined)"
#ADD bar trace.
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                base=base,
                orientation="h",
                name=trace_name,
                marker=dict(color=stage_colors.get(stage, "#999999")),
                opacity=0.95,
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

#ADD reminder markers.
    if len(rem_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=rem_x,
                y=rem_y,
                mode="markers",
                name="Reminder",
                marker=dict(symbol="line-ns-open", size=18, color="#FFD166", line=dict(width=2, color="#FFD166")),
                hovertemplate="Reviewer=%{y}<br>Reminder date=%{x:.1f} days<extra></extra>",
            )
        )

#ADD reviewer decision markers.
    if len(dec_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=dec_x,
                y=dec_y,
                mode="markers+text",
                text=dec_text,
                textposition="middle right",
                name="Reviewer decision",
                marker=dict(symbol="line-ns-open", size=dec_size, color="black", line=dict(width=3, color="black")),
                hovertext=dec_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#SUBMISSION line.
    fig.add_vline(x=0.0, line_width=1, line_dash="solid")
#ANNOTATE submission.
    fig.add_annotation(x=0.0, y=1.02, xref="x", yref="paper", text="Submitted", showarrow=False, font=dict(size=10))

#KEEP only EIC decision as paper-level terminal line.
    if not np.isnan(eic_line):
#ADD line.
        fig.add_vline(x=float(eic_line), line_width=1, line_dash="solid")
#ANNOTATE.
        fig.add_annotation(x=float(eic_line), y=1.02, xref="x", yref="paper", text="EIC decision", showarrow=False, font=dict(size=10))

#LAYOUT.
    fig.update_layout(
        barmode="overlay",
        bargap=0.70,
        title="Reviewer phases (slimmer bars) — x = calendar dates (21-day ticks)",
        height=max(520, 160 + len(reviewers) * 38),
        margin=dict(l=10, r=10, t=60, b=65),
        legend_title_text="Phase",
    )

#TICK interval.
    tick_step = 21.0
#X range max.
    x_max_plot = max(5.0, float(xmax) + 5.0)
#BUILD tick values.
    tickvals = list(np.arange(0.0, x_max_plot + 0.0001, tick_step))
#BUILD tick labels as calendar dates.
    ticktext = [(base_date + pd.Timedelta(days=float(v))).strftime("%d %b %Y") for v in tickvals]

#APPLY x-axis settings.
    fig.update_xaxes(
        type="linear",
        title="Date",
        range=[0.0, x_max_plot],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=45,
    )

#APPLY y-axis ordering.
    fig.update_yaxes(title="Reviewer", categoryorder="array", categoryarray=reviewers[::-1])

#SHOW chart.
    st.plotly_chart(fig, width='stretch')

#DETAIL table heading.
    st.markdown("#### Reviewer details (durations + decision + AE/EIC end state)")

#COPY reviewer table.
    rr2 = rr.copy()

#DURATION invite to accept.
    rr2["InviteToAccept_days"] = (rr2["DateInvitationAccepted"] - rr2["DateReviewerInvited"]).dt.days
#DURATION accept to submit.
    rr2["AcceptToSubmit_days"] = (rr2["DateReviewSubmitted"] - rr2["DateInvitationAccepted"]).dt.days
#DURATION overdue.
    rr2["Overdue_days"] = (rr2["DateReviewSubmitted"] - rr2["DateReviewDue"]).dt.days

#TABLE columns.
    cols = [
        "ReviewerID",
        "InviteOutcome",
        "ReviewerType",
        "ReviewerReliabilityTier",
        "ReviewerWorkloadAtInvite",
        "DateReviewerInvited",
        "DateInvitationAccepted",
        "DateInvitationResolved",
        "DateNoResponseCensor",
        "NoResponseTerminalOutcome",
        "DateNoResponseTerminal",
        "DateReviewDue",
        "DateReviewSubmitted",
        "NumRemindersSent",
        "InviteToAccept_days",
        "AcceptToSubmit_days",
        "Overdue_days",
        "ReviewSentiment_1to5",
        "ReviewerPaperRating",
        "ReviewLengthWords",
        "AE_RecommendationAtEnd",
        "AE_RecommendationDateAtEnd",
        "EIC_DecisionAtEnd",
        "EIC_DecisionDateAtEnd",
        "FinalDecisionOutcomeAtEnd",
    ]

#KEEP only present cols.
    cols = [c for c in cols if c in rr2.columns]

#SHOW table.
    st.dataframe(rr2[cols], width='stretch', height=420)

#REVIEWER TIMELINE TAB ONLY.
if selected_tab == "Reviewer timeline":
#LOCAL imports.
    import numpy as np
#LOCAL imports.
    import pandas as pd
#LOCAL imports.
    import plotly.express as px
#LOCAL imports.
    import plotly.graph_objects as go
#LOCAL imports.
    import streamlit as st

#TITLE.
    st.subheader("Reviewer POV: assignment timeline across papers")

#HELPER: safe datetime parsing.
    def _rt_to_dt(df, cols):
#COPY.
        df = df.copy()
#PARSE requested cols.
        for c in cols:
#CHECK existence.
            if c in df.columns:
#CONVERT.
                df[c] = pd.to_datetime(df[c], errors="coerce")
#RETURN parsed copy.
        return df

#HELPER: short reviewer-decision label.
    def _rt_abbr(dec):
#NORMALIZE.
        dec = str(dec).strip().lower()
#MAP.
        m = {"reject": "REJ", "submit as new": "SNEW", "major revision": "MAJ", "minor revision": "MIN", "accept": "ACC"}
#RETURN.
        return m.get(dec, "")

#USE filtered paper df if available.
    try:
#SOURCE.
        _paper_src = paper_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _paper_src = paper_df

#USE filtered reviewer df if available.
    try:
#SOURCE.
        _rev_src = rev_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _rev_src = rev_df

#PARSE paper dates.
    paper = _rt_to_dt(
        _paper_src,
        [
            "DatePaperSubmitted",
            "DateReviewersFullyAssigned",
            "DateAllReviewsReceived",
            "AE_RecommendationDate",
            "EIC_DecisionDate",
            "DateDecisionLetterSent",
        ],
    )

#PARSE reviewer dates.
    rev = _rt_to_dt(
        _rev_src,
        [
            "DateReviewerInvited",
            "DateInvitationAccepted",
            "DateInvitationResolved",
            "DateNoResponseCensor",
            "DateNoResponseTerminal",
            "DateReviewDue",
            "DateReviewSubmitted",
            "AE_RecommendationDateAtEnd",
            "EIC_DecisionDateAtEnd",
            "DateDecisionLetterSentAtEnd",
        ],
    )

#GUARD required paper cols.
    if ("PaperID" not in paper.columns) or ("SubmissionRound" not in paper.columns):
#ERROR.
        st.error("PaperHeader must contain PaperID and SubmissionRound.")
#STOP.
        st.stop()

#GUARD required reviewer cols.
    if ("PaperID" not in rev.columns) or ("SubmissionRound" not in rev.columns) or ("ReviewerID" not in rev.columns) or ("InviteOutcome" not in rev.columns):
#ERROR.
        st.error("ReviewerRows must contain PaperID, SubmissionRound, ReviewerID, and InviteOutcome.")
#STOP.
        st.stop()

#ADD paper-level context to reviewer rows if missing.
    join_cols = ["PaperID", "SubmissionRound"]

#SELECT paper context cols.
    paper_ctx_cols = [
        "PaperID",
        "SubmissionRound",
        "DatePaperSubmitted",
        "JournalSection",
        "PaperStatusOnSubmission",
        "HandlingAssociateEditorID",
        "HandlingEIC_ID",
    ]

#KEEP only existing cols.
    paper_ctx_cols = [c for c in paper_ctx_cols if c in paper.columns]

#MERGE paper context into reviewer rows.
    rev = rev.merge(paper[paper_ctx_cols].drop_duplicates(), on=join_cols, how="left")

#GUARD reviewer list.
    if rev["ReviewerID"].dropna().empty:
#INFO.
        st.info("No reviewers available in current filter scope.")
#STOP.
        st.stop()

#BUILD reviewer options.
    reviewer_options = sorted(rev["ReviewerID"].dropna().astype(str).unique().tolist())

#SELECT reviewer.
    selected_reviewer = st.selectbox("Select a reviewer", reviewer_options)

#FILTER to selected reviewer.
    rr = rev[rev["ReviewerID"].astype(str) == str(selected_reviewer)].copy()

#GUARD empty after filter.
    if rr.empty:
#INFO.
        st.info("No assignments found for this reviewer.")
#STOP.
        st.stop()

#BUILD friendly row label.
    rr["PaperRoundLabel"] = rr["PaperID"].astype(str) + " | round " + rr["SubmissionRound"].astype(int).astype(str)

#SORT by invite date.
    rr = rr.sort_values(["DateReviewerInvited", "PaperID", "SubmissionRound"], ascending=[True, True, True]).reset_index(drop=True)

#OPTIONAL status filter.
    status_choices = ["accept", "decline", "no_response"]
#MULTISELECT.
    selected_status = st.multiselect("Filter invite outcome", status_choices, default=status_choices)

#APPLY status filter.
    rr = rr[rr["InviteOutcome"].astype(str).isin(selected_status)].copy()

#GUARD after status filter.
    if rr.empty:
#INFO.
        st.info("No assignments match the selected status filter.")
#STOP.
        st.stop()

#COMPUTE quick stats.
    total_assignments = int(len(rr))
#COUNT accepts.
    total_accepts = int((rr["InviteOutcome"].astype(str) == "accept").sum())
#COUNT declines.
    total_declines = int((rr["InviteOutcome"].astype(str) == "decline").sum())
#COUNT no response.
    total_noresp = int((rr["InviteOutcome"].astype(str) == "no_response").sum())
#COUNT submitted reviews.
    total_submitted = int(rr["DateReviewSubmitted"].notna().sum())

#SHOW metrics.
    c1, c2, c3, c4, c5 = st.columns(5)
#METRIC.
    c1.metric("Assignments", total_assignments)
#METRIC.
    c2.metric("Accepted", total_accepts)
#METRIC.
    c3.metric("Declined", total_declines)
#METRIC.
    c4.metric("No response", total_noresp)
#METRIC.
    c5.metric("Submitted", total_submitted)

#COLLECT timeline segments.
    segments = []

#COLLECT submission markers.
    subm_x = []
#COLLECT submission markers.
    subm_y = []
#COLLECT submission hover.
    subm_hover = []

#COLLECT due markers.
    due_x = []
#COLLECT due markers.
    due_y = []
#COLLECT due hover.
    due_hover = []

#COLLECT reminder markers.
    rem_x = []
#COLLECT reminder markers.
    rem_y = []
#COLLECT reminder hover.
    rem_hover = []

#COLLECT reviewer-decision markers.
    dec_x = []
#COLLECT reviewer-decision markers.
    dec_y = []
#COLLECT reviewer-decision marker text.
    dec_text = []
#COLLECT reviewer-decision hover.
    dec_hover = []

#REMINDER policy offsets.
    REM1 = 21
#REMINDER policy offsets.
    REM2 = 42

#BUILD row by row.
    for _, r in rr.iterrows():
#ROW label.
        row_label = str(r["PaperRoundLabel"])
#OUTCOME.
        outcome = str(r.get("InviteOutcome", "")).strip()

#DATES.
        paper_sub_dt = r.get("DatePaperSubmitted", pd.NaT)
#DATES.
        inv_dt = r.get("DateReviewerInvited", pd.NaT)
#DATES.
        acc_dt = r.get("DateInvitationAccepted", pd.NaT)
#DATES.
        res_dt = r.get("DateInvitationResolved", pd.NaT)
#DATES.
        censor_dt = r.get("DateNoResponseCensor", pd.NaT)
#DATES.
        terminal_dt = r.get("DateNoResponseTerminal", pd.NaT)
#DATES.
        due_dt = r.get("DateReviewDue", pd.NaT)
#DATES.
        sub_dt = r.get("DateReviewSubmitted", pd.NaT)
#DATES.
        ae_dt = r.get("AE_RecommendationDateAtEnd", pd.NaT)
#DATES.
        eic_dt = r.get("EIC_DecisionDateAtEnd", pd.NaT)
#DATES.
        letter_dt = r.get("DateDecisionLetterSentAtEnd", pd.NaT)

#MARK paper submission if available.
        if pd.notna(paper_sub_dt):
#APPEND x.
            subm_x.append(paper_sub_dt)
#APPEND y.
            subm_y.append(row_label)
#APPEND hover.
            subm_hover.append(
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"Paper submitted={paper_sub_dt.date().isoformat()}"
            )

#NO RESPONSE: invite phase extends until terminal event.
        if outcome == "no_response":
#FALLBACK terminal to censor.
            if pd.isna(terminal_dt):
#SET fallback.
                terminal_dt = censor_dt
#FINAL fallback to letter/EIC.
            if pd.isna(terminal_dt):
#SET fallback.
                terminal_dt = letter_dt
#FALLBACK.
            if pd.isna(terminal_dt):
#SET fallback.
                terminal_dt = eic_dt
#ONLY draw if invite exists.
            if pd.notna(inv_dt) and pd.notna(terminal_dt) and terminal_dt > inv_dt:
#APPEND segment.
                segments.append(
                    {
                        "PaperRoundLabel": row_label,
                        "Stage": "Invite phase",
                        "Start": inv_dt,
                        "Finish": terminal_dt,
                        "PaperID": r.get("PaperID", ""),
                        "SubmissionRound": r.get("SubmissionRound", ""),
                        "JournalSection": r.get("JournalSection", ""),
                        "PaperStatusOnSubmission": r.get("PaperStatusOnSubmission", ""),
                        "InviteOutcome": outcome,
                        "HoverText": (
                            f"Paper={r.get('PaperID','')}<br>"
                            f"Round={r.get('SubmissionRound','')}<br>"
                            f"Invite sent={inv_dt.date().isoformat() if pd.notna(inv_dt) else 'NA'}<br>"
                            f"No-response terminal={r.get('NoResponseTerminalOutcome','NA')}<br>"
                            f"Terminal date={terminal_dt.date().isoformat() if pd.notna(terminal_dt) else 'NA'}"
                        ),
                    }
                )
#DONE.
            continue

#DECLINE: invite to resolved.
        if outcome == "decline":
#FALLBACK resolved to invite+2d.
            if pd.isna(res_dt) and pd.notna(inv_dt):
#SET fallback.
                res_dt = inv_dt + pd.Timedelta(days=2)
#DRAW if valid.
            if pd.notna(inv_dt) and pd.notna(res_dt) and res_dt > inv_dt:
#APPEND segment.
                segments.append(
                    {
                        "PaperRoundLabel": row_label,
                        "Stage": "Declined",
                        "Start": inv_dt,
                        "Finish": res_dt,
                        "PaperID": r.get("PaperID", ""),
                        "SubmissionRound": r.get("SubmissionRound", ""),
                        "JournalSection": r.get("JournalSection", ""),
                        "PaperStatusOnSubmission": r.get("PaperStatusOnSubmission", ""),
                        "InviteOutcome": outcome,
                        "HoverText": (
                            f"Paper={r.get('PaperID','')}<br>"
                            f"Round={r.get('SubmissionRound','')}<br>"
                            f"Invite sent={inv_dt.date().isoformat() if pd.notna(inv_dt) else 'NA'}<br>"
                            f"Declined={res_dt.date().isoformat() if pd.notna(res_dt) else 'NA'}"
                        ),
                    }
                )
#DONE.
            continue

#ACCEPT PATH.
        if outcome == "accept":
#FALLBACK accept date to resolved date.
            if pd.isna(acc_dt):
#SET fallback.
                acc_dt = res_dt

#INVITE PHASE: invite to accept.
            if pd.notna(inv_dt) and pd.notna(acc_dt) and acc_dt > inv_dt:
#APPEND segment.
                segments.append(
                    {
                        "PaperRoundLabel": row_label,
                        "Stage": "Invite phase",
                        "Start": inv_dt,
                        "Finish": acc_dt,
                        "PaperID": r.get("PaperID", ""),
                        "SubmissionRound": r.get("SubmissionRound", ""),
                        "JournalSection": r.get("JournalSection", ""),
                        "PaperStatusOnSubmission": r.get("PaperStatusOnSubmission", ""),
                        "InviteOutcome": outcome,
                        "HoverText": (
                            f"Paper={r.get('PaperID','')}<br>"
                            f"Round={r.get('SubmissionRound','')}<br>"
                            f"Invite sent={inv_dt.date().isoformat() if pd.notna(inv_dt) else 'NA'}<br>"
                            f"Accepted={acc_dt.date().isoformat() if pd.notna(acc_dt) else 'NA'}"
                        ),
                    }
                )

#REVIEW END: submit if present, else due, else decision anchor.
            if pd.notna(sub_dt):
#SET review end.
                review_end_dt = sub_dt
#ELIF due exists.
            elif pd.notna(due_dt):
#SET review end.
                review_end_dt = due_dt
#ELIF letter exists.
            elif pd.notna(letter_dt):
#SET review end.
                review_end_dt = letter_dt
#ELIF EIC decision exists.
            elif pd.notna(eic_dt):
#SET review end.
                review_end_dt = eic_dt
#ELSE.
            else:
#SET missing.
                review_end_dt = pd.NaT

#REVIEW PHASE: accept to submit / due / anchor.
            if pd.notna(acc_dt) and pd.notna(review_end_dt) and review_end_dt > acc_dt:
#APPEND segment.
                segments.append(
                    {
                        "PaperRoundLabel": row_label,
                        "Stage": "Review phase",
                        "Start": acc_dt,
                        "Finish": review_end_dt,
                        "PaperID": r.get("PaperID", ""),
                        "SubmissionRound": r.get("SubmissionRound", ""),
                        "JournalSection": r.get("JournalSection", ""),
                        "PaperStatusOnSubmission": r.get("PaperStatusOnSubmission", ""),
                        "InviteOutcome": outcome,
                        "HoverText": (
                            f"Paper={r.get('PaperID','')}<br>"
                            f"Round={r.get('SubmissionRound','')}<br>"
                            f"Accepted={acc_dt.date().isoformat() if pd.notna(acc_dt) else 'NA'}<br>"
                            f"Due={due_dt.date().isoformat() if pd.notna(due_dt) else 'NA'}<br>"
                            f"Submitted={sub_dt.date().isoformat() if pd.notna(sub_dt) else 'NA'}"
                        ),
                    }
                )

#ADD due marker if available.
            if pd.notna(due_dt):
#APPEND x.
                due_x.append(due_dt)
#APPEND y.
                due_y.append(row_label)
#APPEND hover.
                due_hover.append(
                    f"Paper={r.get('PaperID','')}<br>"
                    f"Round={r.get('SubmissionRound','')}<br>"
                    f"Review due={due_dt.date().isoformat()}"
                )

#ADD reminder markers if inside review phase.
            try:
#CAST reminder count.
                n_rem = int(r.get("NumRemindersSent", 0))
#EXCEPT.
            except Exception:
#FALLBACK zero.
                n_rem = 0

#REMINDER 1 marker.
            if n_rem > 0 and pd.notna(acc_dt):
#POSITION.
                rem1_dt = acc_dt + pd.Timedelta(days=REM1)
#APPEND if before review end.
                if pd.notna(review_end_dt) and rem1_dt <= review_end_dt:
#APPEND x.
                    rem_x.append(rem1_dt)
#APPEND y.
                    rem_y.append(row_label)
#APPEND hover.
                    rem_hover.append(
                        f"Paper={r.get('PaperID','')}<br>"
                        f"Round={r.get('SubmissionRound','')}<br>"
                        f"Reminder 1={rem1_dt.date().isoformat()}"
                    )

#REMINDER 2 marker.
            if n_rem > 1 and pd.notna(acc_dt):
#POSITION.
                rem2_dt = acc_dt + pd.Timedelta(days=REM2)
#APPEND if before review end.
                if pd.notna(review_end_dt) and rem2_dt <= review_end_dt:
#APPEND x.
                    rem_x.append(rem2_dt)
#APPEND y.
                    rem_y.append(row_label)
#APPEND hover.
                    rem_hover.append(
                        f"Paper={r.get('PaperID','')}<br>"
                        f"Round={r.get('SubmissionRound','')}<br>"
                        f"Reminder 2={rem2_dt.date().isoformat()}"
                    )

#POST-REVIEW EDITORIAL PHASE: reviewer submitted to decision letter.
            if pd.notna(sub_dt):
#CHOOSE editorial end.
                if pd.notna(letter_dt):
#SET end.
                    editor_end_dt = letter_dt
#ELIF EIC decision exists.
                elif pd.notna(eic_dt):
#SET end.
                    editor_end_dt = eic_dt
#ELIF AE recommendation exists.
                elif pd.notna(ae_dt):
#SET end.
                    editor_end_dt = ae_dt
#ELSE.
                else:
#SET missing.
                    editor_end_dt = pd.NaT

#DRAW if valid.
                if pd.notna(editor_end_dt) and editor_end_dt > sub_dt:
#APPEND segment.
                    segments.append(
                        {
                            "PaperRoundLabel": row_label,
                            "Stage": "Editorial outcome phase",
                            "Start": sub_dt,
                            "Finish": editor_end_dt,
                            "PaperID": r.get("PaperID", ""),
                            "SubmissionRound": r.get("SubmissionRound", ""),
                            "JournalSection": r.get("JournalSection", ""),
                            "PaperStatusOnSubmission": r.get("PaperStatusOnSubmission", ""),
                            "InviteOutcome": outcome,
                            "HoverText": (
                                f"Paper={r.get('PaperID','')}<br>"
                                f"Round={r.get('SubmissionRound','')}<br>"
                                f"Review submitted={sub_dt.date().isoformat() if pd.notna(sub_dt) else 'NA'}<br>"
                                f"AE suggestion={r.get('AE_RecommendationAtEnd','NA')}<br>"
                                f"AE recommendation date={ae_dt.date().isoformat() if pd.notna(ae_dt) else 'NA'}<br>"
                                f"EIC decision={r.get('EIC_DecisionAtEnd','NA')}<br>"
                                f"EIC decision date={eic_dt.date().isoformat() if pd.notna(eic_dt) else 'NA'}<br>"
                                f"Decision letter={letter_dt.date().isoformat() if pd.notna(letter_dt) else 'NA'}"
                            ),
                        }
                    )

#ADD reviewer-decision marker at submission.
                if pd.notna(sub_dt):
#APPEND x.
                    dec_x.append(sub_dt)
#APPEND y.
                    dec_y.append(row_label)
#APPEND text.
                    dec_text.append(_rt_abbr(r.get("ReviewerPaperRating", "")))
#APPEND hover.
                    dec_hover.append(
                        f"Paper={r.get('PaperID','')}<br>"
                        f"Round={r.get('SubmissionRound','')}<br>"
                        f"Review submitted={sub_dt.date().isoformat()}<br>"
                        f"Reviewer decision={r.get('ReviewerPaperRating','')}<br>"
                        f"Sentiment={r.get('ReviewSentiment_1to5','')}<br>"
                        f"Words={r.get('ReviewLengthWords','')}"
                    )

#DONE accept path.
            continue

#BUILD segments dataframe.
    seg_df = pd.DataFrame(segments)

#GUARD no segments.
    if seg_df.empty:
#WARN.
        st.warning("No timeline segments were generated for this reviewer.")
#SHOW diagnostic.
        st.dataframe(rr, width='stretch')
#STOP.
        st.stop()

#COLOR map.
    color_map = {
        "Invite phase": "#4C78A8",
        "Review phase": "#54A24B",
        "Editorial outcome phase": "#B279A2",
        "Declined": "#9D9D9D",
    }

#BUILD timeline chart.
    fig = px.timeline(
        seg_df,
        x_start="Start",
        x_end="Finish",
        y="PaperRoundLabel",
        color="Stage",
        color_discrete_map=color_map,
        hover_data={
            "PaperID": True,
            "SubmissionRound": True,
            "JournalSection": True,
            "PaperStatusOnSubmission": True,
            "InviteOutcome": True,
            "Start": True,
            "Finish": True,
            "HoverText": True,
        },
    )

#MAKE rows top-down by earliest invite.
    fig.update_yaxes(autorange="reversed")

#SLIMMER visual style.
    fig.update_traces(marker_line_width=0)

#ADD paper submission markers.
    if len(subm_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=subm_x,
                y=subm_y,
                mode="markers",
                name="Paper submitted",
                marker=dict(symbol="circle-open", size=9, color="#777777", line=dict(width=2, color="#777777")),
                hovertext=subm_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#ADD due markers.
    if len(due_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=due_x,
                y=due_y,
                mode="markers",
                name="Due date",
                marker=dict(symbol="diamond-open", size=9, color="#E45756", line=dict(width=2, color="#E45756")),
                hovertext=due_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#ADD reminder markers.
    if len(rem_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=rem_x,
                y=rem_y,
                mode="markers",
                name="Reminder",
                marker=dict(symbol="line-ns-open", size=18, color="#FFD166", line=dict(width=2, color="#FFD166")),
                hovertext=rem_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#ADD reviewer decision markers.
    if len(dec_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=dec_x,
                y=dec_y,
                mode="markers+text",
                text=dec_text,
                textposition="middle right",
                name="Reviewer decision",
                marker=dict(symbol="line-ns-open", size=16, color="black", line=dict(width=3, color="black")),
                hovertext=dec_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#SET layout.
    fig.update_layout(
        title=f"Timeline for reviewer {selected_reviewer}",
        height=max(520, 160 + len(rr) * 38),
        margin=dict(l=10, r=10, t=60, b=65),
        legend_title_text="Phase",
        bargap=0.70,
    )


    fig.update_xaxes(
    title="Date",
    dtick="M6",
    tick0="2024-01-01",
    tickformat="%b %Y",
    tickangle=45,
)

#SHOW chart.
    # st.plotly_chart(fig, width='stretch')

    #SHOW chart with click-to-paper navigation.
    reviewer_event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="reviewer_plot",
        on_select="rerun"
    )

#CLICK → STORE TARGET PAPER AND MOVE TO PAPER TIMELINE.
    if reviewer_event and "selection" in reviewer_event:
#GET selected points.
        pts = reviewer_event["selection"].get("points", [])
#HANDLE only if something was selected.
        if pts:
#GET y-axis value. In Reviewer timeline this is PaperRoundLabel.
            selected_label = pts[0].get("y")
#AVOID repeated rerun on same selected paper.
            if selected_label and st.session_state.get("last_reviewer_clicked_paper") != selected_label:
#STORE selected paper for Paper timeline selectbox.
                st.session_state["selected_paper_from_ae"] = selected_label
#STORE last clicked reviewer paper.
                st.session_state["last_reviewer_clicked_paper"] = selected_label
#MOVE navigation to Paper timeline.
                st.session_state["active_tab"] = "Paper timeline"
#REUSE your existing nav sync flag.
                st.session_state["from_ae_click"] = True
#RERUN.
                st.rerun()

#DETAILS heading.
    st.markdown("#### Reviewer assignment details")

#COPY details df.
    rr2 = rr.copy()

#DURATION invite to accept.
    rr2["InviteToAccept_days"] = (rr2["DateInvitationAccepted"] - rr2["DateReviewerInvited"]).dt.days

#DURATION accept to submit.
    rr2["AcceptToSubmit_days"] = (rr2["DateReviewSubmitted"] - rr2["DateInvitationAccepted"]).dt.days

#DURATION overdue.
    rr2["Overdue_days"] = (rr2["DateReviewSubmitted"] - rr2["DateReviewDue"]).dt.days

#DETAIL columns.
    cols = [
        "PaperID",
        "SubmissionRound",
        "JournalSection",
        "PaperStatusOnSubmission",
        "InviteOutcome",
        "ReviewerID",
        "ReviewerType",
        "ReviewerReliabilityTier",
        "ReviewerWorkloadAtInvite",
        "DatePaperSubmitted",
        "DateReviewerInvited",
        "DateInvitationAccepted",
        "DateInvitationResolved",
        "DateNoResponseCensor",
        "NoResponseTerminalOutcome",
        "DateNoResponseTerminal",
        "DateReviewDue",
        "DateReviewSubmitted",
        "NumRemindersSent",
        "InviteToAccept_days",
        "AcceptToSubmit_days",
        "Overdue_days",
        "ReviewSentiment_1to5",
        "ReviewerPaperRating",
        "ReviewLengthWords",
        "AE_RecommendationAtEnd",
        "AE_RecommendationDateAtEnd",
        "EIC_DecisionAtEnd",
        "EIC_DecisionDateAtEnd",
        "FinalDecisionOutcomeAtEnd",
    ]

#KEEP existing cols only.
    cols = [c for c in cols if c in rr2.columns]

#SHOW details table.
    st.dataframe(rr2[cols], width='stretch', height=420)


#AE TIMELINE TAB ONLY.
if selected_tab == "AE timeline":
#LOCAL imports.
    import numpy as np
#LOCAL imports.
    import pandas as pd
#LOCAL imports.
    import plotly.express as px
#LOCAL imports.
    import plotly.graph_objects as go
#LOCAL imports.
    import streamlit as st

#TITLE.
    st.subheader("AE POV: invite → review → recommend to EIC")

#HELPER: safe datetime parsing.
    def _ae_to_dt(df, cols):
#COPY.
        df = df.copy()
#PARSE requested cols.
        for c in cols:
#CHECK existence.
            if c in df.columns:
#CONVERT.
                df[c] = pd.to_datetime(df[c], errors="coerce")
#RETURN parsed copy.
        return df

#HELPER: reviewer decision text.
    def _ae_dec_text(dec):
#NORMALIZE.
        dec = str(dec).strip().lower()
#MAP nice text.
        m = {
            "accept": "ACC",
            "minor revision": "MIN",
            "major revision": "MAJ",
            "submit as new": "NEW",
            "reject": "REJ",
        }
#RETURN formatted text.
        return m.get(dec, str(dec).strip())

#USE filtered paper df if available.
    try:
#SOURCE.
        _paper_src = paper_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _paper_src = paper_df

#USE filtered reviewer df if available.
    try:
#SOURCE.
        _rev_src = rev_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _rev_src = rev_df

#PARSE paper dates.
    paper = _ae_to_dt(
        _paper_src,
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

#PARSE reviewer dates.
    rev = _ae_to_dt(
        _rev_src,
        [
            "DateReviewerInvited",
            "DateInvitationAccepted",
            "DateInvitationResolved",
            "DateNoResponseCensor",
            "DateNoResponseTerminal",
            "DateReviewDue",
            "DateReviewSubmitted",
        ],
    )

#GUARD paper cols.
    if ("PaperID" not in paper.columns) or ("SubmissionRound" not in paper.columns) or ("HandlingAssociateEditorID" not in paper.columns):
#ERROR.
        st.error("PaperHeader must contain PaperID, SubmissionRound, and HandlingAssociateEditorID.")
#STOP.
        st.stop()

#GUARD reviewer cols.
    if ("PaperID" not in rev.columns) or ("SubmissionRound" not in rev.columns) or ("InviteOutcome" not in rev.columns):
#ERROR.
        st.error("ReviewerRows must contain PaperID, SubmissionRound, and InviteOutcome.")
#STOP.
        st.stop()

#AVAILABLE AEs.
    ae_options = sorted(paper["HandlingAssociateEditorID"].dropna().astype(str).unique().tolist())

#GUARD no AEs.
    if len(ae_options) == 0:
#INFO.
        st.info("No AE values found in current filter scope.")
#STOP.
        st.stop()

#SELECT AE.
    selected_ae = st.selectbox(
        "Select an AE",
        ae_options,
        key="ae_timeline_select_ae",
    )

#FILTER papers for AE.
    p_ae = paper[paper["HandlingAssociateEditorID"].astype(str) == str(selected_ae)].copy()

#GUARD empty.
    if p_ae.empty:
#INFO.
        st.info("No paper-rounds found for this AE.")
#STOP.
        st.stop()

#JOIN reviewer rows to this AE's papers only.
    join_cols = ["PaperID", "SubmissionRound"]
#PAPER subset ids.
    p_ids = p_ae[join_cols].drop_duplicates()
#INNER join.
    r_ae = rev.merge(p_ids, on=join_cols, how="inner")

#BUILD row label.
    p_ae["PaperRoundLabel"] = p_ae["PaperID"].astype(str) + " | round " + p_ae["SubmissionRound"].astype(int).astype(str)

#AGGREGATE reviewer-side info per paper-round.
    def _n_accept(s):
#COUNT accepts.
        return int((s.astype(str) == "accept").sum())
    def _n_decline(s):
#COUNT declines.
        return int((s.astype(str) == "decline").sum())
    def _n_noresp(s):
#COUNT no responses.
        return int((s.astype(str) == "no_response").sum())
    def _n_submitted(s):
#COUNT submitted.
        return int(pd.to_datetime(s, errors="coerce").notna().sum())
    def _n_late(flag_series):
#COUNT late.
        return int(flag_series.astype(str).str.lower().eq("yes").sum())

#SAFE grouped reviewer stats.
    if not r_ae.empty:
#GROUP reviewer rows.
        rev_grp = r_ae.groupby(join_cols).agg(
            InvitesSent=("ReviewerID", "count"),
            Accepted=("InviteOutcome", _n_accept),
            Declined=("InviteOutcome", _n_decline),
            NoResponse=("InviteOutcome", _n_noresp),
            ReviewsSubmitted=("DateReviewSubmitted", _n_submitted),
            LateReviews=("LateSubmissionFlag", _n_late),
            AvgReminders=("NumRemindersSent", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).mean()),
            FirstInviteDate=("DateReviewerInvited", "min"),
            LastInviteDate=("DateReviewerInvited", "max"),
            FirstAcceptDate=("DateInvitationAccepted", "min"),
            MaxDisagreement=("ReviewerDisagreementScore", lambda s: pd.to_numeric(s, errors="coerce").max()),
            ReviewerDecisionSet=("ReviewerPaperRating", lambda s: ", ".join(sorted([_ae_dec_text(x) for x in s.dropna().astype(str) if str(x).strip() != ""]))),
        ).reset_index()
#ELSE fallback.
    else:
#EMPTY frame.
        rev_grp = pd.DataFrame(columns=join_cols + ["InvitesSent","Accepted","Declined","NoResponse","ReviewsSubmitted","LateReviews","AvgReminders","FirstInviteDate","LastInviteDate","FirstAcceptDate","MaxDisagreement","ReviewerDecisionSet"])

#MERGE aggregates to paper rows.
    ae_df = p_ae.merge(rev_grp, on=join_cols, how="left")

#FILL numeric blanks.
    for c in ["InvitesSent","Accepted","Declined","NoResponse","ReviewsSubmitted","LateReviews"]:
#IF exists.
        if c in ae_df.columns:
#FILL 0.
            ae_df[c] = pd.to_numeric(ae_df[c], errors="coerce").fillna(0).astype(int)

#FILL float blank.
    if "AvgReminders" in ae_df.columns:
#FILL.
        ae_df["AvgReminders"] = pd.to_numeric(ae_df["AvgReminders"], errors="coerce").fillna(0.0)

#COMPUTE durations for KPIs.
    ae_df["InvitePhaseDays"] = (ae_df["DateReviewersFullyAssigned"] - ae_df["DatePaperSubmitted"]).dt.days
#COMPUTE review days.
    ae_df["ReviewPhaseDays"] = (ae_df["DateAllReviewsReceived"] - ae_df["DateReviewersFullyAssigned"]).dt.days
#COMPUTE recommend days.
    ae_df["RecommendPhaseDays"] = (ae_df["AE_RecommendationDate"] - ae_df["DateAllReviewsReceived"]).dt.days

#TOP KPIs.
    c1, c2, c3, c4, c5 = st.columns(5)
#METRIC papers.
    c1.metric("Paper-rounds", f"{len(ae_df):,}")
#METRIC invite median.
    c2.metric("Median invite days", f"{pd.to_numeric(ae_df['InvitePhaseDays'], errors='coerce').median():.0f}")
#METRIC review median.
    c3.metric("Median review days", f"{pd.to_numeric(ae_df['ReviewPhaseDays'], errors='coerce').median():.0f}")
#METRIC recommend median.
    c4.metric("Median recommend days", f"{pd.to_numeric(ae_df['RecommendPhaseDays'], errors='coerce').median():.0f}")
#METRIC backlog risk.
    c5.metric("Papers with no-response", f"{int((ae_df['NoResponse'] > 0).sum()):,}")

#OPTIONAL section filter.
    section_options = ["All"] + sorted(ae_df["JournalSection"].dropna().astype(str).unique().tolist()) if "JournalSection" in ae_df.columns else ["All"]
#SELECT section.
    selected_section = st.selectbox("Filter by section", section_options, key="ae_timeline_select_section")

#APPLY section filter.
    if selected_section != "All" and "JournalSection" in ae_df.columns:
#FILTER.
        ae_df = ae_df[ae_df["JournalSection"].astype(str) == selected_section].copy()

#GUARD after filter.
    if ae_df.empty:
#INFO.
        st.info("No paper-rounds remain after section filter.")
#STOP.
        st.stop()

#SORT rows by submission.
    ae_df = ae_df.sort_values(["DatePaperSubmitted","PaperID","SubmissionRound"], ascending=[True, True, True]).reset_index(drop=True)

#COLLECT phase segments.
    segments = []

#COLLECT markers: first invite.
    first_inv_x = []
#COLLECT markers: first invite.
    first_inv_y = []
#COLLECT markers: first invite hover.
    first_inv_hover = []

#COLLECT markers: first review.
    first_rev_x = []
#COLLECT markers: first review.
    first_rev_y = []
#COLLECT markers: first review hover.
    first_rev_hover = []

#COLLECT AE recommendation text inside recommend phase.
    ae_text_x = []
#COLLECT AE recommendation text inside recommend phase.
    ae_text_y = []
#COLLECT AE recommendation text inside recommend phase.
    ae_text = []
#COLLECT AE recommendation text inside recommend phase hover.
    ae_text_hover = []

#COLLECT EIC decision line text on right.
    eic_text_x = []
#COLLECT EIC decision line text on right.
    eic_text_y = []
#COLLECT EIC decision line text on right.
    eic_text = []
#COLLECT EIC decision line text hover.
    eic_text_hover = []

#TRACK x max.
    xmax_dt = pd.Timestamp.min

#BUILD each paper-round row.
    for _, r in ae_df.iterrows():
#ROW label.
        row_label = str(r["PaperRoundLabel"])

#DATES.
        sub_dt = r.get("DatePaperSubmitted", pd.NaT)
#DATES.
        full_assign_dt = r.get("DateReviewersFullyAssigned", pd.NaT)
#DATES.
        first_review_dt = r.get("DateFirstReviewReceived", pd.NaT)
#DATES.
        all_reviews_dt = r.get("DateAllReviewsReceived", pd.NaT)
#DATES.
        ae_rec_dt = r.get("AE_RecommendationDate", pd.NaT)
#DATES.
        eic_dec_dt = r.get("EIC_DecisionDate", pd.NaT)
#DATES.
        decision_letter_dt = r.get("DateDecisionLetterSent", pd.NaT)

#INVITE PHASE: submission -> reviewers fully assigned.
        if pd.notna(sub_dt) and pd.notna(full_assign_dt) and full_assign_dt > sub_dt:
#BUILD hover.
            invite_hover = (
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"Section={r.get('JournalSection','')}<br>"
                f"Submitted={sub_dt.date().isoformat()}<br>"
                f"First invite={r.get('FirstInviteDate', pd.NaT).date().isoformat() if pd.notna(r.get('FirstInviteDate', pd.NaT)) else 'NA'}<br>"
                f"Last invite={r.get('LastInviteDate', pd.NaT).date().isoformat() if pd.notna(r.get('LastInviteDate', pd.NaT)) else 'NA'}<br>"
                f"Reviewers fully assigned={full_assign_dt.date().isoformat()}<br>"
                f"Invites sent={r.get('InvitesSent',0)}<br>"
                f"Accepted={r.get('Accepted',0)}<br>"
                f"Declined={r.get('Declined',0)}<br>"
                f"No response={r.get('NoResponse',0)}"
            )
#APPEND segment.
            segments.append(
                {
                    "PaperRoundLabel": row_label,
                    "Stage": "Invite phase",
                    "Start": sub_dt,
                    "Finish": full_assign_dt,
                    "HoverText": invite_hover,
                }
            )
#UPDATE xmax.
            xmax_dt = max(xmax_dt, full_assign_dt)

#ADD first invite marker if available.
        if pd.notna(r.get("FirstInviteDate", pd.NaT)):
#APPEND x.
            first_inv_x.append(r.get("FirstInviteDate", pd.NaT))
#APPEND y.
            first_inv_y.append(row_label)
#APPEND hover.
            first_inv_hover.append(
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"First invite={r.get('FirstInviteDate', pd.NaT).date().isoformat()}"
            )

#REVIEW PHASE: reviewers fully assigned -> all reviews received.
        if pd.notna(full_assign_dt) and pd.notna(all_reviews_dt) and all_reviews_dt > full_assign_dt:
#BUILD hover.
            review_hover = (
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"Reviewers fully assigned={full_assign_dt.date().isoformat()}<br>"
                f"First review received={first_review_dt.date().isoformat() if pd.notna(first_review_dt) else 'NA'}<br>"
                f"All reviews received={all_reviews_dt.date().isoformat()}<br>"
                f"Reviews submitted={r.get('ReviewsSubmitted',0)}<br>"
                f"Late reviews={r.get('LateReviews',0)}<br>"
                f"Avg reminders={r.get('AvgReminders',0):.1f}<br>"
                f"Reviewer decisions={r.get('ReviewerDecisionSet','') if str(r.get('ReviewerDecisionSet','')).strip() else 'NA'}<br>"
                f"Max disagreement={r.get('MaxDisagreement','NA')}"
            )
#APPEND segment.
            segments.append(
                {
                    "PaperRoundLabel": row_label,
                    "Stage": "Review phase",
                    "Start": full_assign_dt,
                    "Finish": all_reviews_dt,
                    "HoverText": review_hover,
                }
            )
#UPDATE xmax.
            xmax_dt = max(xmax_dt, all_reviews_dt)

#ADD first review marker if available.
        if pd.notna(first_review_dt):
#APPEND x.
            first_rev_x.append(first_review_dt)
#APPEND y.
            first_rev_y.append(row_label)
#APPEND hover.
            first_rev_hover.append(
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"First review received={first_review_dt.date().isoformat()}"
            )

#RECOMMEND PHASE: all reviews received -> AE recommendation.
        if pd.notna(all_reviews_dt) and pd.notna(ae_rec_dt) and ae_rec_dt > all_reviews_dt:
#BUILD hover.
            rec_hover = (
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"All reviews received={all_reviews_dt.date().isoformat()}<br>"
                f"AE recommendation={_ae_dec_text(r.get('AE_Recommendation',''))}<br>"
                f"AE recommendation date={ae_rec_dt.date().isoformat()}<br>"
                f"EIC decision={_ae_dec_text(r.get('EIC_Decision',''))}<br>"
                f"Final outcome={_ae_dec_text(r.get('FinalDecisionOutcome',''))}"
            )
#APPEND segment.
            segments.append(
                {
                    "PaperRoundLabel": row_label,
                    "Stage": "Recommend to EIC phase",
                    "Start": all_reviews_dt,
                    "Finish": ae_rec_dt,
                    "HoverText": rec_hover,
                }
            )
#UPDATE xmax.
            xmax_dt = max(xmax_dt, ae_rec_dt)

#PUT AE recommendation text inside recommendation phase.
            ae_mid = all_reviews_dt + (ae_rec_dt - all_reviews_dt) / 2
#APPEND x.
            ae_text_x.append(ae_mid)
#APPEND y.
            ae_text_y.append(row_label)
#APPEND text.
            ae_text.append(_ae_dec_text(r.get("AE_Recommendation","")))
#APPEND hover.
            ae_text_hover.append(rec_hover)

#PUT EIC decision text just right of EIC line if available.
        if pd.notna(eic_dec_dt):
#TEXT position.
            text_x = eic_dec_dt + pd.Timedelta(days=5)
#APPEND x.
            eic_text_x.append(text_x)
#APPEND y.
            eic_text_y.append(row_label)
#APPEND text.
            eic_text.append(f"EIC: {_ae_dec_text(r.get('EIC_Decision',''))}")
#APPEND hover.
            eic_text_hover.append(
                f"Paper={r.get('PaperID','')}<br>"
                f"Round={r.get('SubmissionRound','')}<br>"
                f"EIC decision={_ae_dec_text(r.get('EIC_Decision',''))}<br>"
                f"EIC decision date={eic_dec_dt.date().isoformat()}<br>"
                f"Decision letter={decision_letter_dt.date().isoformat() if pd.notna(decision_letter_dt) else 'NA'}<br>"
                f"Final outcome={_ae_dec_text(r.get('FinalDecisionOutcome',''))}"
            )
#UPDATE xmax.
            xmax_dt = max(xmax_dt, text_x)

#BUILD segments dataframe.
    seg_df = pd.DataFrame(segments)

#GUARD no segments.
    if seg_df.empty:
#WARN.
        st.warning("No AE timeline segments were generated for this filter selection.")
#SHOW diagnostic.
        st.dataframe(ae_df, width='stretch')
#STOP.
        st.stop()

#PHASE color map.
    color_map = {
        "Invite phase": "#4C78A8",
        "Review phase": "#54A24B",
        "Recommend to EIC phase": "#B279A2",
    }

#BUILD timeline chart.
    fig = px.timeline(
        seg_df,
        x_start="Start",
        x_end="Finish",
        y="PaperRoundLabel",
        color="Stage",
        color_discrete_map=color_map,
        hover_data={"HoverText": True, "Start": True, "Finish": True},
    )

#REVERSE y-axis for top-down reading.
    fig.update_yaxes(autorange="reversed")

#REMOVE thick outlines.
    fig.update_traces(marker_line_width=0)

#ADD first invite markers.
    if len(first_inv_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=first_inv_x,
                y=first_inv_y,
                mode="markers",
                name="First invite",
                marker=dict(symbol="circle-open", size=9, color="#1F77B4", line=dict(width=2, color="#1F77B4")),
                hovertext=first_inv_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#ADD first review markers.
    if len(first_rev_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=first_rev_x,
                y=first_rev_y,
                mode="markers",
                name="First review received",
                marker=dict(symbol="diamond-open", size=9, color="#E45756", line=dict(width=2, color="#E45756")),
                hovertext=first_rev_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

#ADD AE recommendation text.
    if len(ae_text_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=ae_text_x,
                y=ae_text_y,
                mode="text",
                text=ae_text,
                textposition="middle center",
                name="AE recommendation",
                hovertext=ae_text_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

#ADD EIC decision text to right.
    if len(eic_text_x) > 0:
#ADD trace.
        fig.add_trace(
            go.Scatter(
                x=eic_text_x,
                y=eic_text_y,
                mode="text",
                text=eic_text,
                textposition="middle right",
                name="EIC decision",
                hovertext=eic_text_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

#DRAW EIC decision vertical lines per row.
    for _, r in ae_df.iterrows():
#ROW label.
        row_label = str(r["PaperRoundLabel"])
#GET line date.
        eic_dt = r.get("EIC_DecisionDate", pd.NaT)
#DRAW only if present.
        if pd.notna(eic_dt):
#ADD scatter line segment.
            fig.add_trace(
                go.Scatter(
                    x=[eic_dt, eic_dt],
                    y=[row_label, row_label],
                    mode="lines",
                    line=dict(color="black", width=3),
                    name="EIC decision line",
                    hovertemplate=(
                        f"Paper={r.get('PaperID','')}<br>"
                        f"Round={r.get('SubmissionRound','')}<br>"
                        f"EIC decision={_ae_dec_text(r.get('EIC_Decision',''))}<br>"
                        f"EIC decision date={eic_dt.date().isoformat()}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

#SET layout.
    fig.update_layout(
        title=f"AE timeline for {selected_ae}",
        height=max(520, 160 + len(ae_df) * 38),
        margin=dict(l=10, r=10, t=60, b=65),
        legend_title_text="Phase",
        bargap=0.70,
    )

#SET x-axis date ticks every 21 days.
    fig.update_xaxes(
        title="Date",
        dtick=21 * 24 * 60 * 60 * 1000,
        tickformat="%d %b %Y",
        tickangle=45,
    )

#SHOW chart.
    event = st.plotly_chart(
        fig,
        width='stretch',
        key="ae_plot",
        on_select="rerun"
    )

    if event and "selection" in event and not st.session_state.get("ae_clicked", False):
        pts = event["selection"].get("points", [])
        if pts:
            selected_label = pts[0].get("y")

            st.session_state["selected_paper_from_ae"] = selected_label
            st.session_state["active_tab"] = "Paper timeline"
            st.session_state["from_ae_click"] = True

            st.session_state["ae_clicked"] = True
            st.rerun()

    # ✅ reset after rerun completes
    if st.session_state.get("ae_clicked", False):
        st.session_state["ae_clicked"] = False
#DETAIL table heading.
    st.markdown("#### AE paper-round details")

#TABLE columns.
    cols = [
        "PaperID",
        "SubmissionRound",
        "JournalSection",
        "PaperStatusOnSubmission",
        "DatePaperSubmitted",
        "FirstInviteDate",
        "LastInviteDate",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "AE_Recommendation",
        "EIC_DecisionDate",
        "EIC_Decision",
        "DateDecisionLetterSent",
        "FinalDecisionOutcome",
        "InvitesSent",
        "Accepted",
        "Declined",
        "NoResponse",
        "ReviewsSubmitted",
        "LateReviews",
        "AvgReminders",
        "ReviewerDecisionSet",
        "MaxDisagreement",
        "InvitePhaseDays",
        "ReviewPhaseDays",
        "RecommendPhaseDays",
    ]

#KEEP existing cols only.
    cols = [c for c in cols if c in ae_df.columns]

#SHOW table.
    st.dataframe(ae_df[cols], width='stretch', height=420)

#PAPER STATUS OVERVIEW TAB ONLY.
if selected_tab == "Paper status overview":
#LOCAL import.
    import numpy as np
#LOCAL import.
    import pandas as pd
#LOCAL import.
    import plotly.express as px
#LOCAL import.
    import streamlit as st

#TITLE.
    st.subheader("Paper status overview: where papers stand in the process")

#USE filtered paper dataframe if available.
    try:
#SOURCE.
        _paper_src = paper_f
#FALLBACK.
    except NameError:
#SOURCE fallback.
        _paper_src = paper_df

#COPY paper data.
    paper_status = _paper_src.copy()

#DATE columns used for stage inference.
    date_cols = [
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "EIC_DecisionDate",
        "DateDecisionLetterSent",
    ]

#PARSE date columns.
    for c in date_cols:
#CHECK column.
        if c in paper_status.columns:
#CONVERT to datetime.
            paper_status[c] = pd.to_datetime(paper_status[c], errors="coerce")

#GUARD required columns.
    if "PaperID" not in paper_status.columns or "SubmissionRound" not in paper_status.columns:
#SHOW error.
        st.error("PaperHeader must contain PaperID and SubmissionRound.")
#STOP app.
        st.stop()

#GUARD submission date.
    if "DatePaperSubmitted" not in paper_status.columns:
#SHOW error.
        st.error("PaperHeader must contain DatePaperSubmitted.")
#STOP app.
        st.stop()

#BUILD paper-round key.
    paper_status["PaperRoundKey"] = paper_status["PaperID"].astype(str) + " | round " + paper_status["SubmissionRound"].astype(int).astype(str)

#BUILD all event dates for snapshot range.
    all_event_dates = []
#COLLECT event dates.
    for c in date_cols:
#CHECK column.
        if c in paper_status.columns:
#APPEND non-null dates.
            all_event_dates.append(paper_status[c].dropna())

#COMBINE event dates.
    all_event_dates = pd.concat(all_event_dates) if len(all_event_dates) > 0 else pd.Series(dtype="datetime64[ns]")

#GUARD valid dates.
    if all_event_dates.empty:
#SHOW error.
        st.error("No valid timeline dates found.")
#STOP app.
        st.stop()

#COMPUTE min date.
    min_snapshot = all_event_dates.min()
#COMPUTE max date.
    max_snapshot = all_event_dates.max()

#DEFAULT snapshot around middle-late dataset so some papers are still active.
    default_snapshot = paper_status["DatePaperSubmitted"].dropna().quantile(0.70) + pd.Timedelta(days=90)

#CLAMP default snapshot.
    if pd.isna(default_snapshot):
#FALLBACK default.
        default_snapshot = min_snapshot + (max_snapshot - min_snapshot) / 2

#CLAMP lower.
    if default_snapshot < min_snapshot:
#SET lower.
        default_snapshot = min_snapshot

#CLAMP upper.
    if default_snapshot > max_snapshot:
#SET upper.
        default_snapshot = max_snapshot

#CONTROL row.
    c0, c1, c2 = st.columns([1.2, 1.0, 1.0])

#SNAPSHOT date.
    with c0:
#DATE input.
        snapshot_date = st.date_input(
            "Snapshot / as-of date",
            value=default_snapshot.date(),
            min_value=min_snapshot.date(),
            max_value=max_snapshot.date(),
            key="paper_status_snapshot_date",
        )

#INCLUDE future submitted papers toggle.
    with c1:
#CHECKBOX.
        include_not_submitted = st.checkbox(
            "Include not-yet-submitted papers",
            value=False,
            key="paper_status_include_future",
        )

#AGE bucket size.
    with c2:
#SLIDER.
        age_bucket_months = st.slider(
            "Age bucket size, months",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            key="paper_status_age_bucket_size",
        )

#CONVERT snapshot.
    snapshot_dt = pd.to_datetime(snapshot_date)

#HELPER: event happened by snapshot.
    def _happened(row, col):
#MISSING column.
        if col not in row.index:
#RETURN false.
            return False
#MISSING date.
        if pd.isna(row.get(col, pd.NaT)):
#RETURN false.
            return False
#COMPARE.
        return row.get(col) <= snapshot_dt

#HELPER: decision label.
    def _decision_label(x):
#NORMALIZE.
        x = str(x).strip().lower()
#MAP.
        m = {
            "accept": "Accepted",
            "reject": "Rejected",
            "minor revision": "Minor revision",
            "major revision": "Major revision",
            "submit as new": "Submit as new",
            "resubmit as new": "Submit as new",
            "": "No decision yet",
            "nan": "No decision yet",
            "none": "No decision yet",
        }
#RETURN.
        return m.get(x, str(x).strip().title() if str(x).strip() else "No decision yet")

#HELPER: stage as of snapshot.
    def _stage_as_of(r):
#NOT submitted yet.
        if pd.isna(r.get("DatePaperSubmitted", pd.NaT)) or r.get("DatePaperSubmitted") > snapshot_dt:
#RETURN.
            return "Not yet submitted"
#RESOLVED.
        if _happened(r, "DateDecisionLetterSent"):
#RETURN.
            return "Resolved: decision sent"
#EIC decision made but letter not sent.
        if _happened(r, "EIC_DecisionDate"):
#RETURN.
            return "Waiting: decision letter"
#AE recommendation made but EIC not done.
        if _happened(r, "AE_RecommendationDate"):
#RETURN.
            return "Waiting: EIC decision"
#All reviews in but AE not done.
        if _happened(r, "DateAllReviewsReceived"):
#RETURN.
            return "Waiting: AE recommendation"
#Some reviews received.
        if _happened(r, "DateFirstReviewReceived"):
#RETURN.
            return "In review: partial reviews received"
#Reviewers assigned.
        if _happened(r, "DateReviewersFullyAssigned"):
#RETURN.
            return "In review: waiting for first review"
#Submitted but reviewers not fully assigned.
        return "Waiting: reviewer assignment"

#CREATE stage as of snapshot.
    paper_status["StageAsOf"] = paper_status.apply(_stage_as_of, axis=1)

#CREATE resolution state as of snapshot.
    paper_status["ResolutionStateAsOf"] = np.where(
        paper_status["StageAsOf"].eq("Resolved: decision sent"),
        "Resolved",
        "No decision yet",
    )

#SET not-yet-submitted as separate state.
    paper_status.loc[paper_status["StageAsOf"].eq("Not yet submitted"), "ResolutionStateAsOf"] = "Not yet submitted"

#CREATE decision bucket as of snapshot.
    if "FinalDecisionOutcome" in paper_status.columns:
#USE final decision if resolved.
        paper_status["DecisionBucketAsOf"] = paper_status["FinalDecisionOutcome"].apply(_decision_label)
#ELIF EIC decision exists.
    elif "EIC_Decision" in paper_status.columns:
#USE EIC decision.
        paper_status["DecisionBucketAsOf"] = paper_status["EIC_Decision"].apply(_decision_label)
#ELSE no decision.
    else:
#DEFAULT.
        paper_status["DecisionBucketAsOf"] = "No decision yet"

#FOR unresolved, force no decision.
    paper_status.loc[~paper_status["StageAsOf"].eq("Resolved: decision sent"), "DecisionBucketAsOf"] = "No decision yet"

#DROP not-yet-submitted if unchecked.
    if not include_not_submitted:
#FILTER.
        paper_status = paper_status[~paper_status["StageAsOf"].eq("Not yet submitted")].copy()

#COMPUTE age endpoint.
    paper_status["AgeEndDateAsOf"] = np.where(
        paper_status["StageAsOf"].eq("Resolved: decision sent"),
        paper_status["DateDecisionLetterSent"],
        snapshot_dt,
    )

#CONVERT endpoint to datetime.
    paper_status["AgeEndDateAsOf"] = pd.to_datetime(paper_status["AgeEndDateAsOf"], errors="coerce")

#COMPUTE age days.
    paper_status["AgeDaysAsOf"] = (paper_status["AgeEndDateAsOf"] - paper_status["DatePaperSubmitted"]).dt.days

#CLIP negative age.
    paper_status["AgeDaysAsOf"] = paper_status["AgeDaysAsOf"].clip(lower=0)

#COMPUTE age months.
    paper_status["AgeMonthsAsOf"] = paper_status["AgeDaysAsOf"] / 30.44

#CREATE age buckets dynamically.
    max_age_months = paper_status["AgeMonthsAsOf"].dropna().max()

#FALLBACK max.
    if pd.isna(max_age_months):
#SET fallback.
        max_age_months = age_bucket_months

#CREATE bin upper limit.
    max_bin = int(np.ceil(max_age_months / age_bucket_months) * age_bucket_months + age_bucket_months)

#CREATE bin edges.
    bins = list(range(0, max_bin + age_bucket_months, age_bucket_months))

#ENSURE at least two bins.
    if len(bins) < 2:
#SET fallback bins.
        bins = [0, age_bucket_months]

#CREATE labels.
    labels = [f"{bins[i]}-{bins[i+1]} months" for i in range(len(bins) - 1)]

#CUT age into buckets.
    paper_status["AgeBucketAsOf"] = pd.cut(
        paper_status["AgeMonthsAsOf"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)

#FIX missing age bucket.
    paper_status["AgeBucketAsOf"] = paper_status["AgeBucketAsOf"].replace("nan", "Unknown age")

#CREATE submission month.
    paper_status["SubmissionMonth"] = paper_status["DatePaperSubmitted"].dt.to_period("M").dt.to_timestamp()

#STAGE order.
    stage_order = [
        "Waiting: reviewer assignment",
        "In review: waiting for first review",
        "In review: partial reviews received",
        "Waiting: AE recommendation",
        "Waiting: EIC decision",
        "Waiting: decision letter",
        "Resolved: decision sent",
        "Not yet submitted",
    ]

#FILTER stage order to existing.
    stage_order_existing = [s for s in stage_order if s in paper_status["StageAsOf"].unique().tolist()]

#METRICS.
    m1, m2, m3, m4, m5 = st.columns(5)

#TOTAL.
    m1.metric("Papers in snapshot", f"{len(paper_status):,}")

#NO decision.
    m2.metric("No decision yet", f"{int((paper_status['ResolutionStateAsOf'] == 'No decision yet').sum()):,}")

#RESOLVED.
    m3.metric("Resolved", f"{int((paper_status['ResolutionStateAsOf'] == 'Resolved').sum()):,}")

#WAITING review-related.
    active_review = paper_status["StageAsOf"].isin([
        "In review: waiting for first review",
        "In review: partial reviews received",
    ]).sum()

#METRIC.
    m4.metric("Currently in review", f"{int(active_review):,}")

#MEDIAN age.
    m5.metric("Median age months", f"{paper_status['AgeMonthsAsOf'].median():.1f}")

#SECTION.
    st.markdown("#### Main view: how many papers are in each stage?")

#COUNT by stage.
    stage_counts = (
        paper_status
        .groupby("StageAsOf")
        .size()
        .reset_index(name="PaperCount")
    )

#APPLY order.
    stage_counts["StageAsOf"] = pd.Categorical(stage_counts["StageAsOf"], categories=stage_order, ordered=True)

#SORT.
    stage_counts = stage_counts.sort_values("StageAsOf")

#BUILD stage chart.
    fig_stage = px.bar(
        stage_counts,
        x="StageAsOf",
        y="PaperCount",
        text="PaperCount",
        title=f"Paper count by workflow stage as of {snapshot_dt.date().isoformat()}",
        custom_data=["StageAsOf"],
    )

#FORMAT stage chart.
    fig_stage.update_layout(
        height=500,
        xaxis_title="Workflow stage",
        yaxis_title="Number of papers",
        margin=dict(l=10, r=10, t=60, b=150),
    )

#ROTATE x labels.
    fig_stage.update_xaxes(tickangle=35)

#SHOW chart with selection.
    stage_event = st.plotly_chart(
        fig_stage,
        use_container_width=True,
        key="paper_status_stage_chart",
        on_select="rerun",
    )

#SELECTED stage.
    selected_stage = None

#READ selected stage.
    if stage_event and "selection" in stage_event:
#POINTS.
        pts = stage_event["selection"].get("points", [])
#IF selected.
        if pts:
#GET stage.
            selected_stage = pts[0].get("x")

#SECOND ROW CHARTS.
    left, right = st.columns(2)

#UNRESOLVED stage and age chart.
    with left:
#UNRESOLVED only.
        unresolved_df = paper_status[paper_status["ResolutionStateAsOf"] == "No decision yet"].copy()

#GROUP unresolved.
        unresolved_counts = (
            unresolved_df
            .groupby(["StageAsOf", "AgeBucketAsOf"])
            .size()
            .reset_index(name="PaperCount")
        )

#BUILD chart.
        fig_unresolved = px.bar(
            unresolved_counts,
            x="StageAsOf",
            y="PaperCount",
            color="AgeBucketAsOf",
            text="PaperCount",
            title="No-decision papers by stage and age",
            custom_data=["StageAsOf", "AgeBucketAsOf"],
            category_orders={"StageAsOf": stage_order, "AgeBucketAsOf": labels},
        )

#FORMAT.
        fig_unresolved.update_layout(
            height=480,
            xaxis_title="Current stage",
            yaxis_title="No-decision papers",
            legend_title_text="Age as of snapshot",
            margin=dict(l=10, r=10, t=60, b=150),
        )

#ROTATE.
        fig_unresolved.update_xaxes(tickangle=35)

#SHOW.
        unresolved_event = st.plotly_chart(
            fig_unresolved,
            use_container_width=True,
            key="paper_status_unresolved_chart",
            on_select="rerun",
        )

#RESOLVED decision chart.
    with right:
#RESOLVED only.
        resolved_df = paper_status[paper_status["ResolutionStateAsOf"] == "Resolved"].copy()

#GROUP resolved.
        resolved_counts = (
            resolved_df
            .groupby(["DecisionBucketAsOf", "AgeBucketAsOf"])
            .size()
            .reset_index(name="PaperCount")
        )

#BUILD chart.
        fig_resolved = px.bar(
            resolved_counts,
            x="DecisionBucketAsOf",
            y="PaperCount",
            color="AgeBucketAsOf",
            text="PaperCount",
            title="Resolved papers by final decision and time-to-decision",
            custom_data=["DecisionBucketAsOf", "AgeBucketAsOf"],
            category_orders={"AgeBucketAsOf": labels},
        )

#FORMAT.
        fig_resolved.update_layout(
            height=480,
            xaxis_title="Final decision",
            yaxis_title="Resolved papers",
            legend_title_text="Time to decision",
            margin=dict(l=10, r=10, t=60, b=100),
        )

#ROTATE.
        fig_resolved.update_xaxes(tickangle=25)

#SHOW.
        resolved_event = st.plotly_chart(
            fig_resolved,
            use_container_width=True,
            key="paper_status_resolved_chart",
            on_select="rerun",
        )

#MONTHLY histogram.
    st.markdown("#### Paper volume over time")

#GROUP by month and stage.
    month_counts = (
        paper_status
        .dropna(subset=["SubmissionMonth"])
        .groupby(["SubmissionMonth", "StageAsOf"])
        .size()
        .reset_index(name="PaperCount")
    )

#BUILD month histogram.
    fig_month = px.bar(
        month_counts,
        x="SubmissionMonth",
        y="PaperCount",
        color="StageAsOf",
        title="Paper count by submission month and current stage",
        custom_data=["SubmissionMonth", "StageAsOf"],
        category_orders={"StageAsOf": stage_order},
    )

#FORMAT month chart.
    fig_month.update_layout(
        height=500,
        xaxis_title="Submission month",
        yaxis_title="Number of papers",
        legend_title_text="Stage as of snapshot",
        margin=dict(l=10, r=10, t=60, b=80),
    )

#X-axis every 6 months.
    fig_month.update_xaxes(
        dtick="M6",
        tickformat="%b %Y",
        tickangle=45,
    )

#SHOW selectable month chart.
    month_event = st.plotly_chart(
        fig_month,
        use_container_width=True,
        key="paper_status_month_chart",
        on_select="rerun",
    )

#DETAIL DATAFRAME.
    detail_df = paper_status.copy()

#FILTER from stage chart.
    if selected_stage:
#FILTER stage.
        detail_df = detail_df[detail_df["StageAsOf"] == selected_stage].copy()
#INFO.
        st.info(f"Filtered from main chart: {selected_stage}")

#FILTER from unresolved chart.
    if unresolved_event and "selection" in unresolved_event:
#GET points.
        pts = unresolved_event["selection"].get("points", [])
#IF selected.
        if pts:
#GET custom data.
            cd = pts[0].get("customdata", [])
#IF valid.
            if len(cd) >= 2:
#GET values.
                clicked_stage = cd[0]
#GET age.
                clicked_age = cd[1]
#FILTER.
                detail_df = paper_status[
                    (paper_status["ResolutionStateAsOf"] == "No decision yet")
                    & (paper_status["StageAsOf"] == clicked_stage)
                    & (paper_status["AgeBucketAsOf"] == clicked_age)
                ].copy()
#INFO.
                st.info(f"Filtered unresolved papers: {clicked_stage}, age {clicked_age}")

#FILTER from resolved chart.
    if resolved_event and "selection" in resolved_event:
#GET points.
        pts = resolved_event["selection"].get("points", [])
#IF selected.
        if pts:
#GET custom data.
            cd = pts[0].get("customdata", [])
#IF valid.
            if len(cd) >= 2:
#GET decision.
                clicked_decision = cd[0]
#GET age.
                clicked_age = cd[1]
#FILTER.
                detail_df = paper_status[
                    (paper_status["ResolutionStateAsOf"] == "Resolved")
                    & (paper_status["DecisionBucketAsOf"] == clicked_decision)
                    & (paper_status["AgeBucketAsOf"] == clicked_age)
                ].copy()
#INFO.
                st.info(f"Filtered resolved papers: {clicked_decision}, age {clicked_age}")

#FILTER from month chart.
    if month_event and "selection" in month_event:
#GET points.
        pts = month_event["selection"].get("points", [])
#IF selected.
        if pts:
#GET custom data.
            cd = pts[0].get("customdata", [])
#IF valid.
            if len(cd) >= 2:
#GET month.
                clicked_month = pd.to_datetime(cd[0], errors="coerce")
#GET stage.
                clicked_month_stage = cd[1]
#FILTER.
                detail_df = paper_status[
                    (paper_status["SubmissionMonth"] == clicked_month)
                    & (paper_status["StageAsOf"] == clicked_month_stage)
                ].copy()
#INFO.
                st.info(f"Filtered papers submitted in {clicked_month.strftime('%b %Y')} at stage: {clicked_month_stage}")

#MANUAL filters.
    st.markdown("#### Drill down")

#FILTER layout.
    f1, f2, f3 = st.columns(3)

#STAGE filter.
    with f1:
#MULTISELECT.
        manual_stage = st.multiselect(
            "Stage",
            stage_order_existing,
            default=[],
            key="paper_status_manual_stage",
        )

#RESOLUTION filter.
    with f2:
#MULTISELECT.
        manual_resolution = st.multiselect(
            "Resolution state",
            ["No decision yet", "Resolved", "Not yet submitted"],
            default=[],
            key="paper_status_manual_resolution",
        )

#AGE filter.
    with f3:
#MULTISELECT.
        manual_age = st.multiselect(
            "Age bucket",
            labels + ["Unknown age"],
            default=[],
            key="paper_status_manual_age",
        )

#APPLY manual stage.
    if manual_stage:
#FILTER.
        detail_df = detail_df[detail_df["StageAsOf"].isin(manual_stage)]

#APPLY manual resolution.
    if manual_resolution:
#FILTER.
        detail_df = detail_df[detail_df["ResolutionStateAsOf"].isin(manual_resolution)]

#APPLY manual age.
    if manual_age:
#FILTER.
        detail_df = detail_df[detail_df["AgeBucketAsOf"].isin(manual_age)]

#TABLE columns.
    cols = [
        "PaperRoundKey",
        "PaperID",
        "SubmissionRound",
        "JournalSection",
        "HandlingAssociateEditorID",
        "HandlingEIC_ID",
        "StageAsOf",
        "ResolutionStateAsOf",
        "DecisionBucketAsOf",
        "AgeMonthsAsOf",
        "AgeBucketAsOf",
        "DatePaperSubmitted",
        "DateReviewersFullyAssigned",
        "DateFirstReviewReceived",
        "DateAllReviewsReceived",
        "AE_RecommendationDate",
        "AE_Recommendation",
        "EIC_DecisionDate",
        "EIC_Decision",
        "DateDecisionLetterSent",
        "FinalDecisionOutcome",
        "TotalTime_SubmissionToDecision_Days",
    ]

#KEEP existing columns.
    cols = [c for c in cols if c in detail_df.columns]

#SHOW count.
    st.caption(f"Showing {len(detail_df):,} paper-rounds after filters.")

#SHOW table.
    st.dataframe(
        detail_df[cols].sort_values(["StageAsOf", "AgeMonthsAsOf"], ascending=[True, False]),
        use_container_width=True,
        height=420,
    )