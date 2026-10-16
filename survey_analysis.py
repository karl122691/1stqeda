"""Audit and summarize the anonymous Cor Jesu survey without inventing responses.

Study responses are recorded scores: 8 represents the open-ended answer 8+.
Quantiles use linear interpolation, equivalent to spreadsheet INC functions.
"""

from decimal import Decimal, InvalidOperation
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("respondent_id", "year_level", "study_hours", "commute_minutes")
YEAR_LEVELS = ("Freshman", "Sophomore", "Junior", "Senior")
STUDY_BOUNDARIES = np.array([-0.5, 1.5, 3.5, 5.5, 7.5, 9.5])
STUDY_LABELS = ["0–1", "2–3", "4–5", "6–7", "8–9 (score 8 = 8+)"]
GROUP_COLUMNS = [
    "class_interval", "lower_boundary", "upper_boundary", "midpoint",
    "frequency", "relative_frequency", "percent", "cumulative_frequency",
    "cumulative_relative_frequency",
]
NUMERIC_COLUMNS = ("study_hours", "commute_minutes")


def _whole_number(value):
    """Parse a finite whole number; Decimal avoids merging distinct large IDs."""
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            return None
        if not math.isfinite(float(number)):
            return None
        return int(number)
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _frequency_columns(counts, n):
    counts = np.asarray(counts, dtype=int)
    relative = counts / n if n else np.zeros(len(counts))
    return {
        "frequency": counts,
        "relative_frequency": relative,
        "percent": relative * 100,
        "cumulative_frequency": counts.cumsum(),
        "cumulative_relative_frequency": relative.cumsum(),
    }


def _grouped_table(values, boundaries, labels):
    """Use explicit equal-width classes, retaining classes with zero responses."""
    groups = pd.cut(values, bins=boundaries, labels=labels, include_lowest=True)
    counts = pd.Series(groups).value_counts(sort=False).reindex(labels, fill_value=0)
    return pd.DataFrame({
        "class_interval": labels,
        "lower_boundary": boundaries[:-1],
        "upper_boundary": boundaries[1:],
        "midpoint": (boundaries[:-1] + boundaries[1:]) / 2,
        **_frequency_columns(counts.to_numpy(), len(values)),
    })


def _statistics(values):
    values = np.asarray(values, dtype=float)
    q1, q3, p10, p90 = np.quantile(values, [.25, .75, .1, .9], method="linear")
    distinct, counts = np.unique(values, return_counts=True)
    modes = (
        ", ".join(f"{x:g}" for x in distinct[counts == counts.max()])
        if counts.max() > 1 else "No mode (all values occur once)"
    )
    iqr = q3 - q1
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "modes": modes,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "sample_variance": float(np.var(values, ddof=1)) if len(values) > 1 else np.nan,
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "p10": float(p10),
        "p90": float(p90),
        "lower_fence": float(q1 - 1.5 * iqr),
        "upper_fence": float(q3 + 1.5 * iqr),
    }


def _shape(values, capped=False):
    values = np.asarray(values, dtype=float)
    caveat = (
        " This describes recorded scores; the 8+ cap can hide the upper tail of actual hours."
        if capped else " This describes this sample, not the whole student population."
    )
    if np.ptp(values) == 0:
        return "Constant: all recorded values are equal; skewness is undefined." + caveat
    if len(values) < 3:
        return "Too few observations to assess distribution shape reliably." + caveat
    centered = values - values.mean()
    skewness = float(np.mean((centered / np.sqrt(np.mean(centered ** 2))) ** 3))
    if skewness > .5:
        description = "The recorded distribution shows right skew"
    elif skewness < -.5:
        description = "The recorded distribution shows left skew"
    else:
        description = "The recorded distribution has no strong moment skew"
    return (
        f"{description} (moment skewness {skewness:.3f}; descriptive threshold ±0.5). "
        "Use the plot to check clusters and unusual values; this is not a normality test."
        + caveat
    )


def _make_figures(valid, tables):
    n = len(valid)
    figures = {}
    color = "#225e7a"
    has_cap = bool(valid["study_hours"].eq(8).any())
    study_title = "Recorded study scores" if has_cap else "Daily study hours"
    study_units = "Recorded study score (hours per typical school day; 8 means 8+)" if has_cap else "Study time outside classes (hours per typical school day)"

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    table = tables["q1"]
    bars = ax.bar(table["category"], table["frequency"], color=color)
    ax.bar_label(bars, padding=3)
    ax.set(title=f"Q1: Year-level distribution (n = {n})",
           xlabel="Year-level category", ylabel="Frequency (records)")
    ax.set_xticks(range(len(YEAR_LEVELS)), YEAR_LEVELS)
    ax.set_ylim(0, max(1, int(table["frequency"].max())) * 1.18)
    ax.yaxis.get_major_locator().set_params(integer=True)
    figures["q1_year_level_bar"] = fig

    table = tables["q2"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.hist(valid["study_hours"], bins=STUDY_BOUNDARIES, color=color, edgecolor="white")
    ax.set(title=f"Q2: {study_title} (n = {n})",
           xlabel=study_units,
           ylabel="Frequency (records)")
    ax.set_xticks(table["midpoint"], table["class_interval"])
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.text(.5, -.025, "Equal class width = 2 hours. " + ("The final class contains capped score 8, not measured hours 8–9." if has_cap else "Zero-frequency classes are retained."),
             ha="center", fontsize=9)
    figures["q2_study_histogram"] = fig

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    midpoints = table["midpoint"].to_numpy()
    ax.plot(np.r_[midpoints[0] - 2, midpoints, midpoints[-1] + 2],
            np.r_[0, table["frequency"], 0], marker="o", color=color)
    ax.set(title=f"Q2: Study-hour frequency polygon (n = {n})",
           xlabel="Recorded score class midpoint (hours; score 8 means 8+)" if has_cap else "Study-hour class midpoint (hours)",
           ylabel="Frequency (records)")
    ax.set_xticks(midpoints)
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.text(.5, -.025, ("Midpoint 8.5 represents the capped-score class. " if has_cap else "") + "Zero endpoints close the polygon.",
             ha="center", fontsize=9)
    figures["q2_frequency_polygon"] = fig

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(STUDY_BOUNDARIES, np.r_[0, table["cumulative_frequency"]], marker="o", color=color)
    ax.set(title=f"Q2: Cumulative study-hour frequency (n = {n})",
           xlabel="Recorded score class boundary (hours; score 8 means 8+)" if has_cap else "Study-hour class boundary (hours)",
           ylabel="Cumulative frequency (records)", ylim=(0, max(1, n) * 1.08))
    ax.set_xticks(STUDY_BOUNDARIES)
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.text(.5, -.025, "Starts at zero below the first class. " + ("Boundary 9.5 ends a score class, not actual study hours." if has_cap else "The final cumulative frequency equals the sample size."),
             ha="center", fontsize=9)
    figures["q2_ogive"] = fig

    table = tables["q3"]
    boundaries = np.r_[table["lower_boundary"], table["upper_boundary"].iloc[-1]]
    width = table["upper_boundary"].iloc[0] - table["lower_boundary"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.hist(valid["commute_minutes"], bins=boundaries, color=color, edgecolor="white")
    ax.set(title=f"Q3: One-way commute time (n = {n}; class width = {width:g} min)",
           xlabel="Recorded one-way commute time (minutes)", ylabel="Frequency (records)")
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.text(.5, -.025, "Whole-minute observations use continuous boundaries ±0.5 minute; −0.5 includes recorded zero.",
             ha="center", fontsize=9)
    figures["q3_commute_histogram"] = fig

    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    ax.boxplot(valid["commute_minutes"].to_numpy(dtype=float), vert=False, whis=1.5,
               patch_artist=True, boxprops={"facecolor": "#b9d8e5"},
               medianprops={"color": "#a24724", "linewidth": 2})
    ax.set(title=f"Q3: One-way commute time boxplot (n = {n})",
           xlabel="Recorded one-way commute time (minutes)", ylabel="Dataset")
    ax.set_yticks([1], ["Records"])
    fig.text(.5, -.025, "Whiskers reach the furthest observations within the 1.5 × IQR fences; flagged points remain in the data.",
             ha="center", fontsize=9)
    figures["q3_commute_boxplot"] = fig
    return figures


def analyze(csv_path, output_dir=None):
    """Return audited data, tables, descriptive statistics, and six chart figures.

    Input uses REQUIRED_COLUMNS. Raw cell strings are retained unchanged; a
    stripped copy is used for validation. Invalid rows, including *every* row
    sharing a numeric respondent ID, are excluded and recorded with reasons.
    Equal answer triples belonging to different IDs are legitimate responses.
    No files are written unless output_dir is supplied. Outliers are retained.
    """
    issues = []
    try:
        raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raw = pd.DataFrame(columns=REQUIRED_COLUMNS)
        issues.append(f"Could not read survey CSV: {exc}")

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in raw.columns]
    reasons = [[] for _ in range(len(raw))]
    cleaned = []
    id_rows = {}
    if missing_columns:
        message = "Missing required columns: " + ", ".join(missing_columns)
        issues.append(message)
        for row_reasons in reasons:
            row_reasons.append(message)
    else:
        for position, (_, row) in enumerate(raw.iterrows()):
            cells = {name: str(row[name]).strip() for name in REQUIRED_COLUMNS}
            for name, value in cells.items():
                if not value:
                    reasons[position].append(f"{name}: missing value")

            respondent_id = _whole_number(cells["respondent_id"])
            if respondent_id is None or respondent_id <= 0:
                if cells["respondent_id"]:
                    reasons[position].append("respondent_id: must be a positive whole number")
            else:
                id_rows.setdefault(respondent_id, []).append(position)

            year = cells["year_level"]
            if year and year not in YEAR_LEVELS:
                reasons[position].append("year_level: outside the four year-level categories in the study population")

            study_text = cells["study_hours"]
            study = 8 if study_text in ("8+", "8 or more") else _whole_number(study_text)
            if study is None or not 0 <= study <= 8:
                if study_text:
                    reasons[position].append("study_hours: use whole scores 0–7 or 8 / 8+ / 8 or more")

            commute = _whole_number(cells["commute_minutes"])
            if commute is None or commute < 0:
                if cells["commute_minutes"]:
                    reasons[position].append("commute_minutes: must be a finite nonnegative whole number")
            cleaned.append({"respondent_id": respondent_id, "year_level": year,
                            "study_hours": study, "commute_minutes": commute})

        for respondent_id, positions in id_rows.items():
            if len(positions) > 1:
                for position in positions:
                    reasons[position].append(f"respondent_id: duplicate numeric ID {respondent_id}; all copies excluded")

    excluded_positions = [i for i, row_reasons in enumerate(reasons) if row_reasons]
    excluded = raw.iloc[excluded_positions].copy()
    excluded["source_row"] = [i + 2 for i in excluded_positions]
    excluded["reason"] = ["; ".join(reasons[i]) for i in excluded_positions]
    excluded.reset_index(drop=True, inplace=True)
    valid = pd.DataFrame(
        [row for i, row in enumerate(cleaned) if not reasons[i]], columns=REQUIRED_COLUMNS
    )
    n = len(valid)
    if excluded_positions:
        issues.append(f"Excluded {len(excluded_positions)} of {len(raw)} submitted rows; inspect excluded rows and reasons.")
    if n == 0:
        issues.append("No valid responses are available. Collect or correct actual survey data before interpreting results.")
    elif n == 1:
        issues.append("Only one valid response: sample variance and sample standard deviation are undefined.")

    topcoded_count = int((valid["study_hours"] == 8).sum())
    if topcoded_count:
        issues.append(
            "Study hours are recorded capped scores: every 8 / 8+ / 8 or more is coded 8, meaning at least 8 hours. "
            f"Valid top-coded responses: {topcoded_count}. Study summaries describe these scores; "
            "the actual mean and upper tail cannot be recovered when capped responses occur."
        )
    elif n:
        issues.append("All study-hour responses are finite whole-hour values; no open-ended answer occurs.")
    study_labels = STUDY_LABELS if topcoded_count else [f"{2 * i}–{2 * i + 1}" for i in range(5)]
    year_counts = valid["year_level"].value_counts().reindex(YEAR_LEVELS, fill_value=0)
    tables = {
        "q1": pd.DataFrame({"category": YEAR_LEVELS, **_frequency_columns(year_counts.to_numpy(), n)}),
        "q2": _grouped_table(valid["study_hours"], STUDY_BOUNDARIES, study_labels),
        "q3": pd.DataFrame(columns=GROUP_COLUMNS),
    }
    stats = pd.DataFrame(columns=NUMERIC_COLUMNS)
    stats.index.name = "metric"
    shapes = {"q2": "No valid observations.", "q3": "No valid observations."}
    outliers = {name: [] for name in NUMERIC_COLUMNS}
    figures = {}
    if n:
        max_minutes = int(valid["commute_minutes"].max())
        target_classes = max(1, math.ceil(1 + math.log2(n)))
        width = max(1, math.ceil((max_minutes + 1) / target_classes))
        class_count = (max_minutes // width) + 1
        boundaries = np.arange(class_count + 1, dtype=float) * width - .5
        labels = [f"{i * width}–{(i + 1) * width - 1}" for i in range(class_count)]
        tables["q3"] = _grouped_table(valid["commute_minutes"], boundaries, labels)
        issues.append(
            f"Q3 uses equal-width classes of {width} minute(s), starting at recorded zero; "
            f"target class count = ceil(1 + log2(n)) = {target_classes}. "
            "Real boundaries are whole-minute limits ±0.5; the −0.5 boundary includes zero, not negative travel."
        )
        stats = pd.DataFrame({name: _statistics(valid[name]) for name in NUMERIC_COLUMNS})
        stats.index.name = "metric"
        for name in NUMERIC_COLUMNS:
            mask = ((valid[name] < stats.at["lower_fence", name]) |
                    (valid[name] > stats.at["upper_fence", name]))
            outliers[name] = valid.loc[mask, "respondent_id"].tolist()
        shapes = {"q2": _shape(valid["study_hours"], capped=bool(topcoded_count)),
                  "q3": _shape(valid["commute_minutes"])}
        figures = _make_figures(valid, tables)

    result = {"raw": raw, "valid": valid, "excluded": excluded, "issues": issues,
              "n": n, "tables": tables, "statistics": stats, "outliers": outliers,
              "topcoded_count": topcoded_count, "shapes": shapes, "figures": figures}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        raw.to_csv(destination / "raw_responses.csv", index=False)
        valid.to_csv(destination / "valid_responses.csv", index=False)
        excluded.to_csv(destination / "excluded_responses.csv", index=False)
        for name, table in tables.items():
            table.to_csv(destination / f"{name}_frequency_table.csv", index=False)
        stats.to_csv(destination / "descriptive_statistics.csv")
        pd.DataFrame({"issue_or_note": issues}).to_csv(destination / "validation_notes.csv", index=False)
        pd.DataFrame([
            {"variable": name, "respondent_id": respondent_id}
            for name, ids in outliers.items() for respondent_id in ids
        ], columns=["variable", "respondent_id"]).to_csv(destination / "outlier_flags.csv", index=False)
        for name, fig in figures.items():
            fig.savefig(destination / f"{name}.png", dpi=180, bbox_inches="tight")
    return result
