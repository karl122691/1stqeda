"""Render a three-page analysis, optionally preceded by a personal cover page.

This module only formats results from ``survey_analysis.analyze``. It never
creates respondents or supplies missing collection details.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import re
import tempfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text


_WIDTH, _HEIGHT = 8.2677, 11.6929
_INK = "#203448"
_ACCENT = "#176b79"
_MUTED = "#536477"
_REQUIRED = (
    "population", "rounding_rule", "decision_question", "recommendation",
    "vulnerable_assumption", "next_evidence",
)
_REAL_REQUIRED = (
    "sampling_method", "selection_procedure", "collection_start", "collection_end",
)
_LIMITS = {
    "population": 250, "decision_question": 330, "selection_procedure": 750,
    "rounding_rule": 240, "recommendation": 650,
    "vulnerable_assumption": 350, "next_evidence": 350,
    "collection_note": 300,
}
_COVER_DEFAULTS = {
    "institution": "Cor Jesu College",
    "course": "Engineering Data Analysis",
    "instructor": "Engr. Jamie Eduardo Rosal, MSCpE",
}
_COVER_LIMITS = {
    "student_name": 120, "program": 180, "year_section": 120,
    "course": 120, "course_code": 60, "instructor": 140,
    "submission_date": 80, "institution": 120,
}
_PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|TBC|placeholder|YYYY|insert here|fill in|your answer)\b"
    r"|<[^>]+>|\[[^\]]*(?:term|date|insert|enter|name)[^\]]*\]",
    re.IGNORECASE,
)
_FIGURES = (
    ("q1_year_level_bar", "Year-level frequencies"),
    ("q2_study_histogram", "Recorded daily study hours"),
    ("q2_frequency_polygon", "Study-hour frequency polygon"),
    ("q2_ogive", "Study-hour cumulative frequencies"),
    ("q3_commute_histogram", "Recorded one-way commute times"),
    ("q3_commute_boxplot", "Commute distribution and outliers"),
)
_METRICS = (
    ("mean", "Mean"), ("median", "Median"), ("modes", "Mode(s)"),
    ("minimum", "Minimum"), ("maximum", "Maximum"), ("range", "Range"),
    ("sample_variance", "Sample variance (squared units)"),
    ("sample_sd", "Sample standard deviation"),
    ("q1", "First quartile (Q1)"), ("q3", "Third quartile (Q3)"),
    ("iqr", "Interquartile range (IQR)"),
    ("p10", "10th percentile (P10)"), ("p90", "90th percentile (P90)"),
    ("lower_fence", "Lower outlier fence"), ("upper_fence", "Upper outlier fence"),
)


def _documented_survey(metadata: dict) -> bool:
    return str(metadata.get("report_type", "descriptive")).strip().lower() == "survey"


def _cover_metadata(metadata: dict) -> dict | None:
    if "cover" not in metadata:
        return None
    supplied = metadata["cover"]
    return {**_COVER_DEFAULTS, **supplied} if isinstance(supplied, dict) else None


def report_readiness(result: dict, metadata: dict) -> list[str]:
    """Return specific blockers; an empty list means report export can proceed.

    Descriptive reports summarize the available dataset. Documented survey
    reports additionally require the actual sampling method, selection details,
    and completed collection dates.
    """
    problems = []
    if "cover" in metadata:
        if not isinstance(metadata["cover"], dict):
            problems.append("metadata['cover'] must be a dictionary when a cover page is requested.")
        else:
            cover = _cover_metadata(metadata)
            student_name = cover.get("student_name", "")
            if not isinstance(student_name, str) or not student_name.strip():
                problems.append("Complete metadata['cover']['student_name'] before exporting a personal cover page.")
            for key, limit in _COVER_LIMITS.items():
                value = cover.get(key, "")
                if not isinstance(value, str):
                    problems.append(f"metadata['cover']['{key}'] must be text when supplied.")
                elif len(value) > limit:
                    problems.append(f"Shorten metadata['cover']['{key}'] to at most {limit} characters.")
                elif _PLACEHOLDER.search(value):
                    problems.append(f"Replace the template text in metadata['cover']['{key}'].")
    try:
        n = int(result.get("n", 0))
    except (TypeError, ValueError):
        n = 0
    if n < 30:
        problems.append(f"At least 30 valid records are required; only {n} are available.")
    report_type = str(metadata.get("report_type", "descriptive")).strip().lower()
    if report_type not in {"descriptive", "survey"}:
        problems.append("metadata['report_type'] must be 'descriptive' or 'survey'.")
    documented_survey = _documented_survey(metadata)
    required = _REQUIRED + _REAL_REQUIRED if documented_survey else _REQUIRED
    for key in required:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"n/a", "na", "none", "not collected"}:
            problems.append(f"Complete metadata['{key}'] with the required details.")
        elif _PLACEHOLDER.search(value):
            problems.append(f"Replace the template text in metadata['{key}'].")
    collection_note = metadata.get("collection_note", "")
    if not isinstance(collection_note, str):
        problems.append("metadata['collection_note'] must be text when supplied.")
    elif _PLACEHOLDER.search(collection_note):
        problems.append("Replace the template text in metadata['collection_note'].")
    for key, limit in _LIMITS.items():
        value = metadata.get(key, "")
        if isinstance(value, str) and len(value) > limit:
            problems.append(f"Shorten metadata['{key}'] to at most {limit} characters for the three-page report.")
    if documented_survey:
        if str(metadata.get("sampling_method", "")).strip().lower() not in {"random", "systematic", "stratified", "cluster", "convenience"}:
            problems.append("sampling_method must be random, systematic, stratified, cluster, or convenience, and must describe the method actually used.")
        parsed = {}
        for key in ("collection_start", "collection_end"):
            value = metadata.get(key, "")
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError
                parsed[key] = date.fromisoformat(value)
                if parsed[key] > date.today():
                    problems.append(f"{key} is in the future; enter the completed collection date.")
            except (ValueError, TypeError):
                problems.append(f"metadata['{key}'] must be a completed date in YYYY-MM-DD format.")
        if len(parsed) == 2 and parsed["collection_end"] < parsed["collection_start"]:
            problems.append("collection_end must not precede collection_start.")
    if "valid" not in result or len(result["valid"]) != n:
        problems.append("The valid-row count does not match result['n']; rerun the analysis.")
    for key in ("raw", "excluded", "statistics", "outliers", "shapes"):
        if key not in result:
            problems.append(f"Missing analysis result '{key}'; rerun the analysis.")
    for key in ("q1", "q2", "q3"):
        table = result.get("tables", {}).get(key)
        expected = {"category" if key == "q1" else "class_interval", "frequency", "percent", "cumulative_frequency"}
        if table is None or not expected.issubset(table.columns):
            problems.append(f"Missing or incomplete {key} frequency table; rerun the analysis.")
        elif len(table) > (10 if key != "q3" else 8):
            problems.append(f"The {key} table has too many classes for this three-page layout; use at most {'8' if key == 'q3' else '10'} classes.")
    statistics = result.get("statistics")
    if statistics is not None:
        expected_metrics = {key for key, _ in _METRICS}
        if not expected_metrics.issubset(statistics.index) or not {"study_hours", "commute_minutes"}.issubset(statistics.columns):
            problems.append("The descriptive-statistics table is incomplete; rerun the analysis.")
    for key, _ in _FIGURES:
        if key not in result.get("figures", {}):
            problems.append(f"Missing figure '{key}'; rerun the analysis.")
    return problems


class _Page:
    def __init__(self, number: int, title: str, *, decorated=True):
        self.fig = plt.figure(figsize=(_WIDTH, _HEIGHT), facecolor="white")
        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        if not decorated:
            return
        self.text(.45, .35, 7.36, .32, title, size=17, minimum=17, weight="bold")
        label = "COR JESU COLLEGE | DESCRIPTIVE STATISTICS REPORT"
        self.text(.45, .85, 7.36, .27, label, size=10, minimum=10,
                  weight="bold", color=_ACCENT)
        self.fig.add_artist(plt.Line2D([.45 / _WIDTH, 7.82 / _WIDTH], [1 - 1.22 / _HEIGHT] * 2,
                                      transform=self.fig.transFigure, color="#b9cbd1", linewidth=.8))
        footer = "Descriptive results for this dataset; population conclusions require appropriate sampling."
        self.text(.45, 11.12, 6.8, .22, footer, size=7.5, minimum=7.5, color=_MUTED)
        self.text(7.25, 11.12, .55, .22, f"{number} / 3", size=8, minimum=8, color=_MUTED)

    def _wrap(self, text: str, width: float, size: float, weight: str) -> str:
        font = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        pixel_width = width * self.fig.dpi
        lines = []
        for paragraph in str(text).split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                measured = self.renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
                if measured <= pixel_width:
                    line = candidate
                else:
                    if line:
                        lines.append(line)
                    if self.renderer.get_text_width_height_descent(word, font, ismath=False)[0] > pixel_width:
                        raise ValueError("A report field contains an unbroken word too wide to print; shorten it.")
                    line = word
            lines.append(line)
        return "\n".join(lines)

    def text(self, x, y, width, height, content, *, size=9, minimum=8.3, weight="normal", color=_INK, align="left"):
        candidate = size
        while candidate >= minimum - .01:
            wrapped = self._wrap(str(content), width, candidate, weight)
            anchor_x = x + (width / 2 if align == "center" else width if align == "right" else 0)
            artist = self.fig.text(anchor_x / _WIDTH, 1 - y / _HEIGHT, wrapped, va="top", ha=align,
                                   fontsize=candidate, fontfamily="DejaVu Sans", fontweight=weight,
                                   color=color, linespacing=1.2)
            bounds = artist.get_window_extent(self.renderer)
            if bounds.height <= height * self.fig.dpi + 1 and bounds.width <= width * self.fig.dpi + 1:
                return
            artist.remove()
            candidate -= .2
        raise ValueError(f"Report text exceeds the three-page layout. Shorten this field or narrative: {str(content)[:95]!r}")

    def heading(self, y, title):
        self.text(.45, y, 7.36, .28, title, size=11, minimum=11, weight="bold", color=_ACCENT)

    def table(self, x, y, width, height, headers, rows, col_widths=None, fontsize=8.2):
        ax = self.fig.add_axes([x / _WIDTH, 1 - (y + height) / _HEIGHT, width / _WIDTH, height / _HEIGHT])
        ax.set_axis_off()
        table = ax.table(cellText=rows, colLabels=headers, colWidths=col_widths,
                         cellLoc="right", colLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d7e1e5")
            cell.set_linewidth(.45)
            cell.PAD = .05
            cell.set_facecolor("#e5f0f2" if row == 0 else ("#f4f7f8" if row % 2 == 0 else "white"))
            cell.get_text().set_color(_INK)
            if row == 0:
                cell.get_text().set_weight("bold")
            if col == 0:
                cell.get_text().set_ha("left")
            available = width * (col_widths[col] if col_widths else 1 / len(headers)) - .09
            cell.get_text().set_text(self._wrap(cell.get_text().get_text(), available, fontsize,
                                                "bold" if row == 0 else "normal"))
        self.canvas.draw()
        for cell in table.get_celld().values():
            text_bounds = cell.get_text().get_window_extent(self.renderer)
            cell_bounds = cell.get_window_extent(self.renderer)
            if text_bounds.height > cell_bounds.height - 1 or text_bounds.width > cell_bounds.width - 1:
                raise ValueError("A frequency/statistics table does not fit legibly. Shorten long labels or reduce the number of classes.")

    def chart(self, x, y, width, height, figure, number, caption, *, capped=False):
        self.text(x, y, width, .22, f"Figure {number}. {caption}", size=8.7, minimum=8.3,
                  weight="bold", color=_ACCENT)
        raster = _chart_image(figure, width, height - .25, number, capped=capped)
        ax = self.fig.add_axes([x / _WIDTH, 1 - (y + height) / _HEIGHT, width / _WIDTH, (height - .25) / _HEIGHT])
        ax.imshow(raster, aspect="equal")
        ax.set_axis_off()


def _chart_image(figure, width, height, number, *, capped=False):
    """Render at the destination size, restoring the caller's figure afterward."""
    old_size = figure.get_size_inches().copy()
    texts = [(item, item.get_fontsize(), item.get_text(), item.get_visible(),
              item.get_rotation(), item.get_ha()) for item in figure.findobj(Text)]
    positions = [(axis, axis.get_position().frozen()) for axis in figure.axes]
    original_y_ticks = [(axis, axis.get_yticks(), [item.get_text() for item in axis.get_yticklabels()])
                        for axis in figure.axes]
    layout = figure.get_layout_engine()
    try:
        figure.set_layout_engine(None)
        figure.set_size_inches(width, height, forward=False)
        for item, *_ in texts:
            item.set_fontsize(8)
        for item in figure.texts:
            # Original plot footnotes are stated in the surrounding report text.
            item.set_visible(False)
        for axis in figure.axes:
            original_title = axis.get_title()
            sample = re.search(r"n\s*=\s*\d+", original_title)
            axis.set_title(sample.group(0) if sample else "Valid records")
            axis.title.set_fontsize(8.2)
            axis.set_xlabel({
                1: "Year-level category",
                2: "Recorded study score (hours; 8 = 8+)" if capped else "Study time (hours per school day)",
                3: "Score-class midpoint (hours)" if capped else "Class midpoint (hours)",
                4: "Score-class boundary (hours)" if capped else "Class boundary (hours)",
                5: "One-way commute (minutes)", 6: "One-way commute (minutes)",
            }[number])
            axis.set_ylabel("Cumulative\ncount (records)" if number == 4 else ("" if number == 6 else "Count (records)"))
            if number == 6:
                axis.set_yticks([1], ["Records"])
            if number == 1:
                for item in axis.get_xticklabels():
                    item.set_rotation(22)
                    item.set_ha("right")
            if number == 2 and capped:
                labels = axis.get_xticklabels()
                for item in labels:
                    if "8" in item.get_text() and "score" in item.get_text():
                        item.set_text("8-9 (8+)")
                # Fixed labels must be updated on the locator as well as artists.
                axis.set_xticks(axis.get_xticks(), [item.get_text() for item in labels])
            # Insets leave room for labels even when four charts share a page.
            axis.set_position([.18, .29, .78, .54] if height < 2.1 else [.18, .22, .78, .67])
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=180, facecolor="white")
        buffer.seek(0)
        return mpimg.imread(buffer)
    finally:
        figure.set_size_inches(old_size, forward=False)
        for item, fontsize, content, visible, rotation, alignment in texts:
            item.set_fontsize(fontsize)
            item.set_text(content)
            item.set_visible(visible)
            item.set_rotation(rotation)
            item.set_ha(alignment)
        # Replacing a fixed tick formatter needs explicit restoration.
        if number == 2 and capped:
            for axis in figure.axes:
                labels = [content for item, _, content, *_ in texts if item in axis.get_xticklabels()]
                if len(labels) == len(axis.get_xticks()):
                    axis.set_xticks(axis.get_xticks(), labels)
        if number == 6:
            for axis, ticks, labels in original_y_ticks:
                axis.set_yticks(ticks, labels)
        for axis, position in positions:
            axis.set_position(position)
        figure.set_layout_engine(layout)


def _frequency_rows(table, label):
    return [[str(row[label]), str(int(row["frequency"])), f"{float(row['percent']):.1f}%",
             str(int(row["cumulative_frequency"]))] for _, row in table.iterrows()]


def _value(value):
    if isinstance(value, str):
        return "None (all unique)" if value.startswith("No mode") else value
    return f"{float(value):,.2f}"


def _outliers(result):
    counts = [len(result["outliers"].get(key, [])) for key in ("study_hours", "commute_minutes")]
    summary = f"The 1.5 x IQR rule flags {counts[0]} study-hour and {counts[1]} commute record(s). "
    valid = result["valid"].set_index("respondent_id")
    for key, label, unit in (("study_hours", "Study-hour", "h"), ("commute_minutes", "Commute", "min")):
        identifiers = result["outliers"].get(key, [])
        if 0 < len(identifiers) <= 3:
            details = [f"ID {identifier} = {float(valid.loc[identifier, key]):g} {unit}" for identifier in identifiers]
            summary += f"{label} flags: " + "; ".join(details) + ". "
    summary += "Flagged values are retained for review."
    if any(float(valid.loc[identifier, "commute_minutes"]) == 0
           for identifier in result["outliers"].get("commute_minutes", [])):
        summary += " The 0-minute value is allowed by the survey rule and is retained."
    return summary


def _shape_summary(description):
    # The engine's following sentences repeat caveats already printed here.
    return description.split(". ", 1)[0].rstrip(".") + "."


def _cover_page(cover):
    page = _Page(0, "", decorated=False)
    left, width = .70, _WIDTH - 1.40

    def line(y, height, value, *, size=11, minimum=9.5, weight="normal", color=_INK):
        if value.strip():
            page.text(left, y, width, height, value.strip(), size=size, minimum=minimum,
                      weight=weight, color=color, align="center")

    line(.84, .55, cover.get("institution", ""), size=16, minimum=13, weight="bold", color=_ACCENT)
    line(1.54, .55, cover.get("course", ""), size=12, minimum=10.6)
    line(2.10, .30, cover.get("course_code", ""), size=10.4, minimum=9.4, color=_MUTED)
    page.fig.add_artist(plt.Line2D([left / _WIDTH, (left + width) / _WIDTH], [1 - 2.65 / _HEIGHT] * 2,
                                   transform=page.fig.transFigure, color=_ACCENT, linewidth=1.3))
    line(3.10, 1.12, "Study Habits and\nTravel to Campus", size=29, minimum=25, weight="bold")
    line(4.49, .36, "TAKE-HOME APPLIED EXAM  |  CHAPTERS 1–3", size=10.5, minimum=9.8, color=_ACCENT)

    line(5.65, .28, "SUBMITTED BY", size=9, minimum=9, weight="bold", color=_ACCENT)
    line(6.04, .68, cover["student_name"], size=17, minimum=13, weight="bold")
    detail_y = 6.89
    for key in ("program", "year_section"):
        value = cover.get(key, "")
        if value.strip():
            line(detail_y, .55, value, size=11, minimum=9.5, color=_MUTED)
            detail_y += .57
    instructor = cover.get("instructor", "")
    if instructor.strip():
        line(8.43, .28, "SUBMITTED TO", size=9, minimum=9, weight="bold", color=_ACCENT)
        line(8.83, .61, instructor, size=12, minimum=10.2)
    submission_date = cover.get("submission_date", "")
    if submission_date.strip():
        line(10.00, .28, "SUBMISSION DATE", size=9, minimum=9, weight="bold", color=_ACCENT)
        line(10.39, .45, submission_date, size=11, minimum=9.5)
    return page


def _page_one(result, metadata):
    page = _Page(1, "Study habits and travel to campus")
    page.heading(1.42, "1  Decision question")
    page.text(.45, 1.77, 7.36, .63, metadata["decision_question"], size=10)
    page.heading(2.48, "2  Sample, method, and measurement")
    n = result["n"]
    capped = result.get("topcoded_count", 0) > 0
    total, excluded = len(result["raw"]), len(result["excluded"])
    if _documented_survey(metadata):
        method = (f"Population: {metadata['population']}\n"
                  f"Sample: {n} valid respondents from {total} submitted records; {excluded} excluded by validation. "
                  f"Collection: {metadata['collection_start']} to {metadata['collection_end']}. "
                  f"Sampling method: {metadata['sampling_method'].strip().lower()}. {metadata['selection_procedure']}")
    else:
        method = (f"Analysis includes {n} complete records from {total} input rows; {excluded} excluded by validation. "
                  + (metadata.get("collection_note", "").strip() + " " if metadata.get("collection_note", "").strip() else "")
                  + f"Intended study population: {metadata['population']}")
    study_definition = "Q2 records discrete study hours. "
    if capped:
        study_definition += "The open-ended 8+ answer is scored as 8, a lower bound. "
    method += ("\nA statistic describes these records (for example, their mean); a parameter describes the whole target population. "
               "These records do not establish the population mean.\n"
               "Q1 is ordinal categorical. " + study_definition +
               "Q3 is continuous time recorded in whole minutes. Hours and minutes use ratio scales; "
               "the survey codes walking/on-campus travel as 0.\n"
               f"Rounding: {metadata['rounding_rule']}")
    study_table = result["tables"]["q2"]
    study_width = float(study_table.iloc[0]["upper_boundary"] - study_table.iloc[0]["lower_boundary"])
    midpoints = ", ".join(f"{float(value):g}" for value in study_table["midpoint"])
    commute_table = result["tables"]["q3"]
    width = float(commute_table.iloc[0]["upper_boundary"] - commute_table.iloc[0]["lower_boundary"])
    cap_note = " (the last class contains capped scores)" if capped else ""
    method += (f"\nGrouping: Q2 class width {study_width:g} h; midpoints {midpoints}{cap_note}. "
               f"Q3 width {width:g} min; whole-minute limits have boundaries +/-0.5 min. Midpoint = (lower + upper boundary) / 2.")
    page.text(.45, 2.84, 7.36, 2.55, method, size=9)
    page.heading(5.56, "3  Frequency distributions")
    page.text(.45, 5.94, 3.52, .23, "Q1: Year level", size=9, weight="bold")
    page.text(4.16, 5.94, 3.65, .23, "Q2: Study hours (8 = 8+ score)" if capped else "Q2: Study hours", size=9, weight="bold")
    headers = ["Category", "f", "Rel. %", "Cum. f"]
    page.table(.45, 6.25, 3.52, 2.24, headers, _frequency_rows(result["tables"]["q1"], "category"),
               [.52, .12, .19, .17], fontsize=7.9)
    page.table(4.16, 6.25, 3.65, 2.24, ["Hours / class", "f", "Rel. %", "Cum. f"],
               _frequency_rows(result["tables"]["q2"], "class_interval"), [.49, .12, .21, .18], fontsize=7.9)
    page.text(.45, 8.71, 7.36, .23, "Q3: One-way commute time (minutes)", size=9, weight="bold")
    page.table(.45, 9.02, 7.36, 1.48, ["Minute class", "Frequency", "Relative %", "Cumulative f"],
               _frequency_rows(result["tables"]["q3"], "class_interval"), [.40, .20, .20, .20], fontsize=8.1)
    page.text(.45, 10.65, 7.36, .31,
              f"All three tables use n = {n}; relative frequency = f / n. Percentages may differ from 100% by rounding. "
              "Cumulative year-level counts follow form order only.", size=7.9, minimum=7.9, color=_MUTED)
    return page


def _page_two(result, metadata):
    page = _Page(2, "Description of the recorded values")
    page.heading(1.42, "4  Descriptive statistics and first two figures")
    year = result["tables"]["q1"]
    largest = int(year["frequency"].max())
    common = ", ".join(year.loc[year["frequency"].eq(largest), "category"])
    absent = ", ".join(year.loc[year["frequency"].eq(0), "category"]) or "none"
    page.text(.45, 1.79, 7.36, 1.01,
              "Statistics below use ungrouped valid records. Figures 1 and 2 show year-level composition and recorded study hours. "
              f"Most common year level(s): {common} ({largest} each). Missing year-level categories: {absent}. "
              "Sample variance and standard deviation use n - 1. Inclusive linear quantiles match Sheets INC; "
              "Bluman section 3-3 hand rules may differ.", size=9)
    statistics = result["statistics"]
    rows = [[label, _value(statistics.loc[key, "study_hours"]), _value(statistics.loc[key, "commute_minutes"])]
            for key, label in _METRICS]
    page.table(.45, 3.00, 7.36, 2.67, ["Statistic", "Study hours (h)", "Commute time (min)"], rows,
               [.45, .275, .275], fontsize=8.2)
    capped = result.get("topcoded_count", 0) > 0
    study_note = (f"{result['topcoded_count']} record(s) selected 8+. Study summaries describe capped scores and can understate actual hours. "
                  if capped else "All study-hour answers are finite whole-hour values; no open-ended answer occurs. ")
    page.text(.45, 5.83, 7.36, .65, study_note + _outliers(result), size=8.5)
    page.chart(.45, 6.77, 3.61, 3.97, result["figures"][_FIGURES[0][0]], 1, _FIGURES[0][1])
    page.chart(4.20, 6.77, 3.61, 3.97, result["figures"][_FIGURES[1][0]], 2, _FIGURES[1][1], capped=capped)
    return page


def _page_three(result, metadata):
    page = _Page(3, "Patterns, decision, and limitations")
    capped = result.get("topcoded_count", 0) > 0
    page.text(.45, 1.43, 7.36, .85,
              "Figures 3 and 4 show study-hour frequencies and cumulative counts; Figures 5 and 6 show commute shape and IQR outliers. "
              f"Study hours: {_shape_summary(result['shapes']['q2'])} Commute: {_shape_summary(result['shapes']['q3'])} "
              "These are not normality tests. The polygon closes at zero; the ogive starts at zero below the first class.", size=8.5)
    for index, (x, y) in enumerate(((.45, 2.32), (4.20, 2.32), (.45, 4.63), (4.20, 4.63)), start=2):
        key, caption = _FIGURES[index]
        page.chart(x, y, 3.61, 2.12, result["figures"][key], index + 1, caption, capped=capped)
    page.heading(6.98, "5  Recommendation")
    page.text(.45, 7.32, 7.36, 1.08, metadata["recommendation"], size=9)
    page.heading(8.56, "6  Limitations and next evidence")
    measurement_limits = ("Self-report, study-hour top-coding, and the zero-for-walking rule limit interpretation. "
                          if capped else "Self-report and the zero-for-walking rule limit interpretation. ")
    limitations = (f"Most vulnerable assumption: {metadata['vulnerable_assumption']} "
                   f"Next evidence: {metadata['next_evidence']} "
                   + measurement_limits +
                   "Descriptive associations do not establish cause and effect.")
    page.text(.45, 8.91, 7.36, 1.24, limitations, size=8.5)
    page.text(.45, 10.34, 7.36, .53,
              "References: Allan G. Bluman (2012), Elementary Statistics: A Step by Step Approach, 8th ed., McGraw-Hill; sections 1-1, 1-2, 1-3; "
              "2-1, 2-2, 2-3; 3-1, 3-2, 3-3, 3-4. Cor Jesu College applied exam and supplied survey form.",
              size=7.8, minimum=7.8, color=_MUTED)
    return page


def export_report(result: dict, metadata: dict, path: str | Path) -> Path:
    """Save three A4 analysis pages and an optional cover, with no partial PDF."""
    problems = report_readiness(result, metadata)
    if problems:
        raise ValueError("Report is not ready:\n- " + "\n- ".join(problems))
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".pdf":
        raise ValueError("Report output must have a .pdf extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pages = []
    temporary = None
    existing_figures = set(plt.get_fignums())
    try:
        cover = _cover_metadata(metadata)
        if cover is not None:
            pages.append(_cover_page(cover))
        for build in (_page_one, _page_two, _page_three):
            pages.append(build(result, metadata))
        with tempfile.NamedTemporaryFile(prefix=".survey-report-", suffix=".pdf", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with PdfPages(temporary) as pdf:
            pdf.infodict()["Title"] = "Cor Jesu College descriptive statistics report"
            pdf.infodict()["Subject"] = "Three-page descriptive statistics report"
            if cover is not None:
                pdf.infodict()["Author"] = cover["student_name"].strip()
            for page in pages:
                pdf.savefig(page.fig)
        temporary.replace(destination)
        return destination
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
        if temporary is not None and temporary.exists():
            temporary.unlink()
