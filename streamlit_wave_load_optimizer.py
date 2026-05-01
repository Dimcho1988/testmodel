"""
Wave-based component load optimizer for biathlon.

This file can run in two modes:

1) Streamlit mode, when Streamlit is installed:
   pip install streamlit pandas numpy plotly
   streamlit run streamlit_wave_load_optimizer.py

2) Console/test mode, when Streamlit is NOT installed:
   python streamlit_wave_load_optimizer.py
   python streamlit_wave_load_optimizer.py --test
   python streamlit_wave_load_optimizer.py --export-csv wave_component_training_plan.csv

Why this structure exists:
- Some sandboxed environments do not include Streamlit and do not allow installing packages.
- The optimization/model logic should still be testable without Streamlit.
- The Streamlit UI is loaded only inside run_streamlit_app(), so importing or running the model
  will not fail when Streamlit is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "This script requires pandas. Install it locally with: pip install pandas"
    ) from exc


# ============================================================
# CORE MODEL
# ============================================================
# Main idea:
# - Components are ordered from general to specific.
# - The model creates wave-like stress dynamics over weeks.
# - Each mesocycle has max N accent components.
# - Accents gradually move from general components to more specific/integrative ones.
# - Competition weeks modify the planned stress profile automatically.
# ============================================================


DEFAULT_COMPONENTS = [
    {"component": "Z1 възстановителна издръжливост", "base_load": 180, "group": "Обща издръжливост", "specificity": 1},
    {"component": "Z2 основна аеробна издръжливост", "base_load": 300, "group": "Обща издръжливост", "specificity": 2},
    {"component": "ОСИ обща силова издръжливост", "base_load": 80, "group": "Сила", "specificity": 3},
    {"component": "Z3 темпова издръжливост", "base_load": 70, "group": "Смесена издръжливост", "specificity": 4},
    {"component": "ССИ специална силова издръжливост", "base_load": 55, "group": "Сила", "specificity": 5},
    {"component": "Z4 прагова издръжливост", "base_load": 45, "group": "Специфична издръжливост", "specificity": 6},
    {"component": "Z5 VO₂max интервали", "base_load": 25, "group": "Висока интензивност", "specificity": 7},
    {"component": "Z6 скорост / нервно-мускулна мощност", "base_load": 12, "group": "Скорост", "specificity": 8},
    {"component": "Стрелба ниска интензивност", "base_load": 180, "group": "Стрелба", "specificity": 9},
    {"component": "Стрелба след натоварване", "base_load": 120, "group": "Стрелба", "specificity": 10},
    {"component": "Интегрирана състезателна подготовка", "base_load": 50, "group": "Интеграция", "specificity": 11},
]

STRESS_ZONES = {
    "Възстановително": (0.00, 0.85),
    "Поддържащо": (0.85, 1.10),
    "Развиващо": (1.10, 1.60),
    "Рисково": (1.60, 10.00),
}


@dataclass(frozen=True)
class ModelParams:
    weeks: int = 12
    mesocycle_len: int = 4
    max_accents: int = 3
    progression: float = 0.20
    wave_amplitude: float = 0.25
    recovery_stress: float = 0.75
    maintenance_stress: float = 0.98
    intro_stress: float = 1.20
    dev1_stress: float = 1.35
    dev2_stress: float = 1.45
    risk_limit: float = 1.60
    competition_weeks: Optional[List[int]] = None

    def __post_init__(self):
        if self.weeks < 1:
            raise ValueError("weeks must be >= 1")
        if self.mesocycle_len < 1:
            raise ValueError("mesocycle_len must be >= 1")
        if self.max_accents < 1:
            raise ValueError("max_accents must be >= 1")
        if self.risk_limit <= 0:
            raise ValueError("risk_limit must be positive")


def get_competition_weeks(params: ModelParams) -> List[int]:
    return sorted(set(params.competition_weeks or []))


def parse_competition_weeks(text: str, weeks: int) -> List[int]:
    """Parse comma- or semicolon-separated competition week numbers."""
    if not text or not text.strip():
        return []

    result = []
    for raw in text.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            week = int(raw)
        except ValueError:
            continue
        if 1 <= week <= weeks:
            result.append(week)
    return sorted(set(result))


def classify_stress(value: float) -> str:
    for name, (lo, hi) in STRESS_ZONES.items():
        if lo <= value < hi:
            return name
    return "Неопределено"


def mesocycle_week_type(week: int, mesocycle_len: int) -> str:
    position = ((week - 1) % mesocycle_len) + 1

    if mesocycle_len == 1:
        return "Развиваща"
    if position == 1:
        return "Вработваща"
    if mesocycle_len >= 4 and position == mesocycle_len:
        return "Възстановителна"
    if mesocycle_len == 3 and position == 3:
        return "Възстановителна"
    return "Развиваща"


def accent_stress_for_week(week: int, params: ModelParams) -> float:
    position = ((week - 1) % params.mesocycle_len) + 1

    if params.mesocycle_len == 1:
        return params.dev1_stress
    if position == 1:
        return params.intro_stress
    if position == 2:
        return params.dev1_stress
    if position == 3 and params.mesocycle_len >= 4:
        return params.dev2_stress
    if position == params.mesocycle_len:
        return params.recovery_stress
    return params.dev1_stress


def validate_components(components_df: pd.DataFrame) -> pd.DataFrame:
    required = {"component", "base_load", "group", "specificity"}
    missing = required - set(components_df.columns)
    if missing:
        raise ValueError(f"Missing required component columns: {sorted(missing)}")

    cleaned = components_df.dropna(subset=["component", "base_load", "specificity"]).copy()
    if cleaned.empty:
        raise ValueError("At least one valid component is required.")

    cleaned["component"] = cleaned["component"].astype(str)
    cleaned["group"] = cleaned["group"].astype(str)
    cleaned["base_load"] = cleaned["base_load"].astype(float)
    cleaned["specificity"] = cleaned["specificity"].astype(int)

    if (cleaned["base_load"] < 0).any():
        raise ValueError("base_load must be >= 0 for every component.")
    if (cleaned["specificity"] < 1).any():
        raise ValueError("specificity must be >= 1 for every component.")

    return cleaned.sort_values("specificity").reset_index(drop=True)


def select_accents_for_mesocycle(
    components_df: pd.DataFrame,
    meso_index: int,
    total_mesocycles: int,
    max_accents: int,
) -> List[str]:
    """
    Select max_accents components for the mesocycle.
    Accents gradually move from low specificity to high specificity.
    """
    ordered = components_df.sort_values("specificity").reset_index(drop=True)
    n_components = len(ordered)
    if n_components == 0:
        return []

    max_accents = min(max_accents, n_components)

    if total_mesocycles <= 1:
        center = min(n_components - 1, max_accents // 2)
    else:
        center = round((meso_index / (total_mesocycles - 1)) * (n_components - 1))

    half = max_accents // 2
    start = max(0, center - half)
    end = start + max_accents

    if end > n_components:
        end = n_components
        start = max(0, end - max_accents)

    return ordered.iloc[start:end]["component"].tolist()


def component_phase_envelope(
    week: int,
    weeks: int,
    specificity_rank: int,
    max_specificity: int,
    amplitude: float,
) -> float:
    """
    Smooth wave envelope.
    Earlier components peak earlier; specific components peak later.
    Returns a multiplier around 1.0.
    """
    if weeks <= 1 or max_specificity <= 1:
        return 1.0

    target_week = 1 + ((specificity_rank - 1) / (max_specificity - 1)) * (weeks - 1)
    sigma = max(1.5, weeks / 5)
    gaussian = math.exp(-0.5 * ((week - target_week) / sigma) ** 2)
    sinus = 0.5 + 0.5 * math.sin(2 * math.pi * (week - 1) / 4)

    return 1.0 + amplitude * (0.65 * gaussian + 0.35 * sinus - 0.35)


def apply_competition_logic(
    stress: float,
    component: str,
    week: int,
    competition_weeks: List[int],
    risk_limit: float,
) -> Tuple[float, str]:
    """Modify stress around competition weeks."""
    note = ""
    lower_component = component.lower()
    is_specific = any(k in lower_component for k in ["стрелба след", "интегрирана", "z5", "z6", "прагова"])
    is_strength = any(k in lower_component for k in ["оси", "сси", "силова"])

    if week in competition_weeks:
        if "интегрирана" in lower_component or "стрелба след" in lower_component:
            stress = min(max(stress, 1.10), 1.25)
            note = "Състезателна седмица: специфичният компонент се поддържа/активира."
        elif is_strength:
            stress = min(stress, 0.75)
            note = "Състезателна седмица: силовият компонент се разтоварва."
        else:
            stress = min(stress, 0.90)
            note = "Състезателна седмица: ограничаване на общия обем."

    elif (week + 1) in competition_weeks:
        if is_strength:
            stress = min(stress, 0.80)
            note = "Предсъстезателна седмица: ограничаване на силовата умора."
        elif is_specific:
            stress = min(max(stress, 0.95), 1.10)
            note = "Предсъстезателна седмица: кратък специфичен стимул без натрупване."
        else:
            stress = min(stress, 0.95)
            note = "Предсъстезателна седмица: леко сваляне на обема."

    stress = min(stress, risk_limit)
    return stress, note


def generate_plan(components_df: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    """Generate weekly stress and load targets for each component."""
    components_df = validate_components(components_df)
    competition_weeks = get_competition_weeks(params)

    rows = []
    total_mesocycles = math.ceil(params.weeks / params.mesocycle_len)
    max_specificity = int(components_df["specificity"].max())

    accents_by_meso: Dict[int, List[str]] = {}
    for meso_index in range(total_mesocycles):
        accents_by_meso[meso_index] = select_accents_for_mesocycle(
            components_df=components_df,
            meso_index=meso_index,
            total_mesocycles=total_mesocycles,
            max_accents=params.max_accents,
        )

    for week in range(1, params.weeks + 1):
        meso_index = (week - 1) // params.mesocycle_len
        week_type = mesocycle_week_type(week, params.mesocycle_len)
        accents = accents_by_meso[meso_index]
        progression_factor = 1.0 + params.progression * ((week - 1) / max(1, params.weeks - 1))

        for _, row in components_df.iterrows():
            component = str(row["component"])
            base_load = float(row["base_load"])
            specificity = int(row["specificity"])
            is_accent = component in accents

            if is_accent:
                stress = accent_stress_for_week(week, params)
            elif week_type == "Възстановителна":
                stress = params.recovery_stress
            else:
                stress = params.maintenance_stress

            envelope = component_phase_envelope(
                week=week,
                weeks=params.weeks,
                specificity_rank=specificity,
                max_specificity=max_specificity,
                amplitude=params.wave_amplitude,
            )
            stress *= envelope

            # Do not allow hidden non-accent components to become developing.
            if not is_accent and week_type != "Възстановителна":
                stress = min(stress, 1.09)

            stress, competition_note = apply_competition_logic(
                stress=stress,
                component=component,
                week=week,
                competition_weeks=competition_weeks,
                risk_limit=params.risk_limit,
            )

            # Recovery week should stay recovery unless it is a competition week.
            if week_type == "Възстановителна" and week not in competition_weeks:
                stress = min(stress, 0.85)

            target_load = base_load * stress * progression_factor

            if is_accent:
                status = "Акцент"
            elif classify_stress(stress) == "Възстановително":
                status = "Разтоварване"
            else:
                status = "Поддържане"

            if competition_note:
                note = competition_note
            elif is_accent and week_type != "Възстановителна":
                note = "Планиран развиващ акцент."
            elif week_type == "Възстановителна":
                note = "Възстановителна вълна: целево сваляне на товара."
            else:
                note = "Поддържащ компонент — без скрит допълнителен акцент."

            rows.append(
                {
                    "week": week,
                    "mesocycle": meso_index + 1,
                    "week_type": week_type,
                    "component": component,
                    "group": row["group"],
                    "specificity": specificity,
                    "is_accent": is_accent,
                    "status": status,
                    "stress": round(stress, 3),
                    "stress_zone": classify_stress(stress),
                    "base_load": round(base_load, 1),
                    "target_load": round(target_load, 1),
                    "progression_factor": round(progression_factor, 3),
                    "note": note,
                }
            )

    return pd.DataFrame(rows)


def make_weekly_summary(plan_df: pd.DataFrame) -> pd.DataFrame:
    return (
        plan_df.groupby(["week", "mesocycle", "week_type"], as_index=False)
        .agg(
            total_target_load=("target_load", "sum"),
            mean_stress=("stress", "mean"),
            max_stress=("stress", "max"),
            accent_count=("is_accent", "sum"),
        )
        .round(2)
    )


def make_accent_table(plan_df: pd.DataFrame) -> pd.DataFrame:
    accents = plan_df[plan_df["is_accent"]].copy()
    if accents.empty:
        return pd.DataFrame(columns=["mesocycle", "week", "accent_components"])

    return (
        accents.groupby(["mesocycle", "week"], as_index=False)["component"]
        .apply(lambda values: ", ".join(values))
        .rename(columns={"component": "accent_components"})
    )


def diagnostic_checks(plan_df: pd.DataFrame, params: ModelParams) -> List[str]:
    issues: List[str] = []
    weekly_summary = make_weekly_summary(plan_df)
    competition_weeks = get_competition_weeks(params)

    if (weekly_summary["accent_count"] > params.max_accents).any():
        issues.append("Има седмици с повече акценти от позволеното.")

    if (plan_df["stress"] >= params.risk_limit).any():
        issues.append(f"Има компоненти, достигнали горната рискова граница {params.risk_limit:.2f}.")

    hidden = plan_df[(~plan_df["is_accent"]) & (plan_df["stress"] > 1.10)]
    if not hidden.empty:
        issues.append("Има неакцентирани компоненти, които преминават в развиваща зона.")

    recovery_weeks = plan_df[plan_df["week_type"] == "Възстановителна"]
    if not recovery_weeks.empty:
        recovery_bad = recovery_weeks[
            (recovery_weeks["stress"] > 0.90) & (~recovery_weeks["week"].isin(competition_weeks))
        ]
        if not recovery_bad.empty:
            issues.append("Някои възстановителни седмици не свалят достатъчно стреса.")

    return issues


# ============================================================
# TESTS
# ============================================================


def _default_components_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_COMPONENTS)


def run_tests() -> None:
    """Small self-contained test suite. No pytest required."""
    components = _default_components_df()
    params = ModelParams(weeks=12, mesocycle_len=4, max_accents=3, competition_weeks=[8, 12])
    plan = generate_plan(components, params)

    # Test 1: expected number of rows.
    assert len(plan) == params.weeks * len(components), "Plan row count is incorrect."

    # Test 2: no week exceeds the allowed number of accents.
    weekly = make_weekly_summary(plan)
    assert (weekly["accent_count"] <= params.max_accents).all(), "Too many accents in at least one week."

    # Test 3: competition week strength components are reduced.
    comp8_strength = plan[
        (plan["week"] == 8)
        & (plan["component"].str.contains("ОСИ|ССИ|силова", case=False, regex=True))
    ]
    assert not comp8_strength.empty, "Strength components were not found in test data."
    assert (comp8_strength["stress"] <= 0.75).all(), "Strength should be reduced in competition week."

    # Test 4: recovery weeks without competition should remain controlled.
    recovery_non_comp = plan[(plan["week_type"] == "Възстановителна") & (~plan["week"].isin(params.competition_weeks or []))]
    assert (recovery_non_comp["stress"] <= 0.85).all(), "Recovery week stress should be <= 0.85."

    # Test 5: parser should ignore invalid values and remove duplicates.
    assert parse_competition_weeks("3, 5, x, 5, 99", weeks=12) == [3, 5], "Competition parser failed."

    # Test 6: accent selection should move toward more specific components later.
    total_mesocycles = math.ceil(params.weeks / params.mesocycle_len)
    early = select_accents_for_mesocycle(components, 0, total_mesocycles, 3)
    late = select_accents_for_mesocycle(components, total_mesocycles - 1, total_mesocycles, 3)
    early_mean = components[components["component"].isin(early)]["specificity"].mean()
    late_mean = components[components["component"].isin(late)]["specificity"].mean()
    assert late_mean > early_mean, "Accents should move from general to specific components."

    # Test 7: diagnostic check should return a list, not crash.
    issues = diagnostic_checks(plan, params)
    assert isinstance(issues, list), "diagnostic_checks should return list."

    print("All tests passed.")


# ============================================================
# CONSOLE FALLBACK
# ============================================================


def run_console_demo(export_csv: Optional[str] = None) -> None:
    """Run the model without Streamlit and print useful tables."""
    components = _default_components_df()
    params = ModelParams(weeks=12, mesocycle_len=4, max_accents=3, competition_weeks=[8, 12])
    plan = generate_plan(components, params)
    weekly_summary = make_weekly_summary(plan)
    accent_table = make_accent_table(plan)
    issues = diagnostic_checks(plan, params)

    print("\n=== Wave-based component load optimizer ===")
    print("Streamlit is not required for this console demo.")
    print("To use the visual UI locally, install Streamlit and run:")
    print("  pip install streamlit pandas plotly")
    print("  streamlit run streamlit_wave_load_optimizer.py")

    print("\n=== Weekly summary ===")
    print(weekly_summary.to_string(index=False))

    print("\n=== Accent table ===")
    print(accent_table.to_string(index=False))

    print("\n=== First 20 plan rows ===")
    columns = ["week", "week_type", "component", "status", "stress", "stress_zone", "target_load", "note"]
    print(plan[columns].head(20).to_string(index=False))

    print("\n=== Diagnostic checks ===")
    if issues:
        for issue in issues:
            print(f"WARNING: {issue}")
    else:
        print("OK: Планът спазва основните ограничения.")

    if export_csv:
        plan.to_csv(export_csv, index=False, encoding="utf-8-sig")
        print(f"\nCSV exported to: {export_csv}")


# ============================================================
# STREAMLIT UI
# ============================================================


def style_stress_table(df: pd.DataFrame):
    def color_stress(value):
        try:
            v = float(value)
        except Exception:
            return ""
        if v < 0.85:
            return "background-color: #d9ead3"
        if v < 1.10:
            return "background-color: #fff2cc"
        if v < 1.60:
            return "background-color: #f9cb9c"
        return "background-color: #e06666; color: white"

    # Styler.applymap is still common, but map is preferred in newer pandas.
    if hasattr(df.style, "map"):
        return df.style.map(color_stress)
    return df.style.applymap(color_stress)


def build_full_stress_figure(
    plan_df: pd.DataFrame,
    competition_weeks: List[int],
    title: str = "Цялостна вълнообразна динамика на компонентите",
):
    """
    Build the main Streamlit chart.

    This is intentionally a separate helper so the app always shows one complete,
    coach-readable overview chart when opened in Streamlit.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    ordered_components = (
        plan_df[["component", "specificity"]]
        .drop_duplicates()
        .sort_values("specificity")["component"]
        .tolist()
    )

    for component in ordered_components:
        data = plan_df[plan_df["component"] == component].sort_values("week")
        if data.empty:
            continue

        # Normal line for the whole component.
        fig.add_trace(
            go.Scatter(
                x=data["week"],
                y=data["stress"],
                mode="lines+markers",
                name=component,
                line=dict(width=2),
                marker=dict(size=7),
                customdata=data[["week_type", "status", "stress_zone", "target_load", "note"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Седмица: %{x}<br>"
                    "Стрес: %{y:.2f}<br>"
                    "Тип седмица: %{customdata[0]}<br>"
                    "Статус: %{customdata[1]}<br>"
                    "Зона: %{customdata[2]}<br>"
                    "Целеви товар: %{customdata[3]:.1f}<br>"
                    "Бележка: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

        # Larger accent markers over the line.
        accents = data[data["is_accent"]]
        if not accents.empty:
            fig.add_trace(
                go.Scatter(
                    x=accents["week"],
                    y=accents["stress"],
                    mode="markers",
                    name=f"Акцент: {component}",
                    marker=dict(size=13, symbol="diamond", line=dict(width=1)),
                    showlegend=False,
                    hovertemplate=(
                        "<b>Акцент</b><br>"
                        f"Компонент: {component}<br>"
                        "Седмица: %{x}<br>"
                        "Стрес: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    # Background stress zones.
    fig.add_hrect(y0=0.00, y1=0.85, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.85, y1=1.10, fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=1.10, y1=1.60, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_hrect(y0=1.60, y1=2.00, fillcolor="red", opacity=0.08, line_width=0)

    # Threshold lines.
    fig.add_hline(y=0.85, line_dash="dash", annotation_text="0.85 възстановително", annotation_position="bottom right")
    fig.add_hline(y=1.10, line_dash="dash", annotation_text="1.10 развиващо", annotation_position="bottom right")
    fig.add_hline(y=1.60, line_dash="dash", annotation_text="1.60 риск", annotation_position="top right")

    # Competition weeks.
    for week in competition_weeks:
        fig.add_vrect(
            x0=week - 0.5,
            x1=week + 0.5,
            fillcolor="red",
            opacity=0.06,
            line_width=0,
            annotation_text=f"Съст. {week}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        height=760,
        hovermode="x unified",
        legend_title_text="Компоненти",
        margin=dict(l=40, r=40, t=80, b=40),
        yaxis=dict(title="Стрес по компонент", range=[0.55, 1.75]),
        xaxis=dict(title="Седмица", dtick=1),
    )
    return fig


def build_group_summary_figure(plan_df: pd.DataFrame):
    """Build a cleaner chart by group mean stress."""
    import plotly.express as px

    group_df = (
        plan_df.groupby(["week", "group"], as_index=False)
        .agg(mean_stress=("stress", "mean"), total_load=("target_load", "sum"))
        .round(3)
    )
    fig = px.line(
        group_df,
        x="week",
        y="mean_stress",
        color="group",
        markers=True,
        hover_data=["total_load"],
        title="Обобщена динамика по групи компоненти",
    )
    fig.add_hrect(y0=0.00, y1=0.85, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.85, y1=1.10, fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=1.10, y1=1.60, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_hline(y=0.85, line_dash="dash")
    fig.add_hline(y=1.10, line_dash="dash")
    fig.add_hline(y=1.60, line_dash="dash")
    fig.update_layout(height=560, xaxis=dict(dtick=1), yaxis_title="Среден стрес по група")
    return fig


def run_streamlit_app() -> None:
    """Run the Streamlit UI. Imports Streamlit and Plotly only here."""
    try:
        import plotly.express as px
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover
        missing = exc.name
        print(
            f"Missing optional UI dependency: {missing}\n"
            "Console mode still works: python streamlit_wave_load_optimizer.py\n"
            "For the visual UI, install locally:\n"
            "  pip install streamlit plotly\n"
            "  streamlit run streamlit_wave_load_optimizer.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    st.set_page_config(
        page_title="Biathlon Wave Load Optimizer",
        page_icon="🎯",
        layout="wide",
    )

    st.title("🎯 Вълнообразна динамика на тренировъчните компоненти")
    st.caption(
        "Прототип за автоматизирано планиране: компонентен стрес, максимум акценти, "
        "преход от обща към специфична подготовка и корекции около състезания."
    )

    with st.sidebar:
        st.header("1) Основни настройки")
        weeks = st.slider("Брой седмици", min_value=4, max_value=32, value=12, step=1)
        mesocycle_len = st.selectbox("Дължина на мезоцикъл", [3, 4, 5], index=1)
        max_accents = st.slider("Максимум акценти в мезоцикъл", min_value=1, max_value=5, value=3, step=1)

        st.header("2) Стрес зони")
        intro_stress = st.slider("Вработваща седмица — стрес за акцент", 1.00, 1.40, 1.20, 0.01)
        dev1_stress = st.slider("Развиваща седмица 1 — стрес за акцент", 1.05, 1.60, 1.35, 0.01)
        dev2_stress = st.slider("Развиваща седмица 2 — стрес за акцент", 1.05, 1.60, 1.45, 0.01)
        maintenance_stress = st.slider("Поддържащ стрес", 0.80, 1.10, 0.98, 0.01)
        recovery_stress = st.slider("Възстановителен стрес", 0.50, 0.90, 0.75, 0.01)
        risk_limit = st.slider("Горна граница за риск", 1.20, 1.80, 1.60, 0.01)

        st.header("3) Форма на вълната")
        progression = st.slider("Постепенно увеличение на общия товар", 0.00, 0.50, 0.20, 0.01)
        wave_amplitude = st.slider("Амплитуда на компонентната вълна", 0.00, 0.60, 0.25, 0.01)

        st.header("4) Състезания")
        competition_text = st.text_input(
            "Седмици със състезания",
            value="8, 12",
            help="Въведи седмици, разделени със запетая. Например: 8, 12",
        )

    competition_weeks = parse_competition_weeks(competition_text, weeks)
    params = ModelParams(
        weeks=weeks,
        mesocycle_len=mesocycle_len,
        max_accents=max_accents,
        progression=progression,
        wave_amplitude=wave_amplitude,
        recovery_stress=recovery_stress,
        maintenance_stress=maintenance_stress,
        intro_stress=intro_stress,
        dev1_stress=dev1_stress,
        dev2_stress=dev2_stress,
        risk_limit=risk_limit,
        competition_weeks=competition_weeks,
    )

    st.subheader("Входни компоненти")
    st.write(
        "Подреди компонентите от най-общи към най-специфични чрез колоната `specificity`. "
        "Колоната `base_load` е условен базов седмичен обем — минути, серии, патрони или точки според компонента."
    )

    components_df = pd.DataFrame(DEFAULT_COMPONENTS)
    components_df = st.data_editor(
        components_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "component": st.column_config.TextColumn("Компонент", required=True),
            "base_load": st.column_config.NumberColumn("Базов седмичен обем", min_value=0.0, step=5.0, required=True),
            "group": st.column_config.TextColumn("Група", required=True),
            "specificity": st.column_config.NumberColumn("Специфичност / ред", min_value=1, step=1, required=True),
        },
    )

    try:
        components_df = validate_components(components_df)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()

    plan_df = generate_plan(components_df, params)
    weekly_summary = make_weekly_summary(plan_df)
    accent_table = make_accent_table(plan_df)
    issues = diagnostic_checks(plan_df, params)

    st.subheader("Обобщение на модела")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Седмици", weeks)
    col2.metric("Мезоцикли", math.ceil(weeks / mesocycle_len))
    col3.metric("Макс. акценти", max_accents)
    col4.metric("Състезателни седмици", ", ".join(map(str, competition_weeks)) if competition_weeks else "няма")

    st.subheader("1) Цяла графика на вълнообразната динамика")
    st.write(
        "Това е основната графика на модела. Тя показва всички компоненти за целия период, "
        "праговете на стреса, акцентите и състезателните седмици."
    )

    chart_mode = st.radio(
        "Изглед на графиката",
        options=["Всички компоненти", "Само акценти", "По групи"],
        horizontal=True,
    )

    if chart_mode == "Всички компоненти":
        fig_stress = build_full_stress_figure(
            plan_df=plan_df,
            competition_weeks=competition_weeks,
            title="Цялостна вълнообразна динамика на всички компоненти",
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    elif chart_mode == "Само акценти":
        accent_only_df = plan_df[plan_df["is_accent"]].copy()
        if accent_only_df.empty:
            st.info("В текущите настройки няма избрани акценти.")
        else:
            fig_stress = build_full_stress_figure(
                plan_df=accent_only_df,
                competition_weeks=competition_weeks,
                title="Динамика само на акцентираните компоненти",
            )
            st.plotly_chart(fig_stress, use_container_width=True)

    else:
        fig_group = build_group_summary_figure(plan_df)
        st.plotly_chart(fig_group, use_container_width=True)

    st.subheader("2) Целеви седмичен обем по компоненти")
    fig_load = px.area(
        plan_df,
        x="week",
        y="target_load",
        color="component",
        line_group="component",
        hover_data=["week_type", "status", "stress", "stress_zone"],
        title="Натрупване на целевия седмичен товар",
    )
    fig_load.update_layout(height=550, legend_title_text="Компонент")
    st.plotly_chart(fig_load, use_container_width=True)

    st.subheader("3) Карта на стреса")
    st.write("Вижда се кога даден компонент е възстановителен, поддържащ, развиващ или рисков.")
    stress_matrix = plan_df.pivot(index="component", columns="week", values="stress")
    st.dataframe(style_stress_table(stress_matrix), use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Седмично обобщение")
        st.dataframe(weekly_summary, use_container_width=True)

    with right:
        st.subheader("Акценти по седмици")
        if accent_table.empty:
            st.info("Няма избрани акценти.")
        else:
            st.dataframe(accent_table, use_container_width=True)

    st.subheader("Пълен план по компоненти")
    filter_weeks = st.multiselect(
        "Филтрирай седмици",
        options=list(range(1, weeks + 1)),
        default=list(range(1, min(weeks, 8) + 1)),
    )
    filtered = plan_df[plan_df["week"].isin(filter_weeks)].copy()
    st.dataframe(filtered, use_container_width=True, height=420)

    st.subheader("Автоматична проверка")
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("Планът спазва основните ограничения: максимум акценти, контрол на риска и вълнообразна динамика.")

    st.subheader("Експорт")
    csv = plan_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Изтегли плана като CSV",
        data=csv,
        file_name="wave_component_training_plan.csv",
        mime="text/csv",
    )

    st.info(
        "Следваща стъпка: към всеки компонент може да се свърже база от конкретни тренировки. "
        "Тогава моделът няма само да планира стреса, а ще избира реални тренировки ден по ден."
    )


# ============================================================
# ENTRY POINT
# ============================================================


def is_package_available(package_name: str) -> bool:
    """Return True when a Python package can be imported."""
    return importlib.util.find_spec(package_name) is not None


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point.

    Important for Streamlit Cloud:
    - The default behavior must be to run the Streamlit UI when Streamlit is installed.
    - Older versions tried to detect Streamlit runtime context and could fall back to console mode,
      which creates a blank white Streamlit page.
    """
    parser = argparse.ArgumentParser(description="Wave-based component load optimizer for biathlon.")
    parser.add_argument("--test", action="store_true", help="Run self-contained tests.")
    parser.add_argument("--console", action="store_true", help="Run console demo instead of Streamlit UI.")
    parser.add_argument("--export-csv", type=str, default=None, help="Export console-demo plan to CSV.")

    # parse_known_args prevents Streamlit/Cloud from breaking the app if it passes extra arguments.
    args, _unknown = parser.parse_known_args(argv)

    if args.test:
        run_tests()
        return

    if args.console or args.export_csv:
        run_console_demo(export_csv=args.export_csv)
        return

    # Main cloud/local behavior: if Streamlit is available, always show the UI.
    # This prevents the blank-page problem caused by accidentally running console mode inside Streamlit.
    if is_package_available("streamlit"):
        run_streamlit_app()
        return

    # Fallback for restricted environments without Streamlit.
    run_console_demo(export_csv=None)


if __name__ == "__main__":
    main()
