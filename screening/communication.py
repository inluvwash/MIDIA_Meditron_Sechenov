from __future__ import annotations

from typing import Any

from .schema import CLASSES, LABS


def _short_lab(patient: dict[str, Any], marker: str) -> str | None:
    if marker not in patient.get("labs", {}):
        return None
    value = patient["labs"][marker]
    unit = patient.get("units", {}).get(marker, LABS.get(marker, {}).get("unit", ""))
    return f"{LABS.get(marker, {}).get('label', marker)} {value:g} {unit}".strip()


def build_doctor_brief(result: dict[str, Any]) -> dict[str, Any]:
    """Compress the multilayer result into a small attention budget for the physician."""
    clinical = result.get("clinical", {})
    final = result.get("final", {})
    unc = result.get("uncertainty", {})
    patient = result.get("patient", {})

    priority: list[str] = []
    for alert in clinical.get("alerts", [])[:2]:
        priority.append(alert)
    if final.get("label"):
        priority.append(f"Ведущая скрининговая гипотеза: {final['label']}.")
    if final.get("rule_class") and final.get("model_class") and not final.get("agreement"):
        priority.append("Клинические правила и ML-слой расходятся — конфликт показан явно, а не скрыт усреднением.")
    best = unc.get("best_next_measurement")
    if best:
        priority.append(f"Наиболее информативное недостающее измерение: {best['label']}.")
    if not priority:
        priority.append("Данных недостаточно для устойчивой гипотезы; сначала требуется проверить полноту ключевых показателей.")

    anchors = [x for x in (
        _short_lab(patient, "hemoglobin"), _short_lab(patient, "ferritin"),
        _short_lab(patient, "TSAT"), _short_lab(patient, "Ret_He"),
        _short_lab(patient, "vitamin_B12"), _short_lab(patient, "folate"), _short_lab(patient, "CRP"),
    ) if x]

    return {
        "headline": final.get("label") or "Недостаточно данных для класса",
        "priority": priority[:4],
        "anchors": anchors[:6],
        "uncertainty_status": unc.get("status", "unavailable"),
    }


def build_patient_conversation(result: dict[str, Any]) -> dict[str, Any]:
    """Patient-facing factual scaffold shown by the physician.

    It deliberately contains no scripted empathy, no behavioural coaching for the
    physician and no autonomous treatment advice. It only converts already computed
    findings into a compact, visual explanation that can be shown during a visit.
    """
    final = result.get("final", {})
    unc = result.get("uncertainty", {})
    patient = result.get("patient", {})
    model = result.get("model") or {}

    summary = (
        f"По совокупности доступных лабораторных данных ведущая рабочая гипотеза — «{final['label']}». "
        "Это скрининговый вывод; окончательную клиническую интерпретацию выполняет врач."
        if final.get("label")
        else "Данных пока недостаточно для устойчивой рабочей гипотезы."
    )

    facts = [x for x in (
        _short_lab(patient, "hemoglobin"),
        _short_lab(patient, "ferritin"),
        _short_lab(patient, "TSAT"),
        _short_lab(patient, "Ret_He"),
        _short_lab(patient, "vitamin_B12"),
        _short_lab(patient, "folate"),
        _short_lab(patient, "CRP"),
    ) if x][:4]

    top = unc.get("top_hypothesis") or {}
    alt = unc.get("alternative") or {}
    if top and alt:
        uncertainty_plain = (
            f"Имеющихся данных недостаточно, чтобы полностью разделить «{top.get('label', '')}» "
            f"и «{alt.get('label', '')}». Эти состояния также могут частично сочетаться."
        )
    else:
        uncertainty_plain = unc.get("message") or "Выраженная конкурирующая гипотеза по текущей карте неопределённости не выделена."

    best = unc.get("best_next_measurement")
    if best:
        next_step = (
            f"Для уточнения картины показатель «{best['label']}» выбран как наиболее информативный в текущем наборе данных: "
            f"{best['why']}"
        )
    else:
        next_step = "Необходимость дальнейшего обследования определяется врачом после сопоставления результата с клинической картиной."

    ranked: list[dict[str, Any]] = []
    for code, score in list(model.get("class_scores", {}).items())[:3]:
        ranked.append({
            "code": code,
            "label": CLASSES.get(code, code),
            "rating": max(0, min(100, int(round(float(score) * 100)))),
        })

    causal_simple: list[dict[str, str]] = []
    for chain in result.get("causal_chains", [])[:2]:
        causal_simple.append({
            "observed": "; ".join(chain.get("markers", [])[:3]) or "Лабораторные данные",
            "hypothesis": chain.get("hypothesis", chain.get("label", "")),
            "possible_context": "; ".join(chain.get("causes", [])[:2]) or "Причина требует клинического уточнения",
        })

    return {
        "plain_summary": summary,
        "key_facts": facts,
        "uncertainty_plain": uncertainty_plain,
        "why_next_step": next_step,
        "model_ranking": ranked,
        "causal_simple": causal_simple,
        "source_ids": ["case_su"],
    }
