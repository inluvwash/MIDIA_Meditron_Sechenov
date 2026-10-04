from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .schema import CLASSES, DISCLAIMER, LABS, ROOT

_FONT_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_FONT_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",

]

# Visual language follows the team Figma reference: soft blue background,
# pale periwinkle surfaces and a restrained violet accent.
BRAND_BG = colors.HexColor("#f3f6ff")
BRAND_SOFT = colors.HexColor("#cdd8ff")
BRAND_DEEP = colors.HexColor("#21108e")
BRAND_ACCENT = colors.HexColor("#323fa6")
BRAND_BORDER = colors.HexColor("#d9e1ff")
BRAND_TEXT = colors.HexColor("#222222")
BRAND_TEAL_SOFT = colors.HexColor("#e9f8f6")
BRAND_WARN = colors.HexColor("#fff4e8")
LOGO_PATH = ROOT / "assets" / "midia_logo.png"


def _register_fonts() -> tuple[str, str]:
    if "HSBody" in pdfmetrics.getRegisteredFontNames():
        return "HSBody", "HSBold"
    reg = next((p for p in _FONT_REG if os.path.exists(p)), None)
    bold = next((p for p in _FONT_BOLD if os.path.exists(p)), None)
    if reg:
        pdfmetrics.registerFont(TTFont("HSBody", reg))
        pdfmetrics.registerFont(TTFont("HSBold", bold or reg))
        return "HSBody", "HSBold"
    return "Helvetica", "Helvetica-Bold"


def _p(text: Any, style) -> Paragraph:
    return Paragraph(escape(str(text or "")), style)


def _inline_markdown(text: str) -> str:
    """Very small, safe Markdown subset for ReportLab Paragraph.

    LLM output is escaped first; only **bold** and `code` are reintroduced as
    ReportLab markup. Markdown heading markers are handled line-by-line elsewhere.
    """
    value = escape(str(text or ""))
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", value)
    return value


def _markdown_flowables(text: str, normal, small, h2, h3) -> list[Any]:
    """Render common LLM Markdown cleanly instead of printing raw #/* characters."""
    out: list[Any] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph if x.strip())
            if joined:
                out.append(Paragraph(_inline_markdown(joined), normal))
            paragraph.clear()

    for raw in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush()
            out.append(Paragraph(_inline_markdown(heading.group(2)), h2 if len(heading.group(1)) <= 2 else h3))
            continue
        bullet = re.match(r"^[-*•]\s+(.+)$", line)
        if bullet:
            flush()
            out.append(Paragraph("• " + _inline_markdown(bullet.group(1)), normal))
            continue
        numbered = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if numbered:
            flush()
            out.append(Paragraph(f"{numbered.group(1)}. " + _inline_markdown(numbered.group(2)), normal))
            continue
        paragraph.append(line)
    flush()
    if not out:
        out.append(_p(text, small))
    return out


def _table(rows: list[list[Any]], widths, style, header: bool = True) -> Table:
    cooked = [[_p(v, style) for v in row] for row in rows]
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.35, BRAND_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), BRAND_SOFT))
    t.setStyle(TableStyle(cmds))
    return t


def _important_lab_keys(result: dict[str, Any]) -> set[str]:
    important: set[str] = {"hemoglobin"}
    for chain in result.get("causal_chains", []):
        for marker in chain.get("markers", []):
            marker_text = str(marker)
            for key, meta in LABS.items():
                if marker_text.startswith(str(meta.get("label", key)) + " ="):
                    important.add(key)
    return important


def _compact_lab_table(patient: dict[str, Any], result: dict[str, Any], small, bold_style) -> Table:
    important = _important_lab_keys(result)
    items = list(patient.get("labs", {}).items())
    items = sorted(enumerate(items), key=lambda x: (x[1][0] not in important, x[0]))
    pairs = [item for _, item in items]
    rows: list[list[Any]] = []
    highlight_cells: list[tuple[int, int]] = []
    for i in range(0, len(pairs), 2):
        row: list[Any] = []
        for pair_idx in range(2):
            pos = i + pair_idx
            if pos >= len(pairs):
                row.extend([_p("", small), _p("", small)])
                continue
            key, value = pairs[pos]
            label_style = bold_style if key in important else small
            label = LABS[key]["label"]
            unit = patient.get("units", {}).get(key, LABS[key].get("unit", ""))
            row.extend([_p(label, label_style), _p(f"{value:g} {unit}".strip(), label_style if key in important else small)])
            if key in important:
                col = pair_idx * 2
                highlight_cells.extend([(len(rows), col), (len(rows), col + 1)])
        rows.append(row)
    table = Table(rows, colWidths=[55 * mm, 32 * mm, 55 * mm, 32 * mm], hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.3, BRAND_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r, c in highlight_cells:
        cmds.append(("BACKGROUND", (c, r), (c, r), colors.HexColor("#eef2ff")))
    table.setStyle(TableStyle(cmds))
    return table


def generate_pdf(
    result: dict[str, Any],
    patient_name: str = "",
    ai_text: str | None = None,
    ai_patient_text: str | None = None,
) -> bytes:
    body_font, bold_font = _register_fonts()
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "HSNormal", parent=styles["Normal"], fontName=body_font, fontSize=9, leading=12,
        textColor=BRAND_TEXT, spaceAfter=4,
    )
    small = ParagraphStyle("HSSmall", parent=normal, fontSize=7.7, leading=10, textColor=colors.HexColor("#586174"))
    small_bold = ParagraphStyle("HSSmallBold", parent=small, fontName=bold_font, textColor=BRAND_DEEP)
    h1 = ParagraphStyle(
        "HSTitle", parent=styles["Title"], fontName=bold_font, fontSize=17.5, leading=21,
        textColor=BRAND_DEEP, alignment=0, spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "HSH2", parent=styles["Heading2"], fontName=bold_font, fontSize=12, leading=15,
        textColor=BRAND_DEEP, spaceBefore=10, spaceAfter=5, keepWithNext=True,
    )
    h3 = ParagraphStyle("HSH3", parent=h2, fontSize=9.7, leading=12, textColor=BRAND_ACCENT, spaceBefore=6, spaceAfter=3)
    callout = ParagraphStyle(
        "HSCallout", parent=normal, textColor=BRAND_DEEP, backColor=colors.HexColor("#eef2ff"),
        borderColor=colors.HexColor("#aab8ff"), borderWidth=0.6, borderPadding=7,
    )
    warn = ParagraphStyle(
        "HSWarn", parent=normal, backColor=BRAND_WARN, borderColor=colors.HexColor("#efc38e"),
        borderWidth=0.6, borderPadding=7,
    )
    patient_style = ParagraphStyle(
        "HSPatient", parent=normal, textColor=colors.HexColor("#164e63"), backColor=BRAND_TEAL_SOFT,
        borderColor=colors.HexColor("#91d8d2"), borderWidth=0.6, borderPadding=7,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=24 * mm,
        bottomMargin=16 * mm,
        title="Meditron - врачебный отчёт",
        author="Команда Мидия",
    )

    def _page_theme(canvas, document) -> None:
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(BRAND_BG)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setFillColor(BRAND_SOFT)
        canvas.roundRect(8 * mm, height - 19 * mm, width - 16 * mm, 14 * mm, 4 * mm, fill=1, stroke=0)
        if LOGO_PATH.exists():
            try:
                canvas.drawImage(str(LOGO_PATH), 12 * mm, height - 17 * mm, width=8.5 * mm, height=8.2 * mm, mask="auto", preserveAspectRatio=True)
            except Exception:
                pass
        canvas.setFillColor(BRAND_DEEP)
        canvas.setFont(bold_font, 8.5)
        canvas.drawString(23 * mm, height - 13.5 * mm, "MEDITRON · команда «Мидия»")
        canvas.setFillColor(BRAND_ACCENT)
        canvas.setFont(body_font, 7)
        canvas.drawRightString(width - 12 * mm, height - 13.5 * mm, f"Страница {document.page}")
        canvas.setStrokeColor(BRAND_BORDER)
        canvas.line(14 * mm, 11 * mm, width - 14 * mm, 11 * mm)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.setFont(body_font, 6.5)
        canvas.drawString(14 * mm, 7.5 * mm, "CDSS-прототип: результат требует клинической интерпретации врача")
        canvas.restoreState()

    story: list[Any] = []

    patient = result["patient"]
    clinical = result["clinical"]
    final = result["final"]
    model = result.get("model") or {}
    uncertainty = result.get("uncertainty") or {}
    brief = result.get("doctor_brief") or {}
    communication = result.get("patient_conversation") or {}
    coding = result.get("coding") or {}
    ctx = patient.get("context", {})

    story.append(_p("Многослойный скрининг анемии и дефицитных состояний", h1))
    story.append(_p("Врачебный отчёт · Meditron · команда «Мидия» · кейс Сеченовского университета · 12 целевых классов", small))
    story.append(_p(DISCLAIMER, warn))
    story.append(Spacer(1, 5))

    # Patient card is input data, not an analytical layer.
    story.append(_p("Карточка пациента", h2))
    story.append(_table([
        ["ФИО", patient_name.strip() or "не указано", "Возраст", f"{patient['age_years']:g}"],
        ["Пол", "Женский" if patient["sex"] == "F" else "Мужской", "Дата расчёта", result.get("created_at", "")],
    ], [25 * mm, 60 * mm, 25 * mm, 60 * mm], small, header=False))
    story.append(_p("ФИО добавлено только при формировании PDF и не участвовало в ML/LLM-расчёте.", small))

    context_rows = [
        ["Жалобы / симптомы", ctx.get("complaints") or "—"],
        ["Анамнез / заболевания / значимые события", ctx.get("history") or "—"],
        ["Постоянные препараты", ctx.get("medications") or "—"],
    ]
    story.append(_table(context_rows, [48 * mm, 127 * mm], small, header=False))

    story.append(_p("Исходные лабораторные показатели", h2))
    story.append(_p("Показатели, использованные в ведущей причинной/клинической цепочке, выделены.", small))
    story.append(_compact_lab_table(patient, result, small, small_bold))

    # Executive summary is a synthesis, not a separate analytical layer.
    story.append(_p("Итог скрининга", h2))
    story.append(_p(brief.get("headline", final.get("label", "")), callout))
    method_label = {
        "ml_and_rules": "ML и клинические правила согласны",
        "ml_primary": "ведущий класс предложен ML; правилового класса недостаточно",
        "ml_with_rule_disagreement": "ML и клинические правила расходятся",
        "rules_only": "результат сформирован клиническими правилами без доступного ML",
        "insufficient": "данных недостаточно",
    }.get(final.get("method"), str(final.get("method") or "—"))
    story.append(_p(
        f"Рабочая гипотеза: {final.get('label')}. Класс: {final.get('class_code') or '—'}; "
        f"согласованность слоёв: {final.get('confidence_band')}; {method_label}.",
        normal,
    ))
    story.append(_p(
        f"Гемоглобин {clinical['hemoglobin']:g} г/л; порог из ТЗ {clinical['threshold']:g} г/л; "
        f"анемия по ТЗ: {'да' if clinical['anemia'] else 'нет'}.",
        normal,
    ))
    for item in brief.get("priority", []):
        story.append(_p("• " + item, normal))
    if brief.get("anchors"):
        story.append(_p("Ключевые показатели: " + "; ".join(brief["anchors"]), small))
    for warning in result.get("warnings", []):
        story.append(_p("Предупреждение: " + warning, warn))
    for alert in clinical.get("alerts", []):
        story.append(_p("Приоритет: " + alert, warn))

    story.append(_p("Слои анализа", h2))
    story.append(_p(
        "Нумерация ниже относится только к вычислительным слоям системы. Карточка пациента и итоговое резюме являются входными данными и сводным выводом, а не отдельными слоями.",
        small,
    ))

    # Layer 1. Input normalization and safety.
    story.append(_p("1. Нормализация и контроль входных данных", h2))
    story.append(_p(
        "Лабораторные показатели приведены к каноническим названиям и единицам, проверены на технически невозможные значения и подготовлены как профиль одного взрослого пациента. Распознанные значения перед расчётом остаются доступными врачу для ручной проверки и исправления.",
        small,
    ))

    # Layer 2. Clinical logic.
    story.append(_p("2. Клиническая логика", h2))
    story.append(_p("Прозрачные лабораторные правила служат независимой проверкой ML.", small))
    if clinical.get("findings"):
        for finding in clinical["findings"]:
            story.append(_p("• " + finding["text"], normal))
    else:
        story.append(_p("По доступной панели значимых правиловых находок не сформировано.", normal))

    story.append(_p("Проверка других причин анемии", h3))
    story.append(_p(
        "Если происхождение анемии остаётся неясным, система отдельно проверяет доступные признаки значимой почечной дисфункции, явного гемолиза и выраженного тиреоидного направления. Это узкий safety-check, а не универсальный список диагнозов.",
        small,
    ))
    status_ru = {
        "abnormal": "есть лабораторный сигнал",
        "incomplete": "не хватает данных",
        "no_screening_signal": "по имеющимся данным сигнала нет",
    }
    rows = [["Направление", "Статус", "Что видно / чего не хватает"]]
    for item in clinical.get("exclusions", []):
        detail = item.get("text") or (", ".join(item.get("missing", [])) if item.get("missing") else "—")
        rows.append([item["title"], status_ru.get(item["status"], item["status"]), detail])
    if len(rows) > 1:
        story.append(_table(rows, [55 * mm, 55 * mm, 65 * mm], small))

    # ICD appears in exactly one dedicated place.
    story.append(_p("МКБ-10: справка, не автокодирование", h3))
    story.append(_p(coding.get("note", ""), small))
    if coding.get("candidates"):
        rows = [["Код / семейство", "Название", "Когда уместно"]]
        for item in coding["candidates"]:
            rows.append([item["code"], item["label"], item["when"]])
        story.append(_table(rows, [28 * mm, 70 * mm, 77 * mm], small))

    # Layer 3. ML.
    story.append(_p("3. ML-проверка", h2))
    story.append(_p(
        "ML независимо сравнивает профиль пациента с 12 классами. Model score — относительный рейтинг между классами, а не клиническая вероятность диагноза.",
        small,
    ))
    if model:
        rows = [["Класс", "Model score"]]
        for code, score in list(model.get("class_scores", {}).items())[:6]:
            rows.append([CLASSES.get(code, code), f"{score:.3f}"])
        story.append(_table(rows, [145 * mm, 30 * mm], small))
        if model.get("sensitivity"):
            story.append(_p("Локальная чувствительность показывает устойчивость ML-вывода к изменению отдельных показателей; это не причинный вклад.", small))
            rows = [["Показатель", "Пациент", "Медиана", "Δ score"]]
            for item in model["sensitivity"][:6]:
                rows.append([
                    LABS.get(item["feature"], {}).get("label", item["feature"]),
                    f"{item['value']:g}", f"{item['median']:g}", f"{item['score_change']:+.3f}",
                ])
            story.append(_table(rows, [75 * mm, 30 * mm, 35 * mm, 35 * mm], small))
    else:
        story.append(_p("ML-слой недоступен; клинические правила продолжают работать независимо.", normal))

    # Layer 4. Uncertainty.
    story.append(_p("4. Карта неопределённости", h2))
    story.append(_p("Здесь конкурирующие гипотезы отделены от возможных сопутствующих состояний.", small))
    if uncertainty.get("top_hypothesis"):
        top = uncertainty["top_hypothesis"]
        alt = uncertainty["alternative"]
        story.append(_p(f"Конкурируют: «{top['label']}» ({top['score']:.3f}) ↔ «{alt['label']}» ({alt['score']:.3f}). {uncertainty.get('message', '')}", normal))
        if uncertainty.get("ambiguity_factors"):
            for factor in uncertainty["ambiguity_factors"]:
                story.append(_p(f"• {factor['factor']}: {factor['text']}", small))
        best = uncertainty.get("best_next_measurement")
        if best:
            story.append(_p(f"Наиболее информативное недостающее измерение: {best['label']}. {best['why']}", warn))
    else:
        story.append(_p(uncertainty.get("message", "Карта неопределённости недоступна."), normal))

    coexist = result.get("coexisting_signals", [])
    if coexist:
        story.append(_p("Параллельные / сопутствующие сигналы", h3))
        for item in coexist:
            story.append(_p("• " + item["label"], normal))

    # Layer 5. Causal reasoning.
    story.append(_p("5. Причинно-следственные связи", h2))
    story.append(_p(
        "Слой не объявляет причину установленной. Он связывает лабораторные признаки с рабочей гипотезой и контекстом из жалоб, анамнеза и препаратов.",
        small,
    ))
    for chain in result.get("causal_chains", []):
        rows = [
            ["Что видим", "Что это поддерживает", "Что может объяснять"],
            [
                "\n".join("• " + x for x in chain.get("markers", [])) or "—",
                chain.get("hypothesis", chain.get("label", "")),
                "\n".join("• " + x for x in chain.get("causes", [])) or "—",
            ],
        ]
        story.append(_table(rows, [58 * mm, 55 * mm, 62 * mm], small))
        story.append(Spacer(1, 5))

    # Layer 6. Similar profiles. No ICD duplication here.
    story.append(_p("6. Похожие лабораторные профили", h2))
    local_rows = result.get("similar_cases", {}).get("local_confirmed", [])[:5]
    if local_rows:
        story.append(_p("Локально подтверждённые обезличенные случаи врача; retrieval не переобучает ML автоматически.", small))
        rows = [["Сходство", "Подтверждённый класс", "Диагноз врача"]]
        for item in local_rows:
            rows.append([
                f"{item.get('similarity', 0) * 100:.0f}%",
                CLASSES.get(item.get("class_code"), item.get("class_code")),
                item.get("confirmed_diagnosis") or "—",
            ])
        story.append(_table(rows, [25 * mm, 82 * mm, 68 * mm], small))

    story.append(_p("Reference-выборка кейса: сходство лабораторных профилей, не независимое клиническое доказательство.", small))
    rows = [["Сходство", "Класс", "Причина в выборке", "Возраст / пол"]]
    for item in result.get("similar_cases", {}).get("reference_dataset", [])[:5]:
        sex = "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else str(item.get("sex") or "—")
        rows.append([
            f"{item.get('similarity', 0) * 100:.0f}%",
            CLASSES.get(item.get("class_code"), item.get("class_code")),
            item.get("deficiency_cause") or "—",
            f"{item.get('age_years', '—')} / {sex}",
        ])
    if len(rows) > 1:
        story.append(_table(rows, [22 * mm, 73 * mm, 55 * mm, 25 * mm], small))

    # Layer 7. AI editor. Patient-facing text is never rendered in this section.
    story.append(_p("7. AI-редактор: объяснение для врача", h2))
    story.append(_p(
        "AI не меняет class_code, Model score и клинические правила; ФИО ему не передаётся. Пациентская версия отделена и помещается только на последнюю страницу «Лист для объяснения пациенту».",
        small,
    ))
    if ai_text:
        story.extend(_markdown_flowables(ai_text, normal, small, h2, h3))
    else:
        story.append(_p("Генеративное объяснение не сформировано. Остальные шесть вычислительных слоёв и итог скрининга от него не зависят.", normal))

    story.append(_p("Источники", h2))
    for source in result.get("sources", []):
        destination = source.get("url") or source.get("location", "")
        story.append(_p(f"{source.get('title')}. {destination}", small))

    # Standalone patient page: concise and free of technical implementation details.
    story.append(PageBreak())
    story.append(_p("Лист для объяснения пациенту", h1))
    story.append(_p("Эта страница помогает наглядно обсудить результат с врачом. Она не является диагнозом или назначением лечения.", warn))

    story.append(_p("Что показал скрининг", h2))
    if ai_patient_text:
        # Patient text may still contain Markdown bullets/bold despite the prompt.
        # Render it instead of leaking raw #/** markers into the PDF.
        patient_flow = _markdown_flowables(ai_patient_text, patient_style, patient_style, h2, h3)
        story.extend(patient_flow)
    else:
        story.append(_p(communication.get("plain_summary", final.get("label", "")), patient_style))

    if communication.get("key_facts"):
        story.append(_p("Ключевые факты", h2))
        for fact in communication["key_facts"][:4]:
            story.append(_p("• " + fact, normal))

    story.append(_p("Что пока требует уточнения", h2))
    story.append(_p(communication.get("uncertainty_plain", uncertainty.get("message", "")), normal))

    if result.get("causal_chains"):
        chain = result["causal_chains"][0]
        markers = "; ".join(chain.get("markers", [])[:3]) or "лабораторные данные"
        causes = "; ".join(chain.get("causes", [])[:2]) or "причина требует уточнения"
        story.append(_p("Как могут быть связаны данные", h2))
        story.append(_table([
            ["Что видим", "Рабочая гипотеза", "Что может быть связано"],
            [markers, chain.get("hypothesis", final.get("label", "")), causes],
        ], [58 * mm, 55 * mm, 62 * mm], small))

    story.append(_p("Что поможет уточнить картину", h2))
    story.append(_p(communication.get("why_next_step", "Следующий шаг определяется врачом после сопоставления результата с клинической картиной."), callout))
    story.append(_p(
        "Окончательную интерпретацию, обсуждение индивидуальных обстоятельств и клиническое решение осуществляет врач.",
        small,
    ))

    doc.build(story, onFirstPage=_page_theme, onLaterPages=_page_theme)
    return buf.getvalue()
