from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from screening.casebase import add_case, delete_case, list_cases
from screening.communication import build_patient_conversation
from screening.context_ai import parse_clinical_context_with_ollama
from screening.importers import extract_document_text, read_upload
from screening.rag import BOOKS_DIR, generate_ai_explanation, ollama_status, refresh_knowledge_cache
from screening.report import generate_pdf
from screening.schema import (
    CLASSES,
    DEFAULT_LABS,
    DISCLAIMER,
    LABS,
    SYMPTOM_LABELS,
    UNIT_NOTE,
    convert_value,
    field,
)
from screening.service import screen
from screening.similar import local_confirmed_neighbors
from screening.sources import all_sources

st.set_page_config(page_title="Meditron — скрининг анемии и дефицитов", page_icon="🩺", layout="wide")

ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "midia_logo.png"


def _apply_theme() -> None:
    """Light clinical theme aligned with the team Figma reference.

    Only presentation is changed here; Streamlit controls and backend behavior stay intact.
    """
    st.markdown(
        """
        <style>
        :root {
          --midia-bg: #f3f6ff;
          --midia-panel: #ffffff;
          --midia-soft: #cdd8ff;
          --midia-deep: #21108e;
          --midia-accent: #323fa6;
          --midia-border: #dde4ff;
          --midia-text: #202433;
          --midia-muted: #667085;
        }
        [data-testid="stAppViewContainer"] { background: var(--midia-bg); color: var(--midia-text); }
        [data-testid="stHeader"] { background: rgba(243,246,255,.86); }
        .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
        h1, h2, h3 { color: var(--midia-deep); letter-spacing: -0.02em; }
        [data-testid="stCaptionContainer"], .stCaption { color: var(--midia-muted); }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--midia-panel);
          border: 1px solid var(--midia-border) !important;
          border-radius: 18px;
          box-shadow: 0 6px 22px rgba(33,16,142,.055);
        }
        div[data-testid="stMetric"] {
          background: var(--midia-panel); border: 1px solid var(--midia-border);
          border-radius: 16px; padding: .8rem 1rem;
        }
        .stButton > button, [data-testid="stPopover"] button {
          border-radius: 12px !important; border-color: #b9c7ff !important;
          transition: all .15s ease;
        }
        .stButton > button:hover, [data-testid="stPopover"] button:hover {
          border-color: var(--midia-accent) !important; color: var(--midia-deep) !important;
          box-shadow: 0 4px 14px rgba(50,63,166,.12);
        }
        button[kind="primary"] {
          background: var(--midia-accent) !important; border-color: var(--midia-accent) !important;
          color: white !important; font-weight: 650 !important;
        }
        [data-baseweb="tab-list"] { gap: .35rem; background: transparent; }
        [data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding-left: 1rem; padding-right: 1rem; }
        [aria-selected="true"][data-baseweb="tab"] { color: var(--midia-deep) !important; font-weight: 650; }
        [data-testid="stFileUploaderDropzone"] {
          background: #fafbff; border-color: #b9c7ff; border-radius: 16px;
        }
        [data-testid="stAlert"] { border-radius: 14px; }
        input, textarea { border-radius: 10px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_theme()


def _default_rows() -> list[dict[str, str]]:
    return [
        {"Показатель": LABS[k]["label"], "Значение": "", "Единица": LABS[k]["unit"]}
        for k in DEFAULT_LABS
    ]


def _reset() -> None:
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state.update(
        {
            "rows": _default_rows(),
            "revision": 0,
            "full_name": "",
            "age": "",
            "sex": "Не указан",
            "complaints": "",
            "history_text": "",
            "medications": "",
            "input_mode": "Импортировать файл",
            "authorization": False,
            "use_ai_context_parser": False,
            "import_context_source": "",
        }
    )


def _ensure_state() -> None:
    if "rows" not in st.session_state:
        _reset()


def _merge_text(existing: str, incoming: str) -> str:
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming.lower() in existing.lower():
        return existing
    return existing + "\n" + incoming


def _symptoms_from_text(text: str) -> list[str]:
    low = str(text or "").lower().replace("ё", "е")
    if not low:
        return []
    aliases = {
        "Утомляемость / слабость": ["утом", "слабост", "астени"],
        "Снижение переносимости физической нагрузки": ["переносимост", "нагрузк"],
        "Одышка при нагрузке": ["одыш"],
        "Головокружение": ["головокруж"],
        "Предобморочное состояние / обморок": ["обмор", "предобмор", "синкоп"],
        "Сердцебиение": ["сердцеби", "тахикард"],
        "Головная боль": ["головн", "цефалг"],
        "Бледность кожи / слизистых": ["бледн"],
        "Зябкость / непереносимость холода": ["зяб", "холод"],
        "Извращение вкуса / пика": ["пика", "мел", "лед", "лёд", "извращение вкуса"],
        "Выпадение волос / ломкость ногтей": ["выпадение волос", "ломк", "ногт"],
        "Синдром беспокойных ног": ["беспокойных ног"],
        "Парестезии / покалывание": ["парестез", "покалыв"],
        "Онемение / снижение чувствительности": ["онемен", "чувствительност"],
        "Нарушение походки / равновесия": ["походк", "равновес"],
        "Снижение концентрации / памяти": ["концентрац", "памят"],
        "Глоссит / жжение языка": ["глоссит", "жжение языка"],
        "Стоматит / изменения слизистой полости рта": ["стоматит"],
        "Снижение аппетита": ["аппетит"],
        "Непреднамеренная потеря веса": ["потеря веса", "снижение веса", "похуд"],
        "Хроническая диарея": ["диар", "жидкий стул"],
        "Боль / дискомфорт в животе": ["боль в живот", "дискомфорт в живот"],
        "Чёрный стул / мелена": ["мелена", "черный стул", "чёрный стул"],
        "Кровь в стуле": ["кровь в стуле"],
        "Обильные / длительные менструации": ["обильн", "менорраг", "длительные менстру"],
        "Другие признаки кровопотери": ["кровопот", "кровотеч"],
        "Длительная лихорадка / субфебрилитет": ["лихорад", "субфеб"],
        "Суставные боли / признаки хронического воспаления": ["сустав", "артралг"],
    }
    options = set(SYMPTOM_LABELS.values())
    return [label for label, terms in aliases.items() if label in options and any(term in low for term in terms)]


def _safe_book_name(name: str) -> str:
    base = Path(name or "source.pdf").name
    stem = re.sub(r"[^A-Za-zА-Яа-я0-9._ -]+", "_", base).strip(" .")
    return stem if stem.lower().endswith(".pdf") else stem + ".pdf"


def _apply_import(record: dict) -> None:
    merged = dict(record)
    if isinstance(record.get("labs"), dict):
        merged.update(record["labs"])

    units = record.get("units", {}) if isinstance(record.get("units"), dict) else {}
    rows: list[dict[str, str]] = []
    for key, value in merged.items():
        canonical = field(key)
        if canonical not in LABS:
            continue
        if isinstance(value, dict):
            raw_value = value.get("value", "")
            unit = value.get("unit", "")
        else:
            raw_value = value
            unit = units.get(key, LABS[canonical]["unit"])
        if raw_value is None or str(raw_value).strip().lower() in {"", "nan", "none"}:
            continue
        rows.append({
            "Показатель": LABS[canonical]["label"],
            "Значение": str(raw_value),
            "Единица": unit or LABS[canonical]["unit"],
        })

    st.session_state["rows"] = rows or _default_rows()
    st.session_state["revision"] = st.session_state.get("revision", 0) + 1

    # PDF/structured import may prefill the patient card, but every field remains editable.
    parsed_name = str(record.get("full_name", "") or "").strip()
    if parsed_name and not str(st.session_state.get("full_name", "")).strip():
        st.session_state["full_name"] = parsed_name

    if record.get("age_years", record.get("Возраст")) not in (None, ""):
        st.session_state["age"] = str(record.get("age_years", record.get("Возраст", "")))
    sex = str(record.get("sex", record.get("Пол", ""))).upper()
    if sex in {"F", "Ж", "ЖЕНСКИЙ", "FEMALE"}:
        st.session_state["sex"] = "Женский"
    elif sex in {"M", "М", "МУЖСКОЙ", "MALE"}:
        st.session_state["sex"] = "Мужской"

    st.session_state["complaints"] = _merge_text(
        st.session_state.get("complaints", ""), str(record.get("complaints", "") or "")
    )
    st.session_state["history_text"] = _merge_text(
        st.session_state.get("history_text", ""), str(record.get("history", "") or "")
    )
    st.session_state["medications"] = _merge_text(
        st.session_state.get("medications", ""), str(record.get("medications", "") or "")
    )


def _raw_from_form(rows: pd.DataFrame, units_confirmed: bool) -> tuple[dict, list[str]]:
    sex = {"Женский": "F", "Мужской": "M"}.get(st.session_state.get("sex"), "")
    raw = {
        "patient_id": "anonymous",
        "age_years": st.session_state.get("age", ""),
        "sex": sex,
        "pregnant": False,
        "complaints": st.session_state.get("complaints", ""),
        # Для UX жалобы и симптомы объединены. Структурированные симптомы извлекаются
        # локально из этого же текста только для причинного слоя и не становятся ML-признаками.
        "symptoms": _symptoms_from_text(st.session_state.get("complaints", "")),
        "history": st.session_state.get("history_text", ""),
        "medications": st.session_state.get("medications", ""),
        "labs": {},
    }
    errors: list[str] = []
    for row_num, row in enumerate(rows.fillna("").to_dict("records"), 1):
        label = str(row.get("Показатель", "")).strip()
        value = str(row.get("Значение", "")).strip()
        unit = str(row.get("Единица", "")).strip()
        if not value:
            continue
        canonical = field(label)
        if canonical not in LABS:
            errors.append(f"Строка {row_num}: неизвестный показатель «{label}»")
            continue
        if canonical in raw["labs"]:
            errors.append(f"Строка {row_num}: показатель «{LABS[canonical]['label']}» указан повторно")
            continue
        raw["labs"][canonical] = {"value": value, "unit": unit}

        normalized, canonical_unit, _ = convert_value(
            canonical, value, unit, units_confirmed=units_confirmed
        )
        maximum = LABS[canonical].get("max")
        if normalized is not None and (
            normalized < 0 or (maximum is not None and normalized > float(maximum))
        ):
            errors.append(
                f"Строка {row_num}: {LABS[canonical]['label']} = {normalized:g} вне технического диапазона "
                f"0–{float(maximum):g} {canonical_unit}. Проверьте опечатку и единицы."
            )
    return raw, errors


def _fingerprint(raw: dict, units_confirmed: bool) -> str:
    return hashlib.sha256(
        json.dumps([raw, units_confirmed], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _translated_exclusions(items: list[dict]) -> pd.DataFrame:
    status = {
        "abnormal": "есть лабораторный сигнал",
        "incomplete": "не хватает данных",
        "no_screening_signal": "по имеющимся данным сигнала нет",
    }
    rows = []
    for item in items:
        rows.append(
            {
                "Направление": item.get("title", ""),
                "Статус": status.get(item.get("status"), item.get("status", "")),
                "Что видно": item.get("text", "—") or "—",
                "Не хватает": ", ".join(item.get("missing", [])) or "—",
            }
        )
    return pd.DataFrame(rows)


def _suggest_units(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Fill a canonical unit when the row has no unit or an unsupported one.

    Imported supported alternative units are preserved so explicit conversions still work.
    """
    out = frame.copy()
    changed = False
    for idx, row in out.fillna("").iterrows():
        label = str(row.get("Показатель", "")).strip()
        canonical = field(label)
        if canonical not in LABS:
            continue
        unit = str(row.get("Единица", "")).strip()
        expected = str(LABS[canonical].get("unit", ""))
        supported = False
        if unit:
            try:
                value, _, _ = convert_value(canonical, 1, unit, units_confirmed=True)
                supported = value is not None
            except Exception:
                supported = False
        if not unit or not supported:
            out.at[idx, "Единица"] = expected
            changed = True
    return out, changed


def _render_reference_popover() -> None:
    st.markdown("### Справочник")
    class_rows = [{"Код": code, "Скрининговый класс": label} for code, label in CLASSES.items()]
    st.dataframe(pd.DataFrame(class_rows), hide_index=True, use_container_width=True, height=330)
    st.caption("12 целевых классов скрининга. Это рабочие классы системы, а не автоматически подтверждённые диагнозы.")
    with st.expander("Источники", expanded=False):
        for source in all_sources():
            st.markdown(f"**{source.get('title', source.get('id'))}**")
            if source.get("note"):
                st.caption(source["note"])
            detail = source.get("url") or source.get("location") or ""
            if detail:
                st.caption(detail)


def _render_faq_popover() -> None:
    st.markdown("### Как устроена система")
    st.markdown(
        """
**Почему несколько слоёв?** Один алгоритм не должен быть единственной точкой истины в медицинском прототипе. Независимые слои дают врачу возможность видеть, где правила и статистическая модель согласны, а где остаётся неопределённость.

**Важно:** карточка пациента и итоговое резюме не являются слоями — это входные данные и сводный вывод.

**7 вычислительных слоёв:**
1. импорт, нормализация и контроль входных данных;
2. прозрачная клиническая логика;
3. независимая ML-проверка по 12 классам;
4. карта неопределённости;
5. причинно-следственные связи с клиническим контекстом;
6. похожие лабораторные профили;
7. локальное AI/RAG-объяснение уже рассчитанного результата.

**Что влияет на ML?** Возраст, пол и лабораторные признаки из обучающей матрицы. Жалобы, анамнез и препараты в ML не подмешиваются; они используются отдельно в причинном и объясняющем слоях.

**Почему AI в самом конце?** LLM не пересчитывает class_code, лабораторные значения и Model score. Даже без Ollama медицинское ядро продолжает работать.
"""
    )
    metrics_path = Path(__file__).parent / "models" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.markdown("**Техническая оценка прототипа**")
        st.caption(
            f"Balanced accuracy {metrics.get('balanced_accuracy', 0):.3f} · "
            f"Macro F1 {metrics.get('macro_f1', 0):.3f} · строк датасета {metrics.get('rows')}. "
            "Это внутренняя техническая оценка, а не клиническая валидация."
        )


def _render_tools_popover(current_result: dict | None) -> None:
    st.markdown("### Служебные инструменты")
    tab_cases, tab_books = st.tabs(["Подтверждённые случаи", "Локальные PDF для RAG"])

    with tab_cases:
        st.caption(
            "Это не история пациента. Сохраняются только обезличенные лабораторные признаки, возраст, пол и подтверждённая врачом метка. "
            "Case-base влияет только на поиск похожих профилей и не переобучает ML автоматически."
        )
        if current_result:
            final = current_result.get("final", {})
            codes = list(CLASSES)
            default_idx = codes.index(final.get("class_code")) if final.get("class_code") in codes else 0
            confirmed_code = st.selectbox(
                "Подтверждённый скрининговый класс",
                codes,
                index=default_idx,
                format_func=lambda x: CLASSES[x],
                key="tool_case_class",
            )
            c1, c2 = st.columns(2)
            confirmed_diag = c1.text_input("Диагноз врача", key="tool_case_diag")
            confirmed_icd = c2.text_input("МКБ-10", key="tool_case_icd")
            if st.button("Добавить текущий обезличенный случай", key="tool_case_add", use_container_width=True):
                try:
                    added = add_case(current_result["patient"], confirmed_code, confirmed_diag, confirmed_icd)
                    st.success(f"Случай {added['case_id']} добавлен без ФИО и свободного текста.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Не удалось сохранить случай: {exc}")
        else:
            st.info("После расчёта текущий профиль можно будет добавить сюда как подтверждённый случай.")

        stored = list_cases()
        if stored:
            st.dataframe(
                pd.DataFrame([
                    {
                        "ID": x.get("case_id"),
                        "Класс": CLASSES.get(x.get("class_code"), x.get("class_code")),
                        "Диагноз": x.get("confirmed_diagnosis") or "—",
                        "МКБ-10": x.get("icd10") or "—",
                    }
                    for x in stored
                ]),
                hide_index=True,
                use_container_width=True,
                height=220,
            )
            delete_id = st.selectbox("Удалить случай", ["—"] + [x.get("case_id") for x in stored], key="tool_case_delete_select")
            if st.button("Удалить выбранный", disabled=delete_id == "—", key="tool_case_delete"):
                if delete_case(delete_id):
                    st.success("Случай удалён из локальной retrieval-базы.")
                    st.rerun()
        else:
            st.caption("Локальная база пока пуста.")

    with tab_books:
        st.caption(
            "Папка books/ поставляется пустой. Сюда можно локально добавить разрешённые текстовые PDF с рекомендациями или протоколами. "
            "Документы пациентов сюда загружать не следует."
        )
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        book_upload = st.file_uploader("Добавить PDF", type=["pdf"], key="tool_book_upload")
        if st.button("Добавить источник", disabled=book_upload is None, key="tool_book_add"):
            try:
                name = _safe_book_name(book_upload.name)
                (BOOKS_DIR / name).write_bytes(book_upload.getvalue())
                refresh_knowledge_cache()
                st.success(f"Добавлен источник: {name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось добавить PDF: {exc}")
        book_files = sorted(p.name for p in BOOKS_DIR.glob("*.pdf"))
        if book_files:
            st.caption("Локальная библиотека: " + ", ".join(book_files))
            to_delete = st.selectbox("Удалить PDF", ["—"] + book_files, key="tool_book_delete_select")
            if st.button("Удалить выбранный PDF", disabled=to_delete == "—", key="tool_book_delete"):
                (BOOKS_DIR / to_delete).unlink(missing_ok=True)
                refresh_knowledge_cache()
                st.success("Источник удалён.")
                st.rerun()
        else:
            st.caption("Локальных PDF пока нет; базовый реестр источников работает независимо.")


_ensure_state()
if st.session_state.get("pending_import_record") is not None:
    _apply_import(st.session_state.pop("pending_import_record"))

current_result = st.session_state.get("result")
logo_col, title_col, ref_col, faq_col, tools_col, new_col = st.columns([0.55, 5.15, 1.05, 0.85, 1.0, 1.25], vertical_alignment="center")
with logo_col:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=72)
with title_col:
    st.title("Meditron · скрининг анемии и дефицитов")
    st.caption("Команда «Мидия» · кейс Сеченовского университета · врачебный интерфейс · взрослые 18+ вне беременности")
with ref_col:
    with st.popover("Справочник", use_container_width=True):
        _render_reference_popover()
with faq_col:
    with st.popover("FAQ", use_container_width=True):
        _render_faq_popover()
with tools_col:
    with st.popover("Инструменты", use_container_width=True):
        _render_tools_popover(current_result)
with new_col:
    st.button("＋ Новый пациент", on_click=_reset, use_container_width=True, type="primary")

st.info(
    "Сервис сопоставляет лабораторные данные, клинические правила и ML, чтобы уменьшить рутинную информационную нагрузку. "
    "Карта неопределённости и причинные связи делают результат объяснимым, но окончательное решение остаётся врачу."
)
st.warning(DISCLAIMER)

authorized = st.checkbox(
    "Подтверждаю, что являюсь медицинским работником / уполномоченным пользователем и имею законное основание для обработки внесённых данных пациента.",
    key="authorization",
    help="Организационный safety-check прототипа; он не заменяет локальные документы клиники и юридическую проверку процесса.",
)

st.subheader("Пациент и клинический контекст")
c1, c2, c3 = st.columns([1.8, 0.8, 0.9])
c1.text_input(
    "ФИО пациента",
    key="full_name",
    placeholder="Введите ФИО пациента",
    help="Нужно только для шапки текущего PDF. ФИО не входит в ML, не отправляется локальной LLM и не сохраняется в case-base.",
)
c2.text_input("Возраст, лет", key="age", help="Возраст входит в ML и используется при поиске похожих профилей. Проект предназначен только для 18+.")
c3.selectbox("Пол", ["Не указан", "Женский", "Мужской"], key="sex", help="Пол входит в ML и определяет порог гемоглобина по условиям кейса.")

left, right = st.columns(2)
left.text_area(
    "Жалобы / симптомы",
    key="complaints",
    height=105,
    placeholder="Например: слабость около 3 месяцев, головокружение, одышка при нагрузке…",
    help="В ML не входят. Типовые симптомы автоматически выделяются из текста только для причинного и объясняющего слоёв.",
)
right.text_area(
    "Анамнез / хронические заболевания / значимые события",
    key="history_text",
    height=105,
    placeholder="Например: обильные менструации, ВЗК, целиакия, операции, донорство, хроническое воспаление…",
    help="Используется как клинический контекст причинного и AI/RAG-слоёв; не переписывает ML-класс.",
)
st.text_area(
    "Постоянные препараты",
    key="medications",
    height=70,
    placeholder="Например: метформин, ингибитор протонной помпы…",
    help="Используются только как клинический контекст и не входят в ML.",
)

st.subheader("Лабораторные показатели")
input_mode = st.radio(
    "Как добавить анализы?",
    ["Импортировать файл", "Ввести вручную"],
    horizontal=True,
    key="input_mode",
)

if input_mode == "Импортировать файл":
    upload = st.file_uploader(
        "Загрузите файл одного пациента",
        type=None,
        key="patient_upload",
        help="Поддерживаются текстовые PDF, обычные TXT, CSV/TSV, JSON, XLSX/XLS и HTML. Многостраничный PDF одного пациента объединяется в одну карточку.",
    )
    st.checkbox(
        "Использовать локальную AI-модель для разбора клинического контекста",
        key="use_ai_context_parser",
        help=(
            "Опционально: локальная Ollama только структурирует текст документа по полям «Жалобы / симптомы», "
            "«Анамнез» и «Препараты». Она не меняет лабораторные значения, ML-класс или клинические правила. "
            "ФИО перед отправкой в локальную модель удаляется."
        ),
    )
    if st.session_state.get("use_ai_context_parser"):
        status = ollama_status(timeout=0.8)
        if status.get("server") and status.get("model_installed"):
            st.caption(f"Локальный AI-парсер готов · {status.get('model')}")
        elif status.get("server"):
            st.caption(f"Ollama запущена, но модель {status.get('model')} не найдена — при импорте будет использован обычный parser.")
        else:
            st.caption("Ollama недоступна — при импорте будет использован обычный parser без потери лабораторного разбора.")

    if st.button("Распознать файл", disabled=upload is None):
        try:
            payload = upload.getvalue()
            records, messages = read_upload(payload, upload.name)
            record = dict(records[0])
            context_source = "детерминированный parser"

            if st.session_state.get("use_ai_context_parser"):
                status = ollama_status(timeout=0.8)
                if status.get("server") and status.get("model_installed"):
                    doc_text = extract_document_text(payload, upload.name)
                    if doc_text.strip():
                        structured = parse_clinical_context_with_ollama(
                            doc_text, full_name=str(record.get("full_name", "") or "")
                        )
                        if structured.get("ok"):
                            mapping = {"complaints": "complaints", "history": "history", "medications": "medications"}
                            for src_key, dst_key in mapping.items():
                                value = str(structured.get(src_key, "") or "").strip()
                                if value:
                                    record[dst_key] = value
                            context_source = f"локальная AI-структуризация ({structured.get('model')})"
                            messages.append(
                                "Клинический контекст структурирован локальной AI-моделью. Ничего не считается подтверждённым автоматически — проверьте карточку пациента вручную."
                            )
                        else:
                            messages.append(
                                "AI-разбор клинического контекста не выполнен; сохранён результат обычного parser. "
                                f"Причина: {structured.get('error') or 'неизвестная ошибка'}."
                            )
                    else:
                        messages.append("В файле нет доступного текстового слоя для AI-разбора контекста; использован обычный parser.")
                else:
                    messages.append("Локальная AI-модель недоступна; использован обычный parser клинического контекста.")

            st.session_state["pending_import_record"] = record
            st.session_state["import_messages"] = messages
            st.session_state["import_context_source"] = context_source
            st.session_state["import_ok"] = True
            st.rerun()
        except Exception as exc:
            st.session_state["import_ok"] = False
            st.error(str(exc))
    if st.session_state.get("import_ok"):
        st.success("Файл распознан и предзаполнил доступные поля карточки.")
        source = st.session_state.get("import_context_source")
        if source:
            st.caption("Клинический контекст: " + source + ".")
        st.error("⚠️ ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА: перед расчётом сверьте карточку пациента, распознанные значения и единицы с исходным документом. Любое поле можно исправить вручную.")
    for msg in st.session_state.get("import_messages", []):
        if any(word in msg.lower() for word in ("проверь", "разные значения", "обязательно")):
            st.warning(msg)
        else:
            st.caption("• " + msg)
else:
    st.caption("Для основных показателей единицы подставлены автоматически. При добавлении новой строки система предложит каноническую единицу выбранного анализа.")

st.markdown("**Проверка и редактирование перед расчётом**")
st.caption(UNIT_NOTE)
unit_options = sorted({str(v.get("unit", "")) for v in LABS.values() if v.get("unit")} | {"г/дл", "мг/дл", "нг/мл", "пмоль/л", "нмоль/л"})
rows = st.data_editor(
    pd.DataFrame(st.session_state["rows"]),
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="editor_" + str(st.session_state.get("revision", 0)),
    column_config={
        "Показатель": st.column_config.SelectboxColumn(options=[v["label"] for v in LABS.values()], required=True, width="large"),
        "Значение": st.column_config.TextColumn(width="medium"),
        "Единица": st.column_config.SelectboxColumn(options=unit_options, width="medium", help="Если единица пустая или не подходит выбранному показателю, будет предложена каноническая единица."),
    },
)
rows, units_changed = _suggest_units(rows)
if units_changed:
    st.session_state["rows"] = rows.fillna("").to_dict("records")
    st.session_state["revision"] = st.session_state.get("revision", 0) + 1
    st.rerun()

units_confirmed = st.checkbox(
    "Если единица в исходном бланке отсутствует, считать указанную в таблице единицу подтверждённой",
    value=False,
)
raw, form_errors = _raw_from_form(rows, units_confirmed)
for error in form_errors:
    st.error(error)

if not authorized:
    st.info("Перед расчётом подтвердите статус уполномоченного пользователя и законное основание для обработки данных.")

run_disabled = bool(form_errors) or not authorized
if st.button("Выполнить многослойный скрининг", type="primary", disabled=run_disabled):
    try:
        with st.spinner("Проверяем клинические правила, ML и карту неопределённости…"):
            result = screen(raw, units_confirmed=units_confirmed)
        st.session_state["result"] = result
        st.session_state["result_fingerprint"] = _fingerprint(raw, units_confirmed)
        st.session_state.pop("ai_explanation", None)
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    stale = st.session_state.get("result_fingerprint") != _fingerprint(raw, units_confirmed)
    if stale:
        st.warning("Карточка или анализы изменены после расчёта. Пересчитайте результат перед выгрузкой отчёта.")

    final = result["final"]
    clinical = result["clinical"]
    result["patient_conversation"] = build_patient_conversation(result)

    st.divider()
    st.subheader("Предварительный результат")
    st.markdown(f"### {final['label']}")
    st.caption(f"Класс по 12-классовой схеме: `{final.get('class_code') or '—'}` · {final.get('confidence_band', '')}")

    brief = result.get("doctor_brief", {})
    if brief.get("priority"):
        with st.container(border=True):
            st.markdown("**Ключевое для врача**")
            for item in brief["priority"]:
                st.write("•", item)
            if brief.get("anchors"):
                st.caption("Ключевые показатели: " + " · ".join(brief["anchors"]))

    conversation_mode = st.toggle(
        "Режим разговора с пациентом",
        value=False,
        key="conversation_mode",
        help="Показывает медицинскую логику без технического шума. Система не ведёт разговор вместо врача.",
    )
    if conversation_mode:
        comm = result.get("patient_conversation", {})
        with st.container(border=True):
            st.markdown("### Наглядное объяснение результата")
            st.markdown("**Что мы уже видим**")
            st.write(comm.get("plain_summary", ""))
            if comm.get("key_facts"):
                cols = st.columns(min(3, len(comm["key_facts"])))
                for i, item in enumerate(comm["key_facts"][:3]):
                    cols[i].metric("Ключевой факт", item)

            ranking = comm.get("model_ranking", [])
            if ranking:
                st.markdown("**Как система ранжирует рабочие варианты**")
                st.caption("Шкала 0–100 показывает относительный Model score между 12 классами. Это не процент вероятности диагноза.")
                for item in ranking:
                    st.write(f"**{item['label']}** — модельный рейтинг {item['rating']}/100")
                    st.progress(item["rating"])

            st.markdown("**Что пока остаётся неопределённым**")
            st.write(comm.get("uncertainty_plain", ""))

            if comm.get("causal_simple"):
                st.markdown("**Как могут быть связаны данные**")
                for chain in comm["causal_simple"]:
                    a, arrow1, b, arrow2, c = st.columns([1.5, 0.15, 1.35, 0.15, 1.5])
                    a.markdown("**Что видим**")
                    a.write(chain["observed"])
                    arrow1.markdown("### →")
                    b.markdown("**Рабочая гипотеза**")
                    b.write(chain["hypothesis"])
                    arrow2.markdown("### →")
                    c.markdown("**Что может быть связано**")
                    c.write(chain["possible_context"])

            st.markdown("**Что поможет уточнить картину**")
            st.info(comm.get("why_next_step", ""))

    if final.get("rule_class") and final.get("model_class") and not final.get("agreement"):
        st.warning(
            f"Клинические правила и ML расходятся: правила → `{final['rule_class']}`, ML → `{final['model_class']}`. "
            "Система не усредняет конфликт и показывает его как дополнительную неопределённость."
        )
    for alert in clinical.get("alerts", []):
        st.error(alert)

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Клиническая логика", "ML-проверка", "Неопределённость", "Причинные связи", "Похожие профили", "AI-объяснение"]
    )

    with t1:
        st.caption(
            "Прозрачные лабораторные правила служат независимой проверкой ML и показывают, какие данные поддерживают или ослабляют рабочую гипотезу."
        )
        for finding in clinical.get("findings", []):
            if finding.get("severity") == "warning":
                st.warning(finding["text"])
            else:
                st.write("•", finding["text"])

        st.markdown("**Проверка других причин анемии**")
        st.caption(
            "Если происхождение анемии остаётся неясным, система отдельно проверяет доступные лабораторные признаки значимой почечной дисфункции, явного гемолиза и выраженного тиреоидного направления. Это узкий safety-check, а не универсальный список дифференциальных диагнозов."
        )
        exclusion_items = clinical.get("exclusions", [])
        attention = [x for x in exclusion_items if x.get("status") in {"abnormal", "incomplete"}]
        quiet = [x for x in exclusion_items if x.get("status") == "no_screening_signal"]
        if attention:
            st.dataframe(_translated_exclusions(attention), hide_index=True, use_container_width=True)
        else:
            st.success("По доступным данным дополнительных лабораторных сигналов из этого safety-check не выявлено.")
        if quiet:
            with st.expander("Проверенные направления без лабораторного сигнала", expanded=False):
                st.dataframe(_translated_exclusions(quiet), hide_index=True, use_container_width=True)

        st.markdown("**МКБ-10 — справка для врача, не автокодирование**")
        coding = result.get("coding", {})
        st.caption(coding.get("note", ""))
        if coding.get("candidates"):
            st.dataframe(pd.DataFrame(coding["candidates"]), hide_index=True, use_container_width=True)
        else:
            st.write("Для этого скринингового класса автоматический кандидат кода не предлагается.")

    with t2:
        st.caption(
            "ML — независимая статистическая проверка по предоставленному датасету. Model score сравнивает классы между собой и не является клинической вероятностью диагноза."
        )
        model = result.get("model")
        if model:
            top_score = float(model.get("top_score", 0))
            margin = float(model.get("margin", 0))
            if top_score >= 0.75 and margin >= 0.25:
                human_ml = "Модель выраженно предпочитает ведущий класс среди 12 вариантов."
            elif top_score >= 0.55 and margin >= 0.12:
                human_ml = "У модели есть ведущий класс, но преимущество над альтернативами умеренное."
            else:
                human_ml = "Распределение ML-score близкое: вывод модели стоит трактовать как менее устойчивый."
            st.info(human_ml)

            ranked = list(model.get("class_scores", {}).items())
            st.markdown("**Как модель ранжирует наиболее похожие классы**")
            st.caption(
                "Шкала 0–100 — визуальный перевод внутреннего Model score для сравнения классов между собой. "
                "Это НЕ процент вероятности диагноза."
            )
            for code, score in ranked[:4]:
                label = CLASSES.get(code, code)
                value = max(0, min(100, int(round(float(score) * 100))))
                st.write(f"**{label}** — модельный рейтинг {value}/100")
                st.progress(value)

            with st.expander("Технические ML-scores", expanded=False):
                scores = [
                    {"Класс": CLASSES.get(code, code), "Model score": round(float(score), 4)}
                    for code, score in ranked
                ]
                st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)
                st.caption(model.get("score_note", ""))

            if model.get("sensitivity"):
                st.markdown("**Локальная чувствительность ML-вывода**")
                st.caption(
                    "По одному показателю значение заменяется медианой обучающей выборки и измеряется изменение ведущего score. Это тест устойчивости, а не причинный вклад."
                )
                sens = pd.DataFrame(model["sensitivity"])
                sens["Показатель"] = sens["feature"].map(lambda x: LABS.get(x, {}).get("label", x))
                show = sens[["Показатель", "value", "median", "score_change"]].copy()
                show.columns = ["Показатель", "Значение пациента", "Медиана выборки", "Изменение score"]
                st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("ML-слой недоступен. Клинические правила продолжают работать независимо.")

    with t3:
        st.caption(
            "Карта неопределённости показывает, между какими гипотезами остаётся сомнение и какое недостающее измерение способно лучше всего его уменьшить. Конкурирующие гипотезы не исключают сопутствующие состояния."
        )
        unc = result.get("uncertainty", {})
        if unc.get("top_hypothesis"):
            st.markdown(f"### {unc['top_hypothesis']['label']} ↔ {unc['alternative']['label']}")
            st.write(unc.get("message", ""))
            if unc.get("ambiguity_factors"):
                st.markdown("**Почему остаётся неопределённость**")
                for factor in unc["ambiguity_factors"]:
                    st.write(f"• **{factor['factor']}** — {factor['text']}")
            best = unc.get("best_next_measurement")
            if best:
                st.success(f"**Наиболее информативное недостающее измерение:** {best['label']}\n\n{best['why']}")
            if unc.get("items"):
                dfu = pd.DataFrame(unc["items"])
                cols = [c for c in ["label", "why", "score", "counterfactual_impact", "coverage"] if c in dfu.columns]
                st.dataframe(dfu[cols], hide_index=True, use_container_width=True)
            if unc.get("method_note"):
                st.caption(unc["method_note"])
        else:
            st.info(unc.get("message", "Карта неопределённости недоступна."))

        coexist = result.get("coexisting_signals", [])
        if coexist:
            st.markdown("**Параллельные / сопутствующие сигналы**")
            st.caption("Они могут сосуществовать с ведущей гипотезой и не трактуются как взаимоисключающие альтернативы.")
            for item in coexist:
                st.write("•", item["label"])

    with t4:
        st.caption(
            "Причинный слой не устанавливает причину автоматически. Он связывает лабораторные признаки с рабочей гипотезой и тем контекстом, который действительно найден в жалобах, анамнезе или препаратах."
        )
        for chain in result.get("causal_chains", []):
            with st.container(border=True):
                st.markdown(f"**{chain['label']}**")
                c1, a1, c2, a2, c3 = st.columns([1.45, 0.18, 1.2, 0.18, 1.45])
                c1.markdown("**Что видим**")
                for marker in chain.get("markers", []):
                    c1.write("• " + marker)
                a1.markdown("### →")
                c2.markdown("**Что это поддерживает**")
                c2.write(chain.get("hypothesis", chain.get("label", "")))
                a2.markdown("### →")
                c3.markdown("**Что может объяснять**")
                for cause in chain.get("causes", []):
                    c3.write("• " + cause)

    with t5:
        st.caption(
            "Похожие случаи — дополнительный взгляд на тот же лабораторный профиль. Retrieval по сходству не является самостоятельным доказательством диагноза и не переобучает ML."
        )
        local_refs = local_confirmed_neighbors(result["patient"], k=5)
        result.setdefault("similar_cases", {})["local_confirmed"] = local_refs

        st.markdown("**Локально подтверждённые похожие случаи**")
        if local_refs:
            local_table = [
                {
                    "Сходство": f"{item.get('similarity', 0) * 100:.0f}%",
                    "Подтверждённый класс": CLASSES.get(item.get("class_code"), item.get("class_code")),
                    "Диагноз врача": item.get("confirmed_diagnosis") or "—",
                    "Возраст": item.get("age_years"),
                    "Пол": "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else item.get("sex"),
                }
                for item in local_refs
            ]
            st.dataframe(pd.DataFrame(local_table), hide_index=True, use_container_width=True)
        else:
            st.info("Локальная база подтверждённых случаев пока пуста или нет профилей с достаточным лабораторным пересечением.")

        st.markdown("**Похожие профили из reference-выборки**")
        refs = result.get("similar_cases", {}).get("reference_dataset", [])
        if refs:
            table = [
                {
                    "Сходство": f"{item.get('similarity', 0) * 100:.0f}%",
                    "Класс": CLASSES.get(item.get("class_code"), item.get("class_code")),
                    "Причина в выборке": item.get("deficiency_cause") or "—",
                    "Возраст": item.get("age_years"),
                    "Пол": "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else item.get("sex"),
                    "Совпало лабораторных признаков": item.get("lab_overlap"),
                }
                for item in refs[:5]
            ]
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        else:
            st.caption("Похожие reference-профили не найдены.")

    ai = st.session_state.get("ai_explanation")
    with t6:
        st.caption(
            "Генеративный слой получает уже рассчитанный обезличенный результат и формирует два текста: подробное объяснение врачу и короткое нейтральное объяснение для пациента. Он не меняет расчёт."
        )
        status = ollama_status()
        if status["server"] and status["model_installed"]:
            st.success(f"Ollama подключена · модель {status['model']} готова")
        elif status["server"]:
            st.warning(f"Ollama запущена, но модель {status['model']} не найдена. Установите её: `ollama pull {status['model']}`.")
        else:
            st.info("Ollama сейчас недоступна. Медицинский расчёт уже выполнен; для генеративного объяснения запустите локальную Ollama (`ollama serve`).")

        if st.button("Сформировать / обновить AI-объяснение", key="ai_button"):
            progress_bar = st.progress(0, text="0% · Подготовка…")

            def _progress(percent: int, message: str) -> None:
                progress_bar.progress(percent, text=f"{percent}% · {message}")

            st.session_state["ai_explanation"] = generate_ai_explanation(result, result["patient"], progress=_progress)
            ai = st.session_state.get("ai_explanation")

        ai = st.session_state.get("ai_explanation")
        if ai:
            if ai.get("warning"):
                st.info(ai["warning"])
            st.markdown("**Объяснение для врача**")
            st.markdown(ai["text"])
            if ai.get("patient_text"):
                st.caption("Короткая пациентская версия сформирована отдельно и будет помещена на последнюю страницу PDF «Лист для объяснения пациенту».")
            if ai.get("sources"):
                with st.expander("Какие фрагменты источников использовал AI", expanded=False):
                    for source in ai["sources"]:
                        st.caption(f"{source['source']} · relevance {source['score']:.3f}")
                        st.write(source["text"])
        else:
            st.caption("AI-объяснение ещё не сформировано.")

    st.subheader("Отчёт")
    st.caption("ФИО добавляется только в формируемый PDF и не входит в JSON результата.")
    if not stale:
        ai_text = ai.get("text") if ai else None
        ai_patient_text = ai.get("patient_text") if ai else None
        try:
            pdf = generate_pdf(
                result,
                patient_name=st.session_state.get("full_name", "").strip(),
                ai_text=ai_text,
                ai_patient_text=ai_patient_text,
            )
            c1, c2 = st.columns(2)
            c1.download_button(
                "Скачать врачебный PDF",
                pdf,
                "sechenov_screening_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
            c2.download_button(
                "Скачать обезличенный JSON",
                json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                "screening_result.json",
                "application/json",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Не удалось сформировать PDF: {exc}")
