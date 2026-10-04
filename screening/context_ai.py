from __future__ import annotations

"""Optional local LLM helper for structuring clinical free text.

This module is deliberately separated from the screening core. It never computes
anemia classes, changes laboratory values or influences the trained ML model.
Its only job is to split de-identified document text into three editable fields:
complaints/symptoms, anamnesis/chronic conditions and medications.
"""

import json
import os
import re
from typing import Any

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def _redact_identity(text: str, full_name: str = "") -> str:
    """Remove obvious direct identifiers before sending text to the local LLM."""
    out = str(text or "").replace("\u00a0", " ")
    if full_name:
        out = re.sub(re.escape(full_name), "[ФИО удалено]", out, flags=re.I)

    # Remove labelled names even when deterministic FIO extraction did not succeed.
    out = re.sub(
        r"(?im)^\s*(?:фио(?:\s+пациента)?|ф\.?\s*и\.?\s*о\.?(?:\s+пациента)?|"
        r"фамилия\s*,?\s*имя\s*,?\s*отчество(?:\s+пациента)?|patient\s+name)\s*[:=—-]\s*[^\n]{2,160}",
        "ФИО: [удалено]",
        out,
    )
    # Common direct identifiers. This is a best-effort privacy reduction, not a legal anonymizer.
    out = re.sub(r"(?im)^\s*(?:телефон|phone)\s*[:=—-]\s*[^\n]+", "Телефон: [удалено]", out)
    out = re.sub(r"(?im)^\s*(?:e-?mail|электронная\s+почта)\s*[:=—-]\s*[^\n]+", "E-mail: [удалено]", out)
    out = re.sub(r"(?im)^\s*(?:снилс|полис|страхов(?:ой|ка)|patient\s*id|код\s+пациента)\s*[:=—-]\s*[^\n]+", r"\g<0> [удалено]", out)
    out = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}\b", "[e-mail удалён]", out)
    out = re.sub(r"(?<!\d)(?:\+7|8)[\s()\-]*\d{3}[\s()\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)", "[телефон удалён]", out)
    return out[:30000]


def _clean_field(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "; ".join(str(x) for x in value if str(x).strip())
    text = str(value).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" \t:;-—")
    if text.lower() in {"нет", "не указано", "не указан", "отсутствует", "none", "null", "n/a", "—", "-"}:
        return ""
    return text[:max_chars]


def parse_clinical_context_with_ollama(text: str, *, full_name: str = "", timeout: float = 90.0) -> dict[str, Any]:
    """Return structured clinical context from a document using a local Ollama model.

    The function returns ``ok=False`` instead of raising on connection/model errors so
    callers can safely fall back to the deterministic parser.
    """
    redacted = _redact_identity(text, full_name=full_name)
    system = (
        "Ты локальный модуль структурирования медицинского документа. "
        "Твоя задача — НЕ ставить диагноз и НЕ интерпретировать анализы, а только разнести уже написанный текст "
        "по трем полям. Ничего не придумывай и не дополняй знаниями извне. "
        "Верни строго JSON-объект с ключами complaints, history, medications. "
        "complaints: текущие жалобы и симптомы пациента. "
        "history: анамнез заболевания/жизни, хронические и сопутствующие заболевания, перенесенные события, операции, "
        "кровопотери, донорство и другие явно указанные клинические обстоятельства. "
        "medications: только явно названные лекарства/БАД, дозы, режим или длительность приема. "
        "Не помещай административные данные, даты печати, номера заказов, адреса, врачей, лабораторию, референсы и таблицу анализов. "
        "Если поле отсутствует — верни пустую строку. Сохраняй смысл исходного текста, не делай медицинских выводов."
    )
    user = "ДОКУМЕНТ (прямые идентификаторы по возможности удалены):\n" + redacted
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0},
            },
            timeout=(5, timeout),
        )
        r.raise_for_status()
        payload = r.json() if r.content else {}
        content = str((payload.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("пустой ответ локальной модели")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("модель вернула не JSON-объект")
        return {
            "ok": True,
            "complaints": _clean_field(parsed.get("complaints"), 2500),
            "history": _clean_field(parsed.get("history"), 5000),
            "medications": _clean_field(parsed.get("medications"), 3000),
            "model": OLLAMA_MODEL,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "complaints": "",
            "history": "",
            "medications": "",
            "model": OLLAMA_MODEL,
            "error": str(exc),
        }
