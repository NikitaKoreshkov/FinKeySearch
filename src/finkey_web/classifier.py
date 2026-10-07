# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
SearchNeedClassifier — decides whether live web retrieval is required.

Default mode ``llm_primary`` (``FINKEY_SEARCH_CLASSIFIER_MODE``):
  • Ultra-fast trivial skips only (hello/thanks/bare arithmetic/meta-chat)
  • **LLM semantic judgment** for everything else — no regex “must search” lists
  • JSON includes optional ``category`` for date_restrict / synthesis hints

Legacy mode ``legacy`` restores regex Tier‑2 YES + time-trigger shortcuts + old LLM tier‑3.

Environment:
  FINKEY_SEARCH_CLASSIFIER_MODE = llm_only | llm_primary | legacy
    - llm_only: no regex “trivial skip”; routing is 100% classifier LLM (+ optional legacy fallback).
    - llm_primary (default): trivial skip heuristics then LLM (saves a call on hi/thanks).
    - legacy: regex-heavy pipeline.
  FINKEY_SEARCH_NO_LLM_FALLBACK = legacy_heuristic       (optional bootstrap without LLM)
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from .schema import SearchCategory, SearchDecision, SearchDepth, SearchTier
from .trivial_skip_lexicon import (
    TRIVIAL_NO_MAX_TOKEN_WORDS,
    TRIVIAL_NO_PHRASES,
    TRIVIAL_NO_TOKENS,
    is_trivial_chat_message,
    normalize_trivial_text,
)



_NO_PATTERNS = [
    r"^(привет|хай|хеллоу|hello|hi|hey|добрый (день|вечер|утро))[!.,\s]*$",
    r"^(как дела|how are you|как ты)[?!.,\s]*$",
    r"^(спасибо|thanks|thank you|ок|окей|ok|понял|понял[аи]?|хорошо|ладно)[!.,\s]*$",
    r"^(да|нет|yes|no|ага|угу|не понял|что\??)[?!.,\s]*$",
    r"^\d[\d\s\+\-\*\/\.\,\%\(\)]+[=\?]?\s*$",
    r"(посчитай|рассчитай|вычисли|calculate|compute)\s+[\d]",
    r"^(что такое|объясни|расскажи про|как работает|что значит|what is|explain|how does)\s+\w",
    r"(биография|кто такой|definition of)\s+\w",
    r"(проанализируй|разбор|анализ)\s+(мо[йея]|эт[иоа]|эту|этот)",
    r"(ты (говорил|сказал)|раньше (ты|ты говорил)|в начале разговора)",
    r"^(помоги мне|посоветуй|как ты думаешь|что думаешь|как мне)\s",
    r"^(я хочу|я думаю|я чувствую|мне кажется)\s",
]

_NO_KEYWORDS = [
    "привет", "пока", "спасибо", "пожалуйста",
    "hello", "bye", "thanks", "please",
]


_TRIVIAL_NO_PATTERNS = [
    r"^(привет|хай|хеллоу|hello|hi|hey|добрый (день|вечер|утро))[!.,\s]*$",
    r"^(как дела|how are you|как ты)[?!.,\s]*$",
    r"^(спасибо|thanks|thank you|ок|окей|ok|понял|понял[аи]?|хорошо|ладно)[!.,\s]*$",
    r"^(да|нет|yes|no|ага|угу|не понял|что\??)[?!.,\s]*$",
    r"^\d[\d\s\+\-\*\/\.\,\%\(\)]+[=\?]?\s*$",
    r"(посчитай|рассчитай|вычисли|calculate|compute)\s+[\d]",
    r"(ты (говорил|сказал)|раньше (ты|ты говорил)|в начале разговора)",
]



_YES_RULES: list[tuple[list[str], SearchCategory]] = [
    (
        [
            r"курс (доллара|евро|рубля|тенге|юаня|фунта|\w+) (сейчас|сегодня|на сегодня|на данный момент)",
            r"(сколько стоит|цена) (доллар|евро|рубл|тенге|юань)",
            r"(usd|eur|gbp|jpy|cny|kzt|rub).*(курс|цена|стоит)",
            r"exchange rate.*(today|now|current)",
            r"(dollar|euro|pound).*(rate|price|worth) (now|today)",
        ],
        SearchCategory.CURRENCY_RATE,
    ),
    (
        [
            r"(биткоин|bitcoin|btc|ethereum|eth|крипт\w+).*(цена|стоит|курс|rate|price)",
            r"(цена|курс|стоит).*(биткоин|bitcoin|btc|eth|ethereum|crypto)",
            r"crypto (price|rate|market)",
            r"(сколько стоит|price of).*(btc|eth|bnb|solana|sol|usdt|usdc)",
        ],
        SearchCategory.CRYPTO_PRICE,
    ),
    (
        [
            r"(акции|акция|shares?|stocks?).*(цена|стоит|курс|price|value|worth)",
            r"(цена акций|стоимость акций|котировки)",
            r"(apple|tesla|google|microsoft|amazon|nvidia|sber|gazprom|лукойл).*(акции|stock|share|price)",
            r"(nasdaq|s&p|sp500|dow jones|ммвб|moex).*(сейчас|сегодня|today|now|current)",
        ],
        SearchCategory.STOCK_PRICE,
    ),
    (
        [
            r"(ставка цб|ключевая ставка|ставка рефинансирования).*(сейчас|сегодня|текущ)",
            r"(цб|центробанк|фрс|fed|ecb).*(ставк|rate|решени)",
            r"(current|today).*(interest rate|base rate|key rate)",
            r"(ипотека|mortgage|вклад|deposit).*(процент|ставка|rate).*(сейчас|сегодня|лучш|топ)",
        ],
        SearchCategory.INTEREST_RATE,
    ),
    (
        [
            r"(что|почему|зачем).*(упал|вырос|рухнул|обвалился|взлетел|скачок|сдулся)",
            r"(последние|свежие|актуальные) (новости|события)",
            r"(что сейчас|что происходит) (с|в|на)\s+\w",
            r"(news|latest|recent)\s+(about|on|for|in)\s+\w",
            r"(новости|события) (сегодня|сейчас|на этой неделе|за последнее время)",
        ],
        SearchCategory.FINANCIAL_NEWS,
    ),
    (
        [
            r"(инфляция|inflation).*(сейчас|сегодня|официальная|текущ|процент|уровень)",
            r"(ввп|gdp).*(текущ|current|today|рост|latest)",
            r"(безработица|unemployment).*(уровень|сейчас|данные)",
            r"(экономические показатели|economic indicators).*(текущ|last|latest)",
        ],
        SearchCategory.ECONOMIC_DATA,
    ),
    (
        [
            r"\w+\s+(новости|отчёт|прибыль|релиз|обновление|версия|вышла?|вышел|запустил)",
            r"(компания|проект|продукт|стартап|бренд).*(последн|recent|latest).*(новост|news|отчёт|report)",
            r"(версия|release|update|вышл\w+).*(сейчас|сегодня|последн|latest|current)",
        ],
        SearchCategory.COMPANY_INFO,
    ),
    (
        [
            r"(новый|изменения в|поправки к).*(закон|регулирование|регулятор|правил)",
            r"(закон|law|regulation|rule).*(изменения|changes|новый|new|2025|2026)",
            r"(вступил в силу|вступает|новые правила|новые требования).*(2025|2026|\w+)",
        ],
        SearchCategory.REGULATORY,
    ),
    (
        [
            r"(gpt|claude|gemini|llama|mistral|openai|anthropic|google).*(новый|latest|версия|release|вышел|update)",
            r"(iphone|android|windows|macos|linux).*(вышел|релиз|обновление|версия|latest)",
            r"(новая версия|latest version|current version|вышел) \w+",
            r"(искусственный интеллект|ai|ии).*(новости|достижения|прорыв|что нового|latest)",
        ],
        SearchCategory.GENERAL,
    ),
    (
        [
            r"(счёт|результат|итог).*(матч|игра|турнир|чемпионат).*(сейчас|сегодня|вчера)",
            r"(кто выиграл|кто победил|выиграл).*(матч|чемпионат|турнир)",
            r"(score|result|winner).*(match|game|tournament).*(today|yesterday|now|current)",
        ],
        SearchCategory.GENERAL,
    ),
]

_TIME_TRIGGERS = [
    r"\bсейчас\b", r"\bна данный момент\b", r"\bсегодня\b", r"\bна сегодня\b",
    r"\bна этой неделе\b", r"\bв этом месяце\b", r"\bв этом году\b",
    r"\bнедавно\b", r"\bпоследн\w+\b", r"\bактуальн\w+\b", r"\bтекущ\w+\b",
    r"\bnow\b", r"\btoday\b", r"\bcurrent(ly)?\b", r"\brecent(ly)?\b",
    r"\blatest\b", r"\bthis (week|month|year)\b", r"\bin 2026\b",
]

_TIME_SENSITIVE_TERMS = [
    "курс", "акци", "крипт", "биткоин", "ставка", "вклад",
    "депозит", "ипотека", "кредит", "рынок", "индекс", "валют",
    "инфляц", "ввп", "экономик", "налог", "процент",
    "rate", "stock", "crypto", "bitcoin", "deposit", "mortgage", "market",
    "inflation", "gdp", "economy", "tax",
    "версия", "релиз", "обновление", "вышел", "вышла",
    "release", "version", "update", "launched",
    "новости", "события", "цена", "стоимость",
    "news", "events", "price", "score", "result",
]


def _normalize_trivial(text: str) -> str:
    return normalize_trivial_text(text)


def _matches_any(text: str, patterns: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def _is_trivial_lexicon_match(text: str) -> bool:
    return is_trivial_chat_message(text)


def _has_time_trigger(text: str) -> bool:
    return _matches_any(text, _TIME_TRIGGERS)


def _has_time_sensitive_term(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in _TIME_SENSITIVE_TERMS)


# ``{year}`` is filled with the current year at query time: a hardcoded year turns
# into a wrong-year search query the moment the calendar rolls over.
_QUERY_TEMPLATES: dict[SearchCategory, str] = {
    SearchCategory.CURRENCY_RATE:   "{raw} курс сегодня",
    SearchCategory.CRYPTO_PRICE:    "{raw} цена сейчас",
    SearchCategory.STOCK_PRICE:     "{raw} котировки",
    SearchCategory.INTEREST_RATE:   "{raw} {year}",
    SearchCategory.FINANCIAL_NEWS:  "{raw}",
    SearchCategory.ECONOMIC_DATA:   "{raw} данные {year}",
    SearchCategory.COMPANY_INFO:    "{raw} последние новости {year}",
    SearchCategory.REGULATORY:      "{raw} изменения {year}",
    SearchCategory.PRODUCT_RATES:   "{raw} {year}",
    SearchCategory.GENERAL_FINANCE: "{raw}",
    SearchCategory.GENERAL:         "{raw}",
}


def _optimize_query(message: str, category: SearchCategory) -> str:
    noise = [
        r"(скажи|расскажи|напиши|покажи|найди|подскажи|помоги)\s+(мне\s+)?",
        r"(хочу знать|хочу узнать|интересует|интересно)\s+",
        r"(можешь|может ли ты|мог бы ты)\s+",
        r"(пожалуйста|плз|please)\s*",
    ]
    query = message.strip()
    for pattern in noise:
        query = re.sub(pattern, "", query, flags=re.IGNORECASE)
    query = query.strip("?.!,;:").strip()

    words = query.split()
    if len(words) > 10:
        query = " ".join(words[:10])

    from finkey_web.timeutil import current_datetime

    template = _QUERY_TEMPLATES.get(category, "{raw}")
    return template.format(raw=query, year=current_datetime().year).strip()


def _effective_user_message(
    latest: str,
    conversation_messages: Optional[list[str]],
) -> str:
    if not conversation_messages:
        return latest
    tail = [m.strip() for m in conversation_messages[-6:] if m.strip()]
    if not tail:
        return latest
    return "\n".join(tail) + "\n---\nТекущее сообщение:\n" + latest.strip()


def _classifier_mode() -> str:
    return os.getenv("FINKEY_SEARCH_CLASSIFIER_MODE", "llm_primary").strip().lower()


def _first_balanced_json_object(text: str) -> Optional[str]:
    """
    Из суффикса текста вытащить первый сбалансированный JSON-объект {...}.
    Учитываются строки в двойных кавычках (стандарт JSON).
    ``text`` должен начинаться с «{» либо функция вернёт None.
    """
    if not text or text[0] != "{":
        start = text.find("{")
        if start < 0:
            return None
        text = text[start:]
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return None


def _parse_classifier_json_dict(raw: str) -> Optional[dict]:
    """
    Модель часто вставляет в ответ подсказку-шаблон с первой «{».
    Берём **последний** успешно распарсенный JSON-объект в тексте.
    """
    raw = raw.strip()
    brace_idx = [m.start() for m in re.finditer(r"\{", raw)]
    for start in reversed(brace_idx):
        blob = _first_balanced_json_object(raw[start:])
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "search" in data:
            return data
    return None


def _truthy_llm(raw: object) -> bool:
    """Parse boolean-ish fields from classifier JSON."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False



def _today_label() -> str:
    """Current date for the classifier prompt — it cannot judge freshness blind."""
    from finkey_web.timeutil import current_datetime

    now = current_datetime()
    return now.strftime("%Y-%m-%d (%A, %B %Y)")


_LEGACY_LLM_PROMPT = """You are a search-need classifier. Reply ONLY with JSON.

Today's real date: {today}

User message: "{message}"

Does answering this question require current/fresh information from the internet?

SEARCH when: live prices, today's news, recent events, latest releases, current statistics, sports results, regulatory changes — or any question whose answer could have changed after your knowledge cutoff, including events already dated in the past that you have no record of.
DON'T SEARCH when: conceptual questions, definitions, math, personal advice, small talk, established history well before your cutoff.

JSON answer:
{{"search": true/false, "query": "clean search query if search=true, else empty", "reason": "one line why"}}"""


_SEMANTIC_LLM_PROMPT = """You route web search and browser automation. Follow OUTPUT RULES first.

OUTPUT RULES (mandatory):
• Emit ONE JSON object only. Total output = that object, nothing else.
• The FIRST character you write MUST be `{{` — no preamble, no headings, no reasoning, no markdown fences.
• All keys required: "search" (boolean), "query" (string), "reason" (short string), "category" (string or null), "verified_sources_required" (boolean), "depth" (string "fast" or "deep"), "url" (string — only for browser_action, else "").

Schema:
{{"search": true, "query": "...", "reason": "...", "category": "general", "verified_sources_required": false, "depth": "fast", "url": ""}}

Today's real date is {today}. Your own knowledge ends well before it, so anything dated between then and today is a blind spot: it already happened, you simply have no record of it. Treat "I have never heard of this outcome" as a reason to search, never as evidence that the event is still upcoming.

When to set search=true:
1. FRESH FACTS needed: the true answer could have changed, or come into existence, after the knowledge cutoff — markets, rates, breaking news, regulations, weather, sports results and tournament winners, product releases, named institutions, current schedules/prices. Judge what the question is about, not which words it uses: it needs no "today" or "latest" to be time-sensitive, and a past-dated event you cannot recall still requires a search.
2. BROWSER ACTION requested: user wants to navigate, interact with, or perform actions on a website — keywords like: зайди на, открой сайт, создай аккаунт, зарегистрируйся, войди в, напиши на сайте, купи, заполни форму, нажми кнопку, перейди по ссылке, go to, open, register, login, sign up, create account, send message to website, click, fill in — use category="browser_action", query=the target website name/domain, and url=the exact starting URL.

DO NOT SEARCH only when: timeless concepts, pure math, text rewriting, therapy/meta-chat without factual claims, questions about the AI itself.

If search=true, category must be one of:
currency_rate, crypto_price, stock_price, interest_rate, financial_news, economic_data, company_info, regulatory, product_rates, general_finance, general, browser_action
If category=browser_action: also set url=the best known direct URL for the target service/page (e.g. "chatgpt" → "https://chatgpt.com", "google ai mode" → "https://www.google.com/search?udm=50", "gmail" → "https://mail.google.com"). If unknown, set url="".
If search=false: category=null, query="", url="".

verified_sources_required=true ONLY when user explicitly demands citations/URLs/sources.

"depth" — how much retrieval work the answer needs (judge the substance of the request, any language):
• "fast": a single current fact or quick lookup — one price/rate/quote/score, one headline, "what is X right now", a short factual answer. Search-result snippets alone normally answer it. Optimised for speed.
• "deep": needs reading full articles, comparing several sources, gathering many details, building an analysis/report/summary, or step-by-step research across pages.
Rule of thumb: if a one-to-three sentence factual reply settles it → "fast". If answering well means synthesising across multiple sources or producing a longer structured answer → "deep". When verified_sources_required=true or category=browser_action → use "deep". If search=false → "fast".

User / dialogue context:
"{context}"

Now output JSON starting with `{{`."""


def _depth_from_llm(raw: object) -> SearchDepth:
    """
    Parse the AI's ``depth`` decision. Conservative: only an explicit ``fast`` enables
    the fast path; anything else (incl. missing/invalid) stays DEEP to protect quality.
    """
    if isinstance(raw, str) and raw.strip().lower() == "fast":
        return SearchDepth.FAST
    return SearchDepth.DEEP


def _category_from_llm(raw: object) -> Optional[SearchCategory]:
    if raw is None or raw is False:
        return None
    if not isinstance(raw, str):
        return SearchCategory.GENERAL
    v = raw.strip().lower()
    if not v or v == "null":
        return None
    try:
        return SearchCategory(v)
    except ValueError:
        return SearchCategory.GENERAL


def _llm_classify_legacy(
    message: str,
    generate_fn: Callable[[str], str],
    conversation_messages: Optional[list[str]],
) -> SearchDecision:
    try:
        block = message.strip()[:480]
        if conversation_messages:
            ctx = "\n".join(m.strip() for m in conversation_messages[-8:] if m.strip())
            if ctx:
                block = ctx[:1400] + "\n---\nПоследнее сообщение пользователя:\n" + message.strip()[:480]
        safe = block.replace('"', "'")
        prompt = _LEGACY_LLM_PROMPT.format(message=safe[:1800], today=_today_label())
        raw = generate_fn(prompt).strip()

        data = _parse_classifier_json_dict(raw)
        if not data:
            return SearchDecision(
                should_search=False,
                tier=SearchTier.LLM_CLASSIFIED,
                reason="LLM returned no JSON",
            )

        should_search = bool(data.get("search", False))
        query = str(data.get("query", ""))
        reason = str(data.get("reason", ""))

        return SearchDecision(
            should_search=should_search,
            tier=SearchTier.LLM_CLASSIFIED,
            category=SearchCategory.GENERAL if should_search else None,
            search_query=query,
            original_intent=reason,
            confidence=0.85,
            reason=reason,
        )
    except Exception as exc:
        return SearchDecision(
            should_search=False,
            tier=SearchTier.LLM_CLASSIFIED,
            reason=f"LLM classifier error: {exc}",
        )


def _llm_classify_semantic(
    effective_message: str,
    generate_fn: Callable[[str], str],
) -> SearchDecision:
    try:
        safe = effective_message.replace('"', "'")[:4200]
        prompt = _SEMANTIC_LLM_PROMPT.format(context=safe, today=_today_label())
        raw = generate_fn(prompt).strip()

        data = _parse_classifier_json_dict(raw)
        if not data:
            return SearchDecision(
                should_search=False,
                tier=SearchTier.LLM_CLASSIFIED,
                reason="semantic LLM: no JSON",
            )

        should_search = bool(data.get("search", False))
        query = str(data.get("query", "")).strip()
        reason = str(data.get("reason", ""))
        category = _category_from_llm(data.get("category"))
        verified = _truthy_llm(data.get("verified_sources_required"))
        depth = _depth_from_llm(data.get("depth"))
        browser_url = str(data.get("url", "")).strip()
        if browser_url and not browser_url.startswith("http"):
            browser_url = ""

        if should_search:
            cat = category or SearchCategory.GENERAL
            if not query:
                query = _optimize_query(effective_message, cat)
        else:
            cat = None
            depth = SearchDepth.FAST

        return SearchDecision(
            should_search=should_search,
            tier=SearchTier.LLM_CLASSIFIED,
            category=cat,
            search_query=query if should_search else "",
            original_intent=reason,
            confidence=0.88,
            reason=reason or "semantic_classifier",
            verified_sources_required=verified,
            browser_action_url=browser_url,
            search_depth=depth,
        )
    except Exception as exc:
        return SearchDecision(
            should_search=False,
            tier=SearchTier.LLM_CLASSIFIED,
            reason=f"semantic LLM error: {exc}",
        )




class SearchNeedClassifier:
    """
    ``llm_only``: **semantic LLM only** — no regex trivial-skip gate.
    ``llm_primary`` (default): optional trivial NO heuristics + semantic LLM.
    ``legacy``: original 3‑tier regex‑heavy pipeline.
    """

    def classify(
        self,
        message: str,
        turn_number: int = 0,
        allow_llm_tier: bool = True,
        generate_fn: Optional[Callable[[str], str]] = None,
        conversation_messages: Optional[list[str]] = None,
    ) -> SearchDecision:
        del turn_number
        text = message.strip()
        effective = _effective_user_message(text, conversation_messages)

        mode = _classifier_mode()

        if mode == "legacy":
            return self._classify_legacy(
                text,
                effective,
                allow_llm_tier,
                generate_fn,
                conversation_messages,
            )

        if mode == "llm_only":
            if allow_llm_tier and generate_fn is not None:
                return _llm_classify_semantic(effective, generate_fn)
            fb = os.getenv("FINKEY_SEARCH_NO_LLM_FALLBACK", "").strip().lower()
            if fb == "legacy_heuristic":
                return self._classify_legacy(
                    text,
                    effective,
                    False,
                    None,
                    conversation_messages,
                )
            return SearchDecision(
                should_search=False,
                tier=SearchTier.HEURISTIC_NO,
                reason="llm_only_requires_generate_fn",
            )

        if self._is_trivial_no(text):
            return SearchDecision(
                should_search=False,
                tier=SearchTier.HEURISTIC_NO,
                reason="trivial_skip_llm_primary",
            )

        if allow_llm_tier and generate_fn is not None:
            result = _llm_classify_semantic(effective, generate_fn)
            if "no JSON" in result.reason or result.reason.startswith("semantic LLM error"):
                return self._classify_legacy(
                    text,
                    effective,
                    False,
                    None,
                    conversation_messages,
                )
            return result

        fb = os.getenv("FINKEY_SEARCH_NO_LLM_FALLBACK", "").strip().lower()
        if fb == "legacy_heuristic":
            return self._classify_legacy(
                text,
                effective,
                False,
                None,
                conversation_messages,
            )

        return SearchDecision(
            should_search=False,
            tier=SearchTier.HEURISTIC_NO,
            reason="llm_primary_requires_generate_fn",
        )

    def _classify_legacy(
        self,
        text: str,
        effective: str,
        allow_llm_tier: bool,
        generate_fn: Optional[Callable[[str], str]],
        conversation_messages: Optional[list[str]],
    ) -> SearchDecision:
        if self._is_definitely_no(text):
            return SearchDecision(
                should_search=False,
                tier=SearchTier.HEURISTIC_NO,
                reason="Tier 1: heuristic skip",
            )

        yes_result = self._is_definitely_yes(effective)
        if yes_result is not None:
            category, _ = yes_result
            query = _optimize_query(effective, category)
            return SearchDecision(
                should_search=True,
                tier=SearchTier.HEURISTIC_YES,
                category=category,
                search_query=query,
                original_intent=f"Нужны актуальные данные: {category.value}",
                confidence=0.95,
                reason=f"Tier 2: matched {category.value}",
            )

        if _has_time_trigger(effective) and _has_time_sensitive_term(effective):
            query = _optimize_query(effective, SearchCategory.GENERAL)
            return SearchDecision(
                should_search=True,
                tier=SearchTier.HEURISTIC_YES,
                category=SearchCategory.GENERAL,
                search_query=query,
                original_intent="Time trigger + time-sensitive term detected",
                confidence=0.80,
                reason="Tier 2: time trigger + domain term",
            )

        if allow_llm_tier and generate_fn is not None:
            return _llm_classify_legacy(text, generate_fn, conversation_messages)

        return SearchDecision(
            should_search=False,
            tier=SearchTier.HEURISTIC_NO,
            reason="No signal found, default no-search",
        )

    def _is_trivial_no(self, text: str) -> bool:
        if _is_trivial_lexicon_match(text):
            return True
        if len(text.split()) <= 2:
            lower = text.lower()
            if any(kw in lower for kw in _NO_KEYWORDS):
                return True
        return _matches_any(text, _TRIVIAL_NO_PATTERNS)

    def _is_definitely_no(self, text: str) -> bool:
        if len(text.split()) <= 2:
            lower = text.lower()
            if any(kw in lower for kw in _NO_KEYWORDS):
                return True
        return _matches_any(text, _NO_PATTERNS)

    def _is_definitely_yes(
        self, text: str
    ) -> Optional[tuple[SearchCategory, str]]:
        for patterns, category in _YES_RULES:
            if _matches_any(text, patterns):
                return category, text
        return None
