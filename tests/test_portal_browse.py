"""Tests for ticketing portal routing (canonical URLs + SERP bias)."""
from finkey_search.portal_browse import (
    augment_classifier_query_for_portals,
    resolve_cinema_portal_plan,
)


def test_ticketon_typo_and_city_triggers_plan():
    msg = "Открой тикитон и посмотри какие фильмы завтра в алматы"
    plan = resolve_cinema_portal_plan(msg)
    assert plan is not None
    assert plan.entry_url.startswith("https://ticketon.kz/almaty/cinema")
    assert "date_from=" in plan.entry_url
    assert plan.city_hint == "Алматы"
    assert plan.date_kind == "tomorrow"


def test_ticketon_without_listing_intent_skipped():
    msg = "расскажи что такое сервис ticketon"
    assert resolve_cinema_portal_plan(msg) is None


def test_classifier_query_gets_site_restrict():
    msg = "фильмы завтра в Алматы Тикитон"
    raw_q = "фильмы завтра в Алматы Тикитон"
    q = augment_classifier_query_for_portals(msg, raw_q)
    assert "site:ticketon.kz" in q.lower()
