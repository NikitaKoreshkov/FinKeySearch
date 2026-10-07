from finkey_search.citation_verify import answer_references_allowed_web_sources


def test_full_url_match():
    assert answer_references_allowed_web_sources(
        "Источник: https://example.com/a/b",
        ["https://example.com/a/b"],
    )


def test_host_substring_match():
    assert answer_references_allowed_web_sources(
        "См. сайт example.com/news",
        ["https://www.example.com/page"],
    )


def test_host_from_inline_url():
    assert answer_references_allowed_web_sources(
        "Ссылка https://bank.ru/rates здесь.",
        ["https://bank.ru/stats?q=1"],
    )


def test_no_match_returns_false():
    assert not answer_references_allowed_web_sources(
        "Курс по данным биржи, без доменных имён.",
        ["https://solely-other.org/x"],
    )


def test_empty_citations_false():
    assert not answer_references_allowed_web_sources("https://evil.com/x", [])
