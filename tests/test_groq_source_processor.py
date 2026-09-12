from __future__ import annotations

import json
from pathlib import Path

from humanitarian_forecast.data.groq_source_processor import (
    GroqTextRouter,
    Place,
    _TelegramPreviewParser,
    match_places,
    normalize_channel,
)


def test_public_preview_parser_preserves_source_and_publication_time() -> None:
    parser = _TelegramPreviewParser("example_channel")
    parser.feed(
        '<div class="tgme_widget_message" data-post="example_channel/42">'
        '<div class="tgme_widget_message_text">Update near <b>Bahir Dar</b>.</div>'
        '<time datetime="2026-09-12T06:00:00+00:00"></time></div>'
    )

    assert len(parser.posts) == 1
    assert parser.posts[0].source_id == "telegram:example_channel/42"
    assert parser.posts[0].url == "https://t.me/example_channel/42"
    assert parser.posts[0].posted_at == "2026-09-12T06:00:00+00:00"
    assert parser.posts[0].text == "Update near Bahir Dar."


def test_channel_validation_and_exact_place_matching() -> None:
    assert normalize_channel("https://t.me/example_channel/") == "example_channel"
    places = [Place("Adama", 8.54, 39.27), Place("Bahir Dar", 11.59, 37.39)]
    assert [place.name for place in match_places("Access near Bahir Dar", places)] == ["Bahir Dar"]
    assert match_places("An unrelated word: Adamant", places) == []


def test_router_prefers_cached_working_text_model_and_rotates(tmp_path: Path) -> None:
    cache = tmp_path / "model.json"
    cache.write_text(json.dumps({"model": "cached-text"}), encoding="utf-8")
    router = GroqTextRouter("test-key", cache)
    router.candidates = lambda: ["cached-text", "fallback-text"]  # type: ignore[method-assign]
    router._probe = lambda model: model == "fallback-text"  # type: ignore[method-assign]

    assert router.select() == "fallback-text"
    assert json.loads(cache.read_text(encoding="utf-8"))["model"] == "fallback-text"
