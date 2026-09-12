"""Turn public Telegram web previews into cutoff-safe geospatial source signals.

Groq is used only to structure public reporting. Extracted posts are evidence
signals, never outcome labels. The text model is discovered at runtime, probed,
and cached; a failed cached model is automatically replaced by another active
text-capable model from the provider inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

GROQ_API = "https://api.groq.com/openai/v1"
CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{5,64}$")
NON_TEXT_MARKERS = ("whisper", "tts", "guard", "safeguard", "moderation", "embedding")
PREFERRED_TEXT_MODELS = (
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
)


@dataclass(frozen=True)
class PublicPost:
    source_id: str
    channel: str
    posted_at: str
    url: str
    text: str


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float


class _TelegramPreviewParser(HTMLParser):
    def __init__(self, channel: str) -> None:
        super().__init__(convert_charrefs=True)
        self.channel = channel
        self.posts: list[PublicPost] = []
        self._post_depth: int | None = None
        self._text_depth: int | None = None
        self._depth = 0
        self._source_id = ""
        self._posted_at = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        self._depth += 1
        if "tgme_widget_message" in classes and values.get("data-post"):
            self._post_depth = self._depth
            self._source_id = str(values["data-post"])
            self._posted_at = ""
            self._parts = []
        if self._post_depth is not None and "tgme_widget_message_text" in classes:
            self._text_depth = self._depth
        if self._post_depth is not None and tag == "time" and values.get("datetime"):
            self._posted_at = str(values["datetime"])
        if self._text_depth is not None and tag in {"br", "p"}:
            self._parts.append("\n")

    def handle_endtag(self, _tag: str) -> None:
        if self._text_depth == self._depth:
            self._text_depth = None
        if self._post_depth == self._depth:
            text = " ".join(html.unescape("".join(self._parts)).split())
            if text and self._posted_at:
                self.posts.append(
                    PublicPost(
                        source_id=f"telegram:{self._source_id}",
                        channel=self.channel,
                        posted_at=self._posted_at,
                        url=f"https://t.me/{self._source_id}",
                        text=text,
                    )
                )
            self._post_depth = None
            self._text_depth = None
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._text_depth is not None:
            self._parts.append(data)


def normalize_channel(value: str) -> str:
    channel = value.strip().removeprefix("@").rstrip("/").split("/")[-1]
    if not CHANNEL_RE.fullmatch(channel):
        raise ValueError(f"invalid public channel name: {value!r}")
    return channel


def fetch_public_preview(channel: str, *, timeout: int = 30) -> list[PublicPost]:
    clean = normalize_channel(channel)
    request = urllib.request.Request(
        f"https://t.me/s/{clean}",
        headers={"User-Agent": "HumanitarianForecast/1.0 public-research"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type != "text/html":
            raise RuntimeError(f"unexpected Telegram preview content type: {content_type}")
        page = response.read(2_000_000).decode("utf-8", errors="replace")
    parser = _TelegramPreviewParser(clean)
    parser.feed(page)
    if not parser.posts:
        raise RuntimeError(f"public preview for @{clean} returned no parseable posts")
    return parser.posts


def load_places(path: Path) -> list[Place]:
    document = json.loads(path.read_text(encoding="utf-8"))
    places: list[Place] = []
    for feature in document.get("features", []):
        name = str(feature.get("properties", {}).get("name") or "").strip()
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if name and len(coordinates) >= 2:
            places.append(Place(name, float(coordinates[1]), float(coordinates[0])))
    return places


def match_places(text: str, places: list[Place]) -> list[Place]:
    folded = text.casefold()
    return [
        place
        for place in places
        if re.search(rf"(?<!\w){re.escape(place.name.casefold())}(?!\w)", folded)
    ]


class GroqTextRouter:
    def __init__(self, api_key: str, cache_path: Path, *, timeout: int = 45) -> None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self.api_key = api_key
        self.cache_path = cache_path
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            GROQ_API + path,
            data=body,
            method="GET" if body is None else "POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "HumanitarianForecast/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Groq request failed with HTTP {exc.code}") from exc

    def candidates(self) -> list[str]:
        inventory = self._request("/models").get("data", [])
        available = [
            str(item.get("id"))
            for item in inventory
            if item.get("active", True)
            and item.get("id")
            and not any(marker in str(item["id"]).casefold() for marker in NON_TEXT_MARKERS)
        ]
        preferred = [model for model in PREFERRED_TEXT_MODELS if model in available]
        remaining = sorted(model for model in available if model not in preferred and "/compound" not in model)
        return preferred + remaining

    def _cached(self) -> str | None:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return str(value["model"])
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _probe(self, model: str) -> bool:
        try:
            result = self._request(
                "/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with exactly HF_OK"}],
                    "temperature": 0,
                    "max_completion_tokens": 16,
                },
            )
            content = str(result["choices"][0]["message"]["content"])
            return "HF_OK" in content
        except (KeyError, IndexError, RuntimeError, TypeError):
            return False

    def select(self, *, exclude: set[str] | None = None) -> str:
        excluded = exclude or set()
        candidates = self.candidates()
        cached = self._cached()
        ordered = ([cached] if cached and cached in candidates else []) + candidates
        seen: set[str] = set()
        for model in ordered:
            if model in seen or model in excluded:
                continue
            seen.add(model)
            if self._probe(model):
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps({"model": model, "checkedAt": datetime.now(UTC).isoformat()}),
                    encoding="utf-8",
                )
                return model
        raise RuntimeError("no active Groq text model passed the capability probe")

    def extract(self, posts: list[PublicPost], places: list[Place]) -> tuple[str, list[dict[str, Any]]]:
        model = self.select()
        allowed_places = sorted({place.name for post in posts for place in match_places(post.text, places)})
        payload = [asdict(post) for post in posts]
        prompt = (
            "Convert these public posts into conservative humanitarian source signals. "
            "Return one JSON object with a signals array. Each signal must contain sourceId, "
            "sourceUrl, postedAt, summary, category (conflict|displacement|access|health|food|other), "
            "placeName, and confidence from 0 to 1. Use only placeName values from allowedPlaces. "
            "Do not invent events, coordinates, casualties, or dates. Omit irrelevant posts. "
            "A post is reporting evidence, never verified outcome truth.\n"
            + json.dumps({"allowedPlaces": allowed_places, "posts": payload}, ensure_ascii=False)
        )
        failed: set[str] = set()
        for _attempt in range(3):
            try:
                result = self._request(
                    "/chat/completions",
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You produce strict JSON for an auditable humanitarian data pipeline."},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                        "max_completion_tokens": 4096,
                    },
                )
                content = result["choices"][0]["message"]["content"]
                signals = json.loads(content).get("signals", [])
                if not isinstance(signals, list):
                    raise TypeError("signals must be a list")
                return model, self._validate_signals(signals, posts, places)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
                failed.add(model)
                model = self.select(exclude=failed)
        raise RuntimeError("Groq extraction failed across three probed text models")

    @staticmethod
    def _validate_signals(
        signals: list[dict[str, Any]], posts: list[PublicPost], places: list[Place]
    ) -> list[dict[str, Any]]:
        posts_by_id = {post.source_id: post for post in posts}
        places_by_name = {place.name: place for place in places}
        output: list[dict[str, Any]] = []
        for item in signals:
            source_id = str(item.get("sourceId") or "")
            place_name = str(item.get("placeName") or "")
            post = posts_by_id.get(source_id)
            place = places_by_name.get(place_name)
            if post is None or place is None:
                continue
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0))))
            output.append(
                {
                    "schemaVersion": "source-signal.v1",
                    "sourceId": source_id,
                    "sourceType": "telegram-public-web-preview",
                    "sourceUrl": post.url,
                    "postedAt": post.posted_at,
                    "observedAt": datetime.now(UTC).isoformat(),
                    "category": str(item.get("category") or "other"),
                    "summary": str(item.get("summary") or "")[:240],
                    "placeName": place.name,
                    "latitude": place.latitude,
                    "longitude": place.longitude,
                    "confidence": confidence,
                    "rawTextSha256": hashlib.sha256(post.text.encode("utf-8")).hexdigest(),
                    "trainingRole": "input-signal-not-outcome-label",
                }
            )
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default=os.getenv("HF_PUBLIC_PREVIEW_CHANNELS", ""))
    parser.add_argument("--cities", type=Path, default=Path("HF/public/ethiopia-cities.geojson"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/hf/groq-text-model.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/public-preview-signals.jsonl"))
    parser.add_argument("--max-posts", type=int, default=40)
    args = parser.parse_args()
    channels = [normalize_channel(value) for value in args.channels.split(",") if value.strip()]
    if not channels:
        parser.error("provide at least one public Telegram channel")
    posts: list[PublicPost] = []
    for channel in channels:
        posts.extend(fetch_public_preview(channel))
    newest = sorted({post.source_id: post for post in posts}.values(), key=lambda post: post.posted_at)[-args.max_posts :]
    places = load_places(args.cities)
    relevant = [post for post in newest if match_places(post.text, places)]
    if not relevant:
        raise RuntimeError("no preview posts mentioned a configured Ethiopian city")
    router = GroqTextRouter(os.getenv("GROQ_API_KEY", ""), args.cache)
    model, signals = router.extract(relevant, places)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(signal, ensure_ascii=False, separators=(",", ":")) + "\n" for signal in signals),
        encoding="utf-8",
    )
    print(json.dumps({"model": model, "posts": len(newest), "relevant": len(relevant), "signals": len(signals)}))


if __name__ == "__main__":
    main()
