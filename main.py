import os
import time
import json
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

JST = timezone(timedelta(hours=9))

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# Sources
JHA_FEED_URL = "https://en.hockey.or.jp/feed/"
FIH_NEWS_URL = "https://www.fih.hockey/news"

# Notion に「Original Title」列があるなら True
# ないなら False のままでOK
USE_ORIGINAL_TITLE_COLUMN = False


# ---------------------------
# Helpers
# ---------------------------
def norm_url(url: str) -> str:
    url = (url or "").strip()
    return url[:-1] if url.endswith("/") else url


def fetch_html(url: str) -> BeautifulSoup:
    """Fetch HTML and return BeautifulSoup. Raise on HTTP errors."""
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def take_text(soup: BeautifulSoup, max_chars: int = 5000) -> str:
    ps = []
    total = 0
    for p in soup.select("p"):
        t = p.get_text(" ", strip=True)
        if len(t) >= 40:
            ps.append(t)
            total += len(t)
        if total >= max_chars:
            break
    joined = "\n".join(ps)
    return joined[:max_chars].strip()


def extract_json_object(text: str) -> dict:
    """
    Gemini が ```json ... ``` 付きで返したり、前後に余計な文字を付けても
    できるだけ JSON を取り出す。
    """
    if not text:
        raise ValueError("empty response")

    s = text.strip()

    # ```json ... ``` 対応
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            s = "\n".join(lines[1:-1]).strip()

    # 先頭の { から末尾の } までを抜く
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"JSON object not found: {text}")

    s = s[start:end + 1]
    return json.loads(s)


# ---------------------------
# Scrape sources
# ---------------------------
def scrape_jha(limit: int = 8):
    """JHA: use RSS feed to avoid 403 from GitHub Actions runners."""
    feed = feedparser.parse(JHA_FEED_URL)
    items = []
    for e in feed.entries[: limit * 3]:
        title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "region": "Japan",
                "source_name": "JHA",
                "title": title,
                "url": norm_url(link),
            }
        )

    # dedupe by URL
    seen = set()
    out = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        out.append(it)
        if len(out) >= limit:
            break
    return out


def scrape_fih(limit: int = 8):
    soup = fetch_html(FIH_NEWS_URL)
    items = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        title = a.get_text(" ", strip=True)

        if not href or not title or len(title) < 10:
            continue

        if href.startswith("/"):
            href = "https://www.fih.hockey" + href

        # Keep only FIH news article pages
        if href.startswith("https://www.fih.hockey/") and "/news/" in href:
            items.append(
                {
                    "region": "Overseas",
                    "source_name": "FIH",
                    "title": title,
                    "url": norm_url(href),
                }
            )

    # dedupe by URL
    seen = set()
    out = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        out.append(it)
        if len(out) >= limit:
            break
    return out


# ---------------------------
# Gemini summary / title
# ---------------------------
def post_with_backoff(url, headers, json_payload, timeout=60, max_retries=6):
    wait = 2
    last_response = None
    for _ in range(max_retries):
        r = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        last_response = r
        if r.status_code != 429:
            return r
        time.sleep(wait)
        wait = min(wait * 2, 60)
    return last_response


def gemini_generate_japanese_fields(
    title: str, region: str, source_name: str, url: str, body: str
) -> dict:
    prompt = f"""
あなたはフィールドホッケーのニュース要約アシスタントです。
次の情報から、日本語タイトルと日本語要約を作ってください。

出力は必ず JSON のみ。
説明文やコードブロックは不要です。

{{
  "jp_title": "...",
  "jp_summary": "..."
}}

要件:
- jp_title: 40文字以内、日本語、自然なニュース見出し
- 固有名詞は必要なら英語のままで可
- jp_summary: 2〜4文、100〜180字程度、日本語
- 「何が起きたか」「なぜ注目か」を含める
- 本文が短い/不十分なら推測しすぎない
- 誇張しない
- 原文の意味から大きく外れない

地域: {region}
ソース: {source_name}
タイトル(原文): {title}
URL: {url}

本文:
{body}
""".strip()

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    r = post_with_backoff(
        f"{endpoint}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json_payload=payload,
        timeout=60,
        max_retries=8,
    )
    r.raise_for_status()

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Gemini response parse error: {e}, raw={data}")

    parsed = extract_json_object(text)

    jp_title = (parsed.get("jp_title") or "").strip()
    jp_summary = (parsed.get("jp_summary") or "").strip()

    if not jp_title:
        jp_title = title[:40]
    if not jp_summary:
        jp_summary = "要約の生成に失敗しました。"

    return {
        "jp_title": jp_title[:2000],
        "jp_summary": jp_summary[:2000],
    }


# ---------------------------
# Notion API
# ---------------------------
def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }


def notion_query_existing_urls():
    """Return set of URLs already in DB to prevent duplicates (recent 100)."""
    existing = set()
    endpoint = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"

    payload = {"page_size": 100}

    r = requests.post(endpoint, headers=notion_headers(), json=payload, timeout=30)
    if not r.ok:
        print("Notion query error:", r.status_code, r.text)
    r.raise_for_status()

    results = r.json().get("results", [])
    for page in results:
        props = page.get("properties", {})
        p = props.get("Source URL", {})
        u = p.get("url")
        if u:
            existing.add(norm_url(u))
    return existing


def notion_create_page(item):
    today = datetime.now(JST).date().isoformat()

    properties = {
        # Notion の Title には日本語タイトルを入れる
        "Title": {"title": [{"text": {"content": item["jp_title"][:2000]}}]},
        "Date": {"date": {"start": today}},
        "Region": {"select": {"name": item["region"]}},
        "Summary": {"rich_text": [{"text": {"content": item["jp_summary"][:2000]}}]},
        "Source URL": {"url": item["url"]},
        "Source Name": {"rich_text": [{"text": {"content": item["source_name"][:2000]}}]},
    }

    # 原文タイトル列を使いたい場合
    if USE_ORIGINAL_TITLE_COLUMN:
        properties["Original Title"] = {
            "rich_text": [{"text": {"content": item["title"][:2000]}}]
        }

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": properties,
    }

    r = requests.post(
        "https://api.notion.com/v1/pages",
        headers=notion_headers(),
        json=payload,
        timeout=30,
    )
    if not r.ok:
        print("Notion create page error:", r.status_code, r.text)
    r.raise_for_status()


# ---------------------------
# Main
# ---------------------------
def main():
    # collect
    items = scrape_jha(limit=8) + scrape_fih(limit=8)

    # dedupe by URL
    tmp = {}
    for it in items:
        tmp[it["url"]] = it
    items = list(tmp.values())

    # cap daily posts
    items = items[:2]

    # prevent duplicates in Notion (recent 100)
    existing_urls = notion_query_existing_urls()

    posted = 0
    for it in items:
        if it["url"] in existing_urls:
            continue

        # fetch article text
        try:
            soup = fetch_html(it["url"])
            body = take_text(soup, max_chars=5000)
            if not body:
                body = "(本文が取得できませんでした。)"
        except Exception as e:
            body = f"(本文取得失敗: {e})"

        # generate jp_title / jp_summary
        try:
            gen = gemini_generate_japanese_fields(
                it["title"], it["region"], it["source_name"], it["url"], body
            )
            it["jp_title"] = gen["jp_title"]
            it["jp_summary"] = gen["jp_summary"]
        except Exception as e:
            print("Gemini generation error:", e)
            it["jp_title"] = it["title"][:40]
            it["jp_summary"] = "要約の生成に失敗しました。"

        # write to Notion
        notion_create_page(it)
        posted += 1
        time.sleep(1)  # gentle rate limit

    print(f"Done. Posted {posted} new items.")


if __name__ == "__main__":
    main()
