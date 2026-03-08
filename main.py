import os
import time
import hashlib
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


# ---------------------------
# Helpers
# ---------------------------
def norm_url(url: str) -> str:
    url = (url or "").strip()
    return url[:-1] if url.endswith("/") else url


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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


# ---------------------------
# Scrape sources
# ---------------------------
def scrape_jha(limit: int = 8):
    """JHA: use RSS feed to avoid 403 from GitHub Actions runners."""
    feed = feedparser.parse(JHA_FEED_URL)
    items = []
    for e in feed.entries[:limit * 3]:
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
# Gemini summary
# ---------------------------
def post_with_backoff(url, headers, json_payload, timeout=60, max_retries=6):
    wait = 2
    for i in range(max_retries):
        r = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        if r.status_code != 429:
            return r
        # 429: rate limit -> wait and retry
        time.sleep(wait)
        wait = min(wait * 2, 60)
    return r

def gemini_summarize(title: str, region: str, source_name: str, url: str, body: str) -> str:
    prompt = f"""
あなたはフィールドホッケーのニュース要約アシスタントです。
以下の情報を日本語で要約してください。

要件:
- 2〜4文
- 100〜180字程度
- 「何が起きたか」「なぜ注目か」を含める
- 本文が短い/不十分なら推測しすぎない
- 誇張しない

地域: {region}
ソース: {source_name}
タイトル: {title}
URL: {url}

本文:
{body}
""".strip()

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )

    r = post_with_backoff(
        f"{endpoint}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json_payload={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
        max_retries=8,
    )
    r.raise_for_status()
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return "要約の生成に失敗しました。"


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
        # This prints the real reason for 400 (property name mismatch etc.)
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

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": item["title"][:2000]}}]},
            "Date": {"date": {"start": today}},
            "Region": {"select": {"name": item["region"]}},
            "Summary": {"rich_text": [{"text": {"content": item["summary"][:2000]}}]},
            "Source URL": {"url": item["url"]},
            "Source Name": {"rich_text": [{"text": {"content": item["source_name"][:2000]}}]},
        },
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

        # summarize
        it["summary"] = gemini_summarize(it["title"], it["region"], it["source_name"], it["url"], body)

        # write to Notion
        notion_create_page(it)
        posted += 1
        time.sleep(1)  # gentle rate limit

    print(f"Done. Posted {posted} new items.")


if __name__ == "__main__":
    main()
