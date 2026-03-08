import os, requests, json

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
CANDIDATES = [
    os.environ.get("NOTION_DATABASE_ID", ""),
    "31ddff69531b80c8a299d73a2bd0c34f",
    "31ddff69531b80759737d2009fcef8f7",
]

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
}

def try_get_database(dbid):
    url = f"https://api.notion.com/v1/databases/{dbid}"
    r = requests.get(url, headers=headers, timeout=30)
    return r.status_code, r.text[:400]

def try_get_page(pid):
    url = f"https://api.notion.com/v1/pages/{pid}"
    r = requests.get(url, headers=headers, timeout=30)
    return r.status_code, r.text[:400]

for x in CANDIDATES:
    x = x.strip()
    if not x:
        continue
    s1 = try_get_database(x)
    s2 = try_get_page(x)
    print("ID:", x)
    print("  GET /databases:", s1[0], s1[1])
    print("  GET /pages    :", s2[0], s2[1])
