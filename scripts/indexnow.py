"""Tell IndexNow search engines (Bing, Yandex, …) that the site's pages changed.

Run after a deploy that adds or changes pages:

    uv run python scripts/indexnow.py
"""

import json
import urllib.request
from urllib.parse import urlparse

from backend import i18n


def main() -> None:
    body = {
        "host": urlparse(i18n.BASE_URL).netloc,
        "key": i18n.INDEXNOW_KEY,
        "keyLocation": f"{i18n.BASE_URL}/{i18n.INDEXNOW_KEY}.txt",
        "urlList": i18n.all_urls(),
    }
    request = urllib.request.Request(
        "https://api.indexnow.org/indexnow", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.status, *body["urlList"], sep="\n")


if __name__ == "__main__":
    main()
