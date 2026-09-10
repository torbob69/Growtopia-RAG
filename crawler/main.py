import os
import re
import asyncio
import psutil
import requests
from urllib.parse import quote, unquote
from typing import List

from crawl4ai import AsyncWebCrawler, HTTPCrawlerConfig, CrawlerRunConfig, CacheMode
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

WIKI = "https://growtopiawiki.com"
UA = "Growtopia-RAG-crawler"

# ponytail: namespace pages the allpages API still leaks (odd namespace numbering on this fork)
JUNK_PREFIXES = ("Message Wall", "Board:", "Blog:", "User:", "User talk:", "Talk:",
                 "Help:", "Painting:", "Ns710:", "Cateogry:", "Category:", "Template:")
MIN_CONTENT_CHARS = 400  # after cleaning, incl. the url comment + "# Title" header (~80 chars)


def clean_markdown(md: str) -> str:
    """Strip wiki chrome so chunks/embeddings aren't drowned in link URLs and infobox noise."""
    # wiki footer
    md = re.sub(r'\n*Retrieved from "\[.*$', "", md, flags=re.S)
    # [text](url "title") -> text ; [](url) -> nothing. handles one level of \(...\) in the url.
    md = re.sub(r'\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)', lambda m: m.group(1), md)
    # raw ARGB hex from infobox palette cells
    md = re.sub(r'#[0-9A-Fa-f]{8}\b', "", md)
    # table separator rows (| --- | --- |)
    md = re.sub(r'^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$', "", md, flags=re.M)

    def _row(m):
        cells = [c.strip() for c in m.group(1).split("|")]
        cells = [c for c in cells if c and not re.fullmatch(r':?-{3,}:?', c)]
        if not cells:
            return ""
        if len(cells) == 2:
            return f"{cells[0]}: {cells[1]}"
        return " — ".join(cells)

    md = re.sub(r'^\s*\|(.+?)\|\s*$', _row, md, flags=re.M)
    # leftover pipe edges from multi-line infobox cells; breadcrumb; heading double-space
    md = re.sub(r'^\s*\|[ \t]*', "", md, flags=re.M)
    md = re.sub(r'[ \t]*\|[ \t]*$', "", md, flags=re.M)
    md = re.sub(r'^< .*$', "", md, flags=re.M)
    md = re.sub(r'^(#{2,})  +', r'\1 ', md, flags=re.M)
    # trailing hard-break backslashes / whitespace, then empty bullets left by dropped links
    md = re.sub(r'[ \t]*\\?[ \t]*$', "", md, flags=re.M)
    md = re.sub(r'^\s*[*•]\s*$', "", md, flags=re.M)

    # de-dupe the description, which the scrape repeats 2-3x under the title
    lines = md.split("\n")
    title = lines[2][2:].strip() if len(lines) > 2 and lines[2].startswith("# ") else ""
    head, seen, kept = lines[:4], set(), []
    for ln in lines[4:12]:
        s = ln.strip()
        if s and (s in seen or s == title):
            continue
        seen.add(s)
        kept.append(ln)
    md = "\n".join(head + kept + lines[12:])

    md = re.sub(r'\n{3,}', "\n\n", md)
    return md.strip() + "\n"


def get_growtopia_urls(limit: int | None = None) -> List[str]:
    """All main-namespace article URLs, via the MediaWiki API (no sitemap on this wiki)."""
    params = {
        "action": "query", "list": "allpages",
        "aplimit": "500",        # API max per request
        "apnamespace": "0",      # articles only, no Category:/Template:/File:
        "apfilterredir": "nonredirects",
        "format": "json",
    }
    urls: List[str] = []
    cont: dict = {}
    while True:
        r = requests.get(f"{WIKI}/api.php", params={**params, **cont},
                         headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        data = r.json()
        for p in data["query"]["allpages"]:
            urls.append(f"{WIKI}/w/" + quote(p["title"].replace(" ", "_"), safe="/"))
        if limit and len(urls) >= limit:
            return urls[:limit]
        if "continue" not in data:
            return urls
        cont = data["continue"]


def _outfile(url: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("/w/", 1)[-1])[:150] or "index"
    return os.path.join(__output__, f"{slug}.md")


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    print("\n=== Parallel Crawling with Crawler Reuse + Memory Check ===")
    os.makedirs(__output__, exist_ok=True)

    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss
        peak_memory = max(peak_memory, current_mem)
        print(f"{prefix} Current Memory: {current_mem // (1024*1024)} MB, Peak: {peak_memory // (1024*1024)} MB")

    # ponytail: HTTP fetch, no headless browser. Server-rendered MediaWiki + a 16GB box that OOM'd on Chromium.
    crawler = AsyncWebCrawler(
        crawler_strategy=AsyncHTTPCrawlerStrategy(
            browser_config=HTTPCrawlerConfig(headers={"User-Agent": UA}),
        )
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector="#bodyContent",
        excluded_tags=["nav", "footer", "form", "script", "style", "header"],
        exclude_external_links=True,
        exclude_all_images=True,
        check_robots_txt=True,
        verbose=False,
    )

    await crawler.start()
    try:
        success_count = 0
        skip_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = [crawler.arun(url=url, config=crawl_config) for url in batch]

            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    title = unquote(url.split("/w/", 1)[-1]).replace("_", " ")
                    doc = clean_markdown(
                        f"<!-- {url} -->\n\n# {title}\n\n{result.markdown.raw_markdown}"
                    )
                    if title.startswith(JUNK_PREFIXES) or len(doc) < MIN_CONTENT_CHARS:
                        skip_count += 1
                        continue
                    with open(_outfile(url), "w", encoding="utf-8") as f:
                        f.write(doc)
                    success_count += 1
                else:
                    print(f"Failed {url}: {result.error_message}")
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Skipped (junk/too short): {skip_count}")
        print(f"  - Failed: {fail_count}")
        print(f"  - Output: {__output__}")
    finally:
        print("\nClosing crawler...")
        await crawler.close()
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024*1024)}")


async def main():
    urls = get_growtopia_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls, max_concurrent=5)
    else:
        print("No URLs found to crawl")


def _selftest():
    raw = (
        '<!-- https://growtopiawiki.com/w/Apple -->\n\n# Apple\n\n'
        'This keeps doctors away!\nApple (Rarity: 19)\nThis keeps doctors away!\nProperties\n'
        '| Data |  \n| --- |  \n'
        '| Type  |  Consumable  |  \n'
        '|  **2h 3m 49s** #F91415FF #66D91FFF  |  \n'
        'See [Blueberry](https://growtopiawiki.com/w/Blueberry "Blueberry") and '
        '[Time-Tossed!](https://growtopiawiki.com/w/Time-Tossed!_\\(update\\) "Time-Tossed! \\(update\\)").\n'
        '  * [](https://growtopiawiki.com/w/File:X.png "a caption")\n\n\n\n'
        'Retrieved from "[https://growtopiawiki.com/index.php?title=Apple&oldid=1](https://growtopiawiki.com/x)"\n'
    )
    out = clean_markdown(raw)
    assert "https://growtopiawiki.com/w/Blueberry" not in out, "link url not stripped"
    assert "See Blueberry and Time-Tossed!." in out, out
    assert "Retrieved from" not in out, "footer not stripped"
    assert "#F91415FF" not in out, "hex not stripped"
    assert "| --- |" not in out and "| Data |" not in out, "table pipes survived"
    assert "\n---\n" not in out, "bare hr from collapsed separator survived"
    assert "Type: Consumable" in out, out
    assert out.count("This keeps doctors away!") == 1, "description not de-duped"
    assert "\n\n\n" not in out, "blank-line run not collapsed"
    print("selftest ok\n" + out)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        asyncio.run(main())
