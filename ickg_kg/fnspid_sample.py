"""Random sampling of the FNSPID news corpus by remote byte-offset.

Why offset sampling: `nasdaq_exteral_data.csv` is ~23 GB, grouped by ticker and
newest-first, so reading the head returns only symbol "A" / recent dates. To get
a temporally- and symbol-diverse *random* draw without downloading 23 GB, we pick
random byte offsets and read a small window via HTTP Range requests (the HF CDN
returns 206 / Accept-Ranges: bytes), then snap to the next real record boundary.

Boundary anchor: every record starts with `\n<rowindex>.0,<YYYY-MM-DD ...>,`.
Article bodies contain newlines/commas/escaped quotes, so we parse the recovered
record with the stdlib csv reader (quote-aware) rather than splitting on commas.

Bias note (documented in the README): selecting "the record after a random
offset" makes selection probability proportional to the *previous* record's
length, i.e. approximately uniform over records. Very long articles are
marginally under-represented. This is acceptable for a research substrate.
"""
from __future__ import annotations

import csv
import io
import re
import random
import time
from dataclasses import dataclass, asdict

import requests

# record start: newline, float row-index, comma, an ISO date, up to next comma
_REC_RE = re.compile(rb"\n(\d+\.0,\d{4}-\d{2}-\d{2}[^,\n]*,)")
_WINDOW = 262_144          # 256 KB read window (safely spans several records)
_HEADER_BYTES = 65_536


@dataclass
class Article:
    row_index: str
    date: str
    symbol: str
    title: str
    body: str
    url: str
    publisher: str
    author: str
    byte_offset: int

    @property
    def article_id(self) -> str:
        return f"r{self.row_index}"


class FnspidSampler:
    def __init__(self, repo: str, news_path: str, timeout: int = 120):
        self.url = f"https://huggingface.co/datasets/{repo}/resolve/main/{news_path}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ickg-kg/0.1"})
        self.timeout = timeout
        self.size = self._content_length()
        self.header, self._header_end = self._read_header()
        self._col = {name: i for i, name in enumerate(self.header)}

    # -- low level ----------------------------------------------------------
    def _content_length(self) -> int:
        r = self.session.head(self.url, allow_redirects=True, timeout=self.timeout)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl is None:
            raise RuntimeError("server did not report Content-Length")
        return int(cl)

    def _read_range(self, start: int, length: int) -> bytes:
        end = min(start + length - 1, self.size - 1)
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                             allow_redirects=True, timeout=self.timeout)
        if r.status_code != 206:
            raise RuntimeError(f"expected HTTP 206 partial content, got {r.status_code}")
        return r.content

    def _read_header(self) -> tuple[list[str], int]:
        buf = self._read_range(0, _HEADER_BYTES)
        nl = buf.find(b"\n")
        line = buf[:nl].decode("utf-8", "ignore")
        header = next(csv.reader([line]))
        return header, nl + 1

    # -- record recovery ----------------------------------------------------
    def _record_at(self, offset: int) -> list[str] | None:
        buf = self._read_range(offset, _WINDOW)
        starts = [m.start() for m in _REC_RE.finditer(buf)]
        if len(starts) < 2:
            return None
        rec = buf[starts[0] + 1:starts[1]].decode("utf-8", "ignore")
        rows = list(csv.reader(io.StringIO(rec)))
        if len(rows) != 1 or len(rows[0]) != len(self.header):
            return None
        return rows[0]

    def _to_article(self, row: list[str], offset: int) -> Article | None:
        def g(name: str) -> str:
            i = self._col.get(name)
            return row[i].strip() if i is not None and i < len(row) else ""
        body = g("Article")
        date = g("Date")
        if not body or not date:
            return None
        return Article(
            row_index=g(self.header[0]) or "?",
            date=date,
            symbol=g("Stock_symbol"),
            title=g("Article_title"),
            body=body,
            url=g("Url"),
            publisher=g("Publisher"),
            author=g("Author"),
            byte_offset=offset,
        )

    # -- public -------------------------------------------------------------
    def sample(self, n: int, seed: int, min_body_chars: int = 200,
               min_body_words: int = 0, year_range: tuple[int, int] | None = None,
               verbose: bool = True) -> list[Article]:
        rng = random.Random(seed)
        out: list[Article] = []
        seen: set[str] = set()
        tries = 0
        max_tries = n * 40 + 200
        t0 = time.time()
        lo, hi = self._header_end, self.size - _WINDOW
        while len(out) < n and tries < max_tries:
            tries += 1
            off = rng.randint(lo, hi)
            try:
                row = self._record_at(off)
            except Exception:
                continue
            if row is None:
                continue
            art = self._to_article(row, off)
            if art is None or len(art.body) < min_body_chars:
                continue
            if min_body_words and len(art.body.split()) < min_body_words:
                continue
            if year_range is not None:
                try:
                    yr = int(art.date[:4])
                except (ValueError, TypeError):
                    continue
                if not (year_range[0] <= yr <= year_range[1]):
                    continue
            key = art.article_id if art.row_index != "?" else f"{art.symbol}|{art.date}|{art.title}"
            if key in seen:
                continue
            seen.add(key)
            out.append(art)
            if verbose and len(out) % 50 == 0:
                rate = len(out) / max(1e-6, time.time() - t0)
                print(f"  sampled {len(out)}/{n}  ({rate:.1f}/s, {tries} reads)")
        if len(out) < n:
            print(f"WARNING: only sampled {len(out)}/{n} after {tries} reads")
        return out


def articles_to_records(articles: list[Article]) -> list[dict]:
    return [asdict(a) for a in articles]
