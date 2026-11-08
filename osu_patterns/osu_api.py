"""Small osu! API v2 client (client-credentials flow) for user tags and .osu files.

Needs OSU_CLIENT_ID / OSU_CLIENT_SECRET (create an OAuth app at
https://osu.ppy.sh/home/account/edit#oauth). Stays well under the API's rate guidance.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://osu.ppy.sh/api/v2"


class OsuApi:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None, min_interval: float = 1.1):
        self.client_id = client_id or os.environ["OSU_CLIENT_ID"]
        self.client_secret = client_secret or os.environ["OSU_CLIENT_SECRET"]
        self.min_interval = min_interval
        self._token: str | None = None
        self._token_exp = 0.0
        self._last = 0.0

    def _auth(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        body = urllib.parse.urlencode({"client_id": self.client_id, "client_secret": self.client_secret,
                                       "grant_type": "client_credentials", "scope": "public"}).encode()
        with urllib.request.urlopen(urllib.request.Request("https://osu.ppy.sh/oauth/token", data=body)) as r:
            tok = json.load(r)
        self._token, self._token_exp = tok["access_token"], time.time() + tok["expires_in"]
        return self._token

    def _wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()

    def get(self, path: str, **params) -> dict:
        self._wait()
        q = urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(f"{API}{path}{'?' + q if q else ''}",
                                     headers={"Authorization": f"Bearer {self._auth()}", "Accept": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(5 * 2**attempt)
                    continue
                raise
        raise RuntimeError(f"giving up on {path}")

    # ------------------------------------------------------------------ helpers
    def tag_names(self) -> dict[int, str]:
        return {t["id"]: t["name"] for t in self.get("/tags")["tags"]}

    def beatmapset(self, set_id: int) -> dict:
        return self.get(f"/beatmapsets/{set_id}")

    def search_ranked(self, pages: int, sort: str = "plays_desc", mode: int = 0):
        """Yield ranked beatmapsets (search payload), newest-API cursor pagination."""
        cursor = None
        for _ in range(pages):
            params = {"m": mode, "s": "ranked", "sort": sort}
            if cursor:
                params["cursor_string"] = cursor
            res = self.get("/beatmapsets/search", **params)
            yield from res.get("beatmapsets", [])
            cursor = res.get("cursor_string")
            if not cursor:
                break

    def download_osu(self, beatmap_id: int, dest: Path) -> Path:
        """Raw .osu file (no auth needed)."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        self._wait()
        with urllib.request.urlopen(f"https://osu.ppy.sh/osu/{beatmap_id}", timeout=30) as r:
            dest.write_bytes(r.read())
        return dest


def beatmap_tags(set_json: dict, names: dict[int, str]) -> dict[int, dict[str, int]]:
    """{beatmap_id: {tag_name: votes}} for the osu!standard difficulties of a set."""
    out = {}
    for b in set_json.get("beatmaps", []):
        if b.get("mode_int", 0) != 0 and b.get("mode") not in (None, "osu"):
            continue
        tv = {}
        for e in b.get("top_tag_ids") or []:
            n = names.get(e.get("tag_id"))
            if n:
                tv[n] = int(e.get("count", 0))
        out[b["id"]] = tv
    return out
