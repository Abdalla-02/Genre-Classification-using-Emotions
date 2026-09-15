"""Fetch box-office gross for every Eerola film, keyed by the CSV's ``imdb_id`` column.

Supervisor request: "check the correlation between box office revenue and genre
classification from the soundtrack". This script only ACQUIRES the data and writes it to
``data/processed/Eerola_DB/box_office.csv`` with its provenance; the analysis is
``experiments/diagnostics/exp_box_office.py``.

Two sources, in order of preference:

  1. Wikidata (SPARQL, property P2142 "box office"), matched on IMDb id (P345). Clean,
     no scraping, no API key. Returns several figures per film (worldwide, domestic,
     different reporting dates); the LARGEST USD figure is taken as the worldwide gross.
  2. Box Office Mojo title pages, for films Wikidata lacks. Public pages, one polite
     request per second. Where BOM shows only a domestic figure that is what is stored,
     and the ``scope`` column says so.

Every row records ``source`` and ``scope`` so the analysis can restrict to worldwide-only
or to a single source. Non-USD figures are kept with their currency and excluded from the
analysis rather than converted at a guessed rate.

Known data-quality issues in the enriched CSV surfaced by this fetch (documented, not
fixed here): ``Blanc`` carries the IMDb id of "Adèle Blanc-Sec" (2010) rather than
Kieslowski's "Trois couleurs: Blanc" (1994), and ``Pride and Prejudice`` carries the 1940
film's id where the scored soundtrack is more likely the 2005 film. Neither film has box
office data from either source, so they do not enter the analysis.

Run:  python experiments/features/fetch_box_office.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402

OUT = config.PROCESSED_DIR / "box_office.csv"
UA = {"User-Agent": "thesis-box-office/1.0 (student research; OVGU Magdeburg)"}


def from_wikidata(ids: list[str]) -> pd.DataFrame:
    values = " ".join(f'"{i}"' for i in ids)
    q = f"""
    SELECT ?imdb ?filmLabel ?box ?unitLabel ?date WHERE {{
      VALUES ?imdb {{ {values} }}
      ?film wdt:P345 ?imdb .
      OPTIONAL {{ ?film p:P2142 ?s . ?s ps:P2142 ?box .
                 OPTIONAL {{ ?s psv:P2142/wikibase:quantityUnit ?unit . }} }}
      OPTIONAL {{ ?film wdt:P577 ?date . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": q, "format": "json"})
    data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                            timeout=90))
    rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
    df = pd.DataFrame(rows)
    df["box"] = pd.to_numeric(df.get("box"), errors="coerce")
    df["year"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True).dt.year
    out = []
    for imdb, g in df.groupby("imdb"):
        usd = g[g["unitLabel"] == "United States dollar"].dropna(subset=["box"])
        title = g["filmLabel"].iloc[0]
        year = g["year"].min()
        if len(usd):
            out.append({"imdb_id": imdb, "title": title, "year": year,
                        "gross": float(usd["box"].max()), "currency": "USD",
                        "scope": "worldwide (max of reported figures)",
                        "source": "wikidata"})
        else:
            other = g.dropna(subset=["box"])
            out.append({"imdb_id": imdb, "title": title, "year": year,
                        "gross": float(other["box"].max()) if len(other) else None,
                        "currency": other["unitLabel"].iloc[0] if len(other) else None,
                        "scope": "non-USD figure" if len(other) else None,
                        "source": "wikidata" if len(other) else None})
    return pd.DataFrame(out)


def from_box_office_mojo(imdb_id: str) -> dict | None:
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(f"https://www.boxofficemojo.com/title/{imdb_id}/",
                                   headers=UA), timeout=20).read().decode("utf-8", "replace")
    except Exception:
        return None
    def grab(label: str):
        # a figure must start with a digit: "$," (an empty cell) must not match
        m = re.search(label + r".*?\$(\d[\d,]*)", html, re.S)
        return int(m.group(1).replace(",", "")) if m else None
    ww_v, dom_v = grab("Worldwide"), grab("Domestic")
    if ww_v is None and dom_v is None:
        return None
    if ww_v and dom_v and ww_v > dom_v:
        return {"gross": float(ww_v), "scope": "worldwide"}
    return {"gross": float(ww_v or dom_v), "scope": "domestic only"}


def main() -> None:
    df = pd.read_csv(config.SET1_CSV)
    films = (df.dropna(subset=["imdb_id"])
               .drop_duplicates("imdb_id")[["imdb_id", "soundtrack"]]
               .rename(columns={"soundtrack": "soundtrack"}))
    ids = sorted(films["imdb_id"])
    print(f"{len(ids)} films with an IMDb id")

    wd = from_wikidata(ids)
    have = set(wd.dropna(subset=["gross"]).loc[wd["currency"] == "USD", "imdb_id"])
    print(f"wikidata: USD gross for {len(have)}")

    for imdb in ids:
        if imdb in have:
            continue
        bom = from_box_office_mojo(imdb)
        time.sleep(1.0)
        if bom:
            i = wd.index[wd["imdb_id"] == imdb]
            if len(i):
                wd.loc[i, ["gross", "currency", "scope", "source"]] = \
                    [bom["gross"], "USD", bom["scope"], "boxofficemojo"]
            else:
                wd = pd.concat([wd, pd.DataFrame([{"imdb_id": imdb, "title": None,
                                                    "year": None, "currency": "USD",
                                                    "source": "boxofficemojo", **bom}])])
            print(f"  boxofficemojo: {imdb} -> {bom['gross']:,.0f} ({bom['scope']})")

    out = films.merge(wd, on="imdb_id", how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    usable = out[(out["currency"] == "USD") & out["gross"].notna()]
    print(f"\nwrote {OUT}: {len(out)} films, {len(usable)} with a USD gross "
          f"({(usable['scope'].str.startswith('worldwide')).sum()} worldwide, "
          f"{(usable['scope'] == 'domestic only').sum()} domestic-only)")


if __name__ == "__main__":
    main()
