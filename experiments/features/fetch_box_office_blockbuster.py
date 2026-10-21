"""Fetch box-office gross (and budget) for the 110 films of the Blockbuster dataset.

Supervisor request, fourth meeting: correlate box-office gross with the Blockbuster
dataset. The dataset ships no identifiers beyond a title slug
(``film_genre_master_list.csv``), so each slug is first resolved to a Wikidata item and
its IMDb id, using the film's title and release year as given in Ma et al.'s Appendix S1
(the year below is the US release year; the match allows one year either side, because
festival premieres are often dated a year earlier).

Gross, in order of preference:

  1. Box Office Mojo, worldwide figure, by IMDb id. One source and one definition for
     every film, which matters more here than on Eerola: these are all 2014-2019 studio
     releases, so a consistent worldwide figure is available for nearly all of them.
  2. Wikidata (P2142 "box office", largest USD value) where Box Office Mojo has none.

Budget comes from Wikidata (P2130 "cost", USD) and is optional; it is used only as a
control, since gross largely follows budget.

Every row records the resolved Wikidata item, its label and year, and the source and
scope of the gross, so every match can be checked by eye. Nothing is inflation-adjusted;
over six years of release that is a small effect, and the analysis controls for year.

Run:  python experiments/features/fetch_box_office_blockbuster.py [--fresh]
      (network; about 5 min when it can reuse an earlier run's matches, up to 30 min
      from scratch because Wikimedia rate-limits the title search)
Writes data/processed/Blockbuster/box_office.csv; the analysis is
``experiments/diagnostics/exp_box_office_blockbuster.py`` and runs offline from it.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_box_office import UA, from_box_office_mojo  # noqa: E402

OUT = config.DATA_ROOT / "processed" / "Blockbuster" / "box_office.csv"
FILM_TYPES = {"Q11424", "Q24869", "Q202866", "Q29168811", "Q17517379", "Q506240",
              "Q18011172", "Q20650540"}   # film, feature, animated (feature), ...

# slug -> (title as in Ma et al., Appendix S1, US release year)
TITLES = {
    "300_rise_of_an_empire": ("300: Rise of an Empire", 2014),
    "aladdin": ("Aladdin", 2019),
    "alita_battle_angel": ("Alita: Battle Angel", 2019),
    "annabelle": ("Annabelle", 2014),
    "antman": ("Ant-Man", 2015),
    "antman_and_the_wasp": ("Ant-Man and the Wasp", 2018),
    "aquaman": ("Aquaman", 2018),
    "avengers_age_of_ultron": ("Avengers: Age of Ultron", 2015),
    "avengers_endgame": ("Avengers: Endgame", 2019),
    "avengers_infinity_war": ("Avengers: Infinity War", 2018),
    "beauty_and_the_beast": ("Beauty and the Beast", 2017),
    "black_mass": ("Black Mass", 2015),
    "blended": ("Blended", 2014),
    "bohemian_rhapsody": ("Bohemian Rhapsody", 2018),
    "captain_america_civil_war": ("Captain America: Civil War", 2016),
    "captain_america_the_winter_soldier": ("Captain America: The Winter Soldier", 2014),
    "captain_marvel": ("Captain Marvel", 2019),
    "chappaquiddick": ("Chappaquiddick", 2018),
    "christopher_robin": ("Christopher Robin", 2018),
    "cinderella": ("Cinderella", 2015),
    "collateral_beauty": ("Collateral Beauty", 2016),
    "crazy_rich_asians": ("Crazy Rich Asians", 2018),
    "creed": ("Creed", 2015),
    "doctor_strange": ("Doctor Strange", 2016),
    "dora_and_the_lost_city_of_gold": ("Dora and the Lost City of Gold", 2019),
    "dumb_and_dumber_to": ("Dumb and Dumber To", 2014),
    "dunkirk": ("Dunkirk", 2017),
    "edge_of_tomorrow": ("Edge of Tomorrow", 2014),
    "entourage": ("Entourage", 2015),
    "first_man": ("First Man", 2018),
    "focus": ("Focus", 2015),
    "geostorm": ("Geostorm", 2017),
    "going_in_style": ("Going in Style", 2017),
    "halloween_2018": ("Halloween", 2018),
    "happy_death_day": ("Happy Death Day", 2017),
    "hitman_agent_47": ("Hitman: Agent 47", 2015),
    "horrible_bosses_2": ("Horrible Bosses 2", 2014),
    "house_with_a_clock_in_its_walls": ("The House with a Clock in Its Walls", 2018),
    "how_to_be_single": ("How to Be Single", 2016),
    "in_the_heart_of_the_sea": ("In the Heart of the Sea", 2015),
    "incredibles_2": ("Incredibles 2", 2018),
    "interstellar": ("Interstellar", 2014),
    "into_the_woods": ("Into the Woods", 2014),
    "it_chapter_two": ("It Chapter Two", 2019),
    "johnny_english_strikes_again": ("Johnny English Strikes Again", 2018),
    "jumanji_welcome_to_the_jungle": ("Jumanji: Welcome to the Jungle", 2017),
    "jupiter_ascending": ("Jupiter Ascending", 2015),
    "justice_league": ("Justice League", 2017),
    "king_arthur_legend_of_the_sword": ("King Arthur: Legend of the Sword", 2017),
    "kingsman_the_secret_service": ("Kingsman: The Secret Service", 2015),
    "kong_skull_island": ("Kong: Skull Island", 2017),
    "krampus": ("Krampus", 2015),
    "lights_out": ("Lights Out", 2016),
    "mad_max_fury_road": ("Mad Max: Fury Road", 2015),
    "magic_mike_xxl": ("Magic Mike XXL", 2015),
    "maleficent": ("Maleficent", 2014),
    "me_before_you": ("Me Before You", 2016),
    "megan_leavey": ("Megan Leavey", 2017),
    "miss_peregrines_home_for_peculiar_children":
        ("Miss Peregrine's Home for Peculiar Children", 2016),
    "mission_impossible_fallout": ("Mission: Impossible – Fallout", 2018),
    "mission_impossible_rogue_nation": ("Mission: Impossible – Rogue Nation", 2015),
    "moonlight": ("Moonlight", 2016),
    "mortal_engines": ("Mortal Engines", 2018),
    "murder_on_the_orient_express": ("Murder on the Orient Express", 2017),
    "need_for_speed": ("Need for Speed", 2014),
    "oceans_8": ("Ocean's 8", 2018),
    "paddington_2": ("Paddington 2", 2017),
    "pan": ("Pan", 2015),
    "parasite": ("Parasite", 2019),
    "pet_sematary": ("Pet Sematary", 2019),
    "pirates_of_the_caribbean_dead_men_tell_no_tales":
        ("Pirates of the Caribbean: Dead Men Tell No Tales", 2017),
    "pokemon_detective_pikachu": ("Pokémon Detective Pikachu", 2019),
    "queen_of_katwe": ("Queen of Katwe", 2016),
    "rambo_last_blood": ("Rambo: Last Blood", 2019),
    "rampage": ("Rampage", 2018),
    "ready_player_one": ("Ready Player One", 2018),
    "run_all_night": ("Run All Night", 2015),
    "san_andreas": ("San Andreas", 2015),
    "shazam": ("Shazam!", 2019),
    "sicario": ("Sicario", 2015),
    "smallfoot": ("Smallfoot", 2018),
    "solo_a_star_wars_story": ("Solo: A Star Wars Story", 2018),
    "spotlight": ("Spotlight", 2015),
    "star_trek_beyond": ("Star Trek Beyond", 2016),
    "star_wars_the_force_awakens": ("Star Wars: The Force Awakens", 2015),
    "star_wars_the_last_jedi": ("Star Wars: The Last Jedi", 2017),
    "stuber": ("Stuber", 2019),
    "suicide_squad": ("Suicide Squad", 2016),
    "sully": ("Sully", 2016),
    "tammy": ("Tammy", 2014),
    "teen_titans_go_to_the_movies": ("Teen Titans Go! To the Movies", 2018),
    "the_boy": ("The Boy", 2016),
    "the_conjuring_2": ("The Conjuring 2", 2016),
    "the_hustle": ("The Hustle", 2019),
    "the_imitation_game": ("The Imitation Game", 2014),
    "the_intern": ("The Intern", 2015),
    "the_judge": ("The Judge", 2014),
    "the_legend_of_tarzan": ("The Legend of Tarzan", 2016),
    "the_man_from_uncle": ("The Man from U.N.C.L.E.", 2015),
    "the_meg": ("The Meg", 2018),
    "the_peanut_butter_falcon": ("The Peanut Butter Falcon", 2019),
    "the_post": ("The Post", 2017),
    "this_is_where_i_leave_you": ("This Is Where I Leave You", 2014),
    "thor_ragnarok": ("Thor: Ragnarok", 2017),
    "tolkien": ("Tolkien", 2019),
    "tomorrowland": ("Tomorrowland", 2015),
    "transcendence": ("Transcendence", 2014),
    "vacation": ("Vacation", 2015),
    "venom": ("Venom", 2018),
    "wonder_woman": ("Wonder Woman", 2017),
}


def _get(url: str, timeout: int = 60, tries: int = 6):
    """GET JSON, backing off politely when Wikimedia rate-limits (HTTP 429)."""
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout))
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 5 * 2 ** k))


def search(title: str) -> list[str]:
    """Candidate Wikidata items for a title (English label/alias search)."""
    q = urllib.parse.urlencode({"action": "wbsearchentities", "search": title,
                                "language": "en", "type": "item", "limit": 15,
                                "format": "json"})
    return [h["id"] for h in _get(f"https://www.wikidata.org/w/api.php?{q}")["search"]]


def by_enwiki(title: str, year: int) -> list[str]:
    """Items whose English Wikipedia article is "<title> (<year> film)" or "<title> (film)".

    For one-word titles ("Focus", "Pan") the label search returns fifteen other things
    before the film; Wikipedia's disambiguated article title names it exactly.
    """
    out = []
    for page in (f"{title} ({year} film)", f"{title} (film)"):
        q = urllib.parse.urlencode({"action": "wbgetentities", "sites": "enwiki",
                                    "titles": page, "props": "info", "format": "json"})
        ents = _get(f"https://www.wikidata.org/w/api.php?{q}").get("entities", {})
        out += [k for k in ents if k.startswith("Q")]
        time.sleep(1.0)
    return out


def describe(qids: list[str]) -> pd.DataFrame:
    """Type, year, IMDb id, USD box office and USD budget for a set of items."""
    values = " ".join(f"wd:{q}" for q in qids)
    usd = "wd:Q4917"
    query = f"""
    SELECT ?item ?itemLabel ?type ?date ?imdb ?box ?boxUnit ?cost ?costUnit WHERE {{
      VALUES ?item {{ {values} }}
      OPTIONAL {{ ?item wdt:P31 ?type . }}
      OPTIONAL {{ ?item wdt:P577 ?date . }}
      OPTIONAL {{ ?item wdt:P345 ?imdb . }}
      OPTIONAL {{ ?item p:P2142 ?bs . ?bs psv:P2142 ?bv .
                 ?bv wikibase:quantityAmount ?box ; wikibase:quantityUnit ?boxUnit . }}
      OPTIONAL {{ ?item p:P2130 ?cs . ?cs psv:P2130 ?cv .
                 ?cv wikibase:quantityAmount ?cost ; wikibase:quantityUnit ?costUnit . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": query, "format": "json"})
    rows = [{k: v["value"] for k, v in b.items()}
            for b in _get(url, 120)["results"]["bindings"]]
    cols = ["item", "itemLabel", "type", "date", "imdb", "box", "boxUnit", "cost", "costUnit"]
    df = pd.DataFrame(rows).reindex(columns=cols)   # an item may lack every optional field
    for c in ("item", "type", "boxUnit", "costUnit"):
        df[c] = df[c].astype("string").str.rsplit("/", n=1).str[-1]
    df["year"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.year
    for c in ("box", "cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["boxUnit"] != usd.split(":")[1], "box"] = None
    df.loc[df["costUnit"] != usd.split(":")[1], "cost"] = None
    return df


def resolve(slug: str, title: str, year: int, info: pd.DataFrame, cands: list[str]):
    """The candidate that is a film released within a year of ``year`` (exact year first)."""
    best = None
    for rank, q in enumerate(cands):
        g = info[info["item"] == q]
        if g.empty or not set(g["type"].dropna()) & FILM_TYPES:
            continue
        years = g["year"].dropna()
        if years.empty:
            continue
        gap = int(abs(years - year).min())
        if gap > 1:
            continue
        key = (gap, rank)
        if best is None or key < best[0]:
            best = (key, q, g)
    if best is None:
        return {"slug": slug, "title": title, "year_hint": year}
    _, q, g = best
    return {"slug": slug, "title": title, "year_hint": year, "wikidata": q,
            "wikidata_label": g["itemLabel"].iloc[0],
            "year": int(g["year"].dropna().min()),
            "imdb_id": g["imdb"].dropna().iloc[0] if g["imdb"].notna().any() else None,
            "wikidata_gross_usd": float(g["box"].max()) if g["box"].notna().any() else None,
            "budget_usd": float(g["cost"].max()) if g["cost"].notna().any() else None}


def main() -> None:
    slugs = pd.read_csv(config.BLOCKBUSTER_DIR / "film_genre_master_list.csv",
                        header=None)[0].tolist()
    missing = set(slugs) - set(TITLES)
    if missing:
        sys.exit(f"no title for: {sorted(missing)}")
    print(f"{len(slugs)} films; resolving on Wikidata ...", flush=True)

    # Resolving 110 titles takes a while under Wikimedia's rate limit, so a re-run keeps
    # the items an earlier run already resolved (pass --fresh to resolve everything anew).
    prev = {}
    if OUT.exists() and "--fresh" not in sys.argv:
        old = pd.read_csv(OUT)
        prev = {r["slug"]: r for _, r in old.iterrows() if pd.notna(r.get("wikidata"))}
    cands = {}
    for s in slugs:
        if s in prev:
            cands[s] = [prev[s]["wikidata"]]
            continue
        cands[s] = search(TITLES[s][0])
        time.sleep(1.0)
    allq = sorted({q for c in cands.values() for q in c})
    info = pd.concat([describe(allq[i:i + 150]) for i in range(0, len(allq), 150)])
    rows = [resolve(s, *TITLES[s], info, cands[s]) for s in slugs]
    # second chance for titles the label search could not pin down
    for k, r in enumerate(rows):
        if pd.isna(r.get("wikidata")):
            extra = by_enwiki(*TITLES[r["slug"]])
            if extra:
                rows[k] = resolve(r["slug"], *TITLES[r["slug"]], describe(extra), extra)
    df = pd.DataFrame(rows)
    unresolved = df[df["wikidata"].isna()]["slug"].tolist() if "wikidata" in df else slugs
    print(f"resolved {df['wikidata'].notna().sum()} / {len(df)}"
          + (f"; unresolved: {unresolved}" if unresolved else ""), flush=True)

    print("Box Office Mojo (worldwide, by IMDb id) ...", flush=True)
    gross, scope, source = [], [], []
    for _, r in df.iterrows():
        bom = from_box_office_mojo(r["imdb_id"]) if isinstance(r.get("imdb_id"), str) else None
        time.sleep(1.0)
        if bom:
            gross.append(bom["gross"]); scope.append(bom["scope"])
            source.append("boxofficemojo")
        elif pd.notna(r.get("wikidata_gross_usd")):
            gross.append(r["wikidata_gross_usd"])
            scope.append("worldwide (max of reported figures)"); source.append("wikidata")
        else:
            gross.append(None); scope.append(None); source.append(None)
    df["gross"], df["scope"], df["source"] = gross, scope, source

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, lineterminator="\n")
    ok = df["gross"].notna()
    print(f"\nwrote {OUT}: {ok.sum()} / {len(df)} with a gross "
          f"({(df['scope'] == 'worldwide').sum()} BOM worldwide, "
          f"{(df['source'] == 'wikidata').sum()} Wikidata, "
          f"{(df['scope'] == 'domestic only').sum()} domestic-only); "
          f"{df['budget_usd'].notna().sum()} with a budget")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[["slug", "wikidata_label", "year", "imdb_id", "gross", "scope",
                  "budget_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
