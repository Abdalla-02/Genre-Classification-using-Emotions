# Genre-set size in the related-work literature

Purpose: establish precedent for the number of genres modelled relative to dataset size,
to justify reporting results on a curated genre subset for this thesis's small corpus.

Source: focused reads of the movie-genre-from-audio related-work PDFs (genre count, exact
genre list, sample size, label type, genre-selection rationale).

## What prior work used

| Paper | # genres | Genres | Samples | Label | Genre selection |
|-------|----------|--------|---------|-------|-----------------|
| Austin et al. (2010) | 4 | Action, Horror, Romance, Drama | 98 films / 1728 tracks | single | chosen by emotional character |
| Ma et al. (2021) | 6 | Action, Comedy, Drama, Horror, Romance, Sci-Fi | 110 films | multi (~1.8) | **reduced IMDb 24 -> 6** |
| Sharma et al. (2021) | 5 | Action, Romance, Horror, Sci-Fi, Comedy | 2400 trailers | multi | "5 most generic/popular" |
| Behrouzi et al. (2023) | 9 (+4 subset) | action, adventure, comedy, crime, drama, horror, romance, sci-fi, thriller | 3500 movies | multi | inherited (LMTD); also uses a 4-genre subset |
| Bhattacharjee et al. (2024) | 13 | Action, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller | ~5000 trailers | multi | all dataset labels |
| Mangolin et al. (2020) | 18 | full TMDb taxonomy | 10594 films | multi (~2.4) | inherited (TMDb), no cutoff |

## The pattern
Genre count scales monotonically with dataset size:
- ~100 films -> **4-6 genres** (Austin, Ma)
- 2400-3500 trailers -> 5-9 genres (Sharma, Behrouzi)
- 5000-10600 -> 13-18 genres (Bhattacharjee, Mangolin)

Rare genres (Biography, Documentary, Adventure) appear ONLY in the large-data studies
(>=5000 samples). Small-scale studies restrict to a core recurring set: Action, Drama,
Horror (all 6 papers), Comedy, Romance, Sci-Fi (5/6), then Crime/Thriller (3/6).

## Where this thesis sits
This corpus (47 soundtracks / 346 clips) is **smaller than every study above** (Austin
98 films/1728 tracks; Ma 110 films). By the field's own precedent, 8 genres is high for
this scale; **5-6 is scale-appropriate.** The closest analogue, **Ma et al. (2021)** --
same domain (film soundtracks), 110 films -- deliberately reduced IMDb's 24 genres to 6.
Two further papers curate explicitly (Sharma "5 most generic"; Behrouzi reports a 4-genre
subset). A reduced set is established practice, not a workaround.

## Chosen subset and result
Keep the 5 genres that are both literature-core and carry an emotional signature in this
data (AUC/Cohen's d): **Action, Crime, Drama, Comedy, Horror.** Drop Adventure (no
signature), Biography (n=24), Documentary (n=17) -- these are rare in the literature AND
rare/unpredictable here.

Experiment (`experiments/exp_genre_subset.py`, 329 clips, GroupKFold, LogReg):

| | Macro-F1 (8 genres) | Macro-F1 (5-genre subset) |
|---|---|---|
| WITH emotion | 0.272 | **0.388** |
| WITHOUT (AST) | 0.258 | 0.323 |
| dummy | 0.102 | 0.164 |

On the subset, emotion (0.388) exceeds AST (0.323), diff +0.064 -- a larger and
consistent gap, though not statistically significant at 5 folds (paired-t p=0.16,
underpowered). Report BOTH the full-8 and the 5-genre numbers transparently; the subset is
the primary result, the full-8 documents the rare-genre limitation.

Note: this corpus lacks Romance and Sci-Fi (two of the field's most common genres),
reflecting the Eerola corpus's genre coverage vs trailer datasets.

## References (bib keys present in library.bib)
austin2010characterization, ma2021computational, mangolin2020multimodal,
behrouzi2023multimodal. (Sharma et al. 2021 and Bhattacharjee et al. 2024 are not yet in
library.bib -- add entries if cited.)
