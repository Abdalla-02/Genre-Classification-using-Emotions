# Clip / segment length in the related-work literature

Purpose: position the clip length used in this thesis against prior work on
audio-based movie-genre classification and music emotion recognition (MER), and provide
a citable justification for the ~10–31 s clips and the fixed 10.24 s AST analysis window.

Source: focused reads of the related-work PDFs (segment length, dataset, audio source,
and any stated justification extracted per paper).

## This thesis (for reference)

- **Eerola & Vuoskoski (2011) clips (our data):** variable length, **10.15–31.15 s**, mean **17.35 s**.
- **AST analysis window:** fixed **10.24 s** (1024 frames @ 16 kHz); truncates ~99 % of our clips, pads ~none.

## Movie-genre classification from audio (most relevant)

| Paper | Audio source | Segment length used | Fixed / variable | Justification stated |
|-------|--------------|---------------------|------------------|----------------------|
| Austin et al. (2010) | Film score, commercial MP3s (16 kHz mono) | middle **60 s** of each track | fixed | none |
| Ma et al. (2021), Blockbuster | Commercial soundtrack cues | **5 s** base segments → variable **cues** (~10–15 s); MIR 5 s texture-window (33 % overlap); VGGish 0.96 s frames → resampled to 1 Hz | fixed base / variable cue | 5 s window follows prior work; VGGish default |
| Mangolin et al. (2020) | Movie-trailer audio | **30 s** (middle of clip, zero-padded if shorter) for the CNN/spectrogram branch; whole trailer (30–300 s) for handcrafted MFCC/SSD | fixed / whole | standardize spectrogram width |
| Behrouzi et al. (2023) | Movie-trailer audio (LMTD) | **no fixed length** — whole trailer split into 5 equal time sections, mean MFCC+delta per section | variable / whole | reduce computational complexity |
| Sharma et al. (2021), "Unified Framework" | Movie-trailer audio | **5 s** non-overlapping chunks (50 ms window / 25 ms hop for frames) | fixed | a trailer contains several sound types → use small chunks |
| Bhattacharjee et al. (2024) | Movie-trailer audio (Moviescope) | 1 s feature segments → **30 s** classifier input; **30 s found empirically optimal** (accuracy rises to 30 s, then saturates) | fixed | empirically optimal |

## MER / music-genre context (excerpt-length norm)

| Source | Typical excerpt length |
|--------|------------------------|
| Kim et al. (2010), MER survey | **30 s** standard fixed clip; example datasets 20–30 s; ~1 s for dynamic/continuous labels |
| Kang & Herremans (2025), MER survey | 30 s–1 min typical; some full-song; **as short as 10 s**; 30 s is modal |
| Oramas et al. (2017), music genre tagging | 15–30 s audio previews |

## Synthesis

1. **30 s is the de-facto standard.** It is the modal excerpt length in MER (Kim 2010;
   Kang & Herremans 2025) and the standard input in audio movie-genre work; Bhattacharjee
   et al. (2024) show empirically that performance increases up to ~30 s and then
   saturates. Across all papers the full range of fixed segment lengths is **5 s–60 s**,
   with 30 s dominant.
2. **Segment length ≠ frame length.** Every method separates the *segment* used for
   classification (5–60 s) from the short-time *analysis frame* (0.96 s for VGGish,
   50/25 ms elsewhere). The AST 10.24 s figure is a segment/window, not a frame.
3. **Use of the track middle.** Several works deliberately take the *middle* of the
   track (Austin: middle 60 s; Mangolin: middle 30 s) to avoid intros/quiet lead-ins.
4. **Weak justification is the norm.** Most papers state the length procedurally without
   rationale; the exceptions are standardization (Mangolin), complexity (Behrouzi), and
   empirical optimality (Bhattacharjee).

### Where this thesis sits, and the honest caveat

Our clips (10–31 s, mean 17 s) fall **within** the literature range but at its **short
end**, and the effective AST input (**10.24 s**) is **below the 30 s field norm**. This is
dictated by the AST architecture (1024-frame window), not a free design choice. Given the
saturation-at-30 s result of Bhattacharjee et al. (2024), it is best framed as a
**deliberate, architecture-driven trade-off**, with a candidate limitation / future-work
note: pooling several ~10 s windows per clip to approach the 30 s norm.

**Citable sentence:** Segment lengths in prior audio-genre and MER work span 5–60 s with
30 s as the de-facto standard (Kim et al. 2010; Kang & Herremans 2025; Bhattacharjee et
al. 2024); this thesis's variable ~10–31 s clips fall within this range, and the
AST-imposed 10.24 s window sits at the lower but still-established end (cf. the 10 s
excerpts noted by Kang & Herremans 2025).

## References

- Austin, A., Moore II, E., Gupta, U., & Chordia, P. (2010). Characterization of Movie Genre Based on Music Score. *ICASSP 2010*, 421–424. IEEE.
- Ma, B., Greer, T., Knox, D., & Narayanan, S. (2021). A Computational Lens into How Music Characterizes Genre in Film. *PLOS ONE*, 16(4): e0249957.
- Mangolin, R. B., Pereira, R. M., Britto Jr., A. S., Silla Jr., C. N., Feltrim, V. D., Bertolini, D., & Costa, Y. M. G. (2020). A Multimodal Approach for Multi-Label Movie Genre Classification. (Preprint; later in *Multimedia Tools and Applications*.)
- Behrouzi, T., Toosi, R., & Akhaee, M. A. (2023). Multimodal movie genre classification using recurrent neural network. *Multimedia Tools and Applications*, 82(4), 5763–5784.
- Sharma, A., Jindal, M., Mittal, A., & Vishwakarma, D. K. (2021). A Unified Audio Analysis Framework For Movie Genre Classification Using Movie Trailers. *ESCI 2021*, 510–515. IEEE. DOI: 10.1109/ESCI50559.2021.9396892.
- Bhattacharjee, M., Prasanna, S. R. M., & Guha, P. (2024). Exploration of Speech and Music Information for Movie Genre Classification. *ACM TOMM*, 20(8), Art. 241.
- Kim, Y. E., Schmidt, E. M., Migneco, R., Morton, B. G., Richardson, P., Scott, J., Speck, J. A., & Turnbull, D. (2010). Music Emotion Recognition: A State of the Art Review. *ISMIR 2010*.
- Kang, J., & Herremans, D. (2025). Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges. *IEEE Transactions on Affective Computing*, 16(4).
- Oramas, S., Nieto, O., Barbieri, F., & Serra, X. (2017). Multi-label Music Genre Classification from Audio, Text, and Images using Deep Features. *ISMIR 2017*.
- Eerola, T., & Vuoskoski, J. K. (2011). A comparison of the discrete and dimensional models of emotion in music. *Psychology of Music*, 39(1), 18–49.

*Note: verify page numbers, venues, and the Mangolin/Bhattacharjee author lists against
the source PDFs before citing in the thesis.*
