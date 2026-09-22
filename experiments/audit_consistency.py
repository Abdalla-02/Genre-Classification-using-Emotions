"""Consistency audit: do the documents still agree with the saved results?

Run this after changing any experiment or any document. It checks the things that drift
silently and are embarrassing to discover late:

  1. every file path quoted in the documents (Markdown and the LaTeX snippets) resolves
     to a file that exists;
  2. every headline number quoted in docs/current_state.md matches results/*.json to
     three decimals -- this catches a table updated in one place but not another;
     2b does the same for the round-3 results across BOTH documents, and 2c for the
     thesis-ready snippets in docs/latex/, which are pasted into the submitted document;
  3. the zero-shot bootstrap and the zero-shot score table use the SAME feature-scaling
     regime (they once did not, which flattered the reported margin by 0.013);
  4. the dataset still cleans to 346 / 319 clips;
  5. all five embedding caches hold 360 clips;
  6. no file in the working tree has CRLF endings (the repo is LF);
  7. every section cross-reference in the docs points at a heading that exists;
  8. every results/*.json is listed in results/README.md and named in docs/README.md;
  9. every results/*.json still renders through show_results.py, which is shape-driven
     and therefore drops an unrecognised block silently rather than failing.

    python experiments/audit_consistency.py

Exits non-zero if anything fails, so it can be wired into a pre-commit hook.
"""
import io, json, re, sys, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
problems, notes = [], []

# ---------------------------------------------------------------- 1. paths --
DOCS = list(ROOT.glob('*.md')) + list(ROOT.rglob('docs/**/*.md')) + \
       list(ROOT.rglob('experiments/**/*.md')) + list(ROOT.rglob('results/*.md')) + \
       list(ROOT.rglob('docs/latex/*.tex'))
path_re = re.compile(r'`([A-Za-z0-9_./\\-]+\.(?:py|md|tex|json|bib|csv|npy|txt))`')
for d in DOCS:
    text = io.open(d, encoding='utf-8', newline='').read()
    for m in path_re.findall(text):
        if m.startswith(('http', 'word/')) or '<' in m:
            continue
        # historical / external references that intentionally do not resolve locally
        if m in {'set1_ast.npy', 'bib/library.bib', 'current_state.docx',
                 'film_genre_master_list.csv', 'mir_feature_names.csv',
                 'mean_ratings_set1_enriched.csv', 'thesis.tex'}:   # Overleaf-only files
            continue
        base = m.split('/')[-1].split(chr(92))[-1]
        if (ROOT / m).exists() or (d.parent / m).exists() or list(ROOT.rglob(base)):
            continue
        problems.append(f'[broken path] {d.relative_to(ROOT)} -> {m}')

# ------------------------------------------------- 2. numbers vs the JSONs --
R = ROOT / 'results'
def j(n): return json.loads((R / f'{n}.json').read_text(encoding='utf-8'))

expected = {}
sp = j('statistical_power')
for k, v in sp['subset5']['tuned']['arms'].items(): expected[f'sp5:{k}'] = v['mean']
for k, v in sp['full8']['arms'].items(): expected[f'sp8:{k}'] = v['mean']
bd = j('blockbuster_deep')['logreg']
for k, v in bd['arms'].items(): expected[f'bb:{k}'] = v['mean']
zs = j('zero_shot')
expected['zs:loo'] = zs['blockbuster_loo_ours']['macro_f1']
expected['zs:eerola6'] = zs['eerola_in_domain']['macro_f1']
for k, v in zs['zero_shot']['strict (source scaler)'].items():
    expected[f'zs:{k}'] = v['macro_f1']          # docs quote the conservative regime
for k, v in zs['zero_shot']['target-standardised'].items():
    expected[f'zsA:{k}'] = v['macro_f1']
w = j('waveform_vs_spectrogram')
for k, v in w['stage1'].items(): expected[f'r2:{k}'] = v['mean_r2']
for k, v in w['stage2']['arms'].items(): expected[f'wg:{k}'] = v['mean']

cs = io.open(ROOT / 'docs' / 'current_state.md', encoding='utf-8', newline='').read()
checks = [
    ('0.397', 'sp5:ground-truth emotion(11) [ceiling]'), ('0.394', 'sp5:VGGish -> PREDICTED emotion(11)'),
    ('0.350', 'sp5:VGGish-128 (direct)'), ('0.345', 'sp5:AST-768'),
    ('0.343', 'sp5:PCA-8(VGGish) [control]'), ('0.335', 'sp5:CLAP-512'),
    ('0.300', 'sp8:ground-truth emotion(11)'), ('0.257', 'sp8:AST-768'),
    ('0.621', 'bb:VGGish, instance majority voting'), ('0.616', 'bb:emotion(11), per-cue -> pooled'),
    ('0.593', 'bb:VGGish-128'), ('0.587', 'bb:PCA-8(VGGish) [control]'),
    ('0.585', 'bb:MIR-140 (full)'), ('0.557', 'bb:emotion(11), pooled -> per-film'),
    ('0.516', 'bb:MFCC-78'), ('0.620', 'zs:loo'), ('0.315', 'zs:eerola6'),
    ('0.511', 'zs:VGGish -> predicted emotion(11), per-cue'),
    ('0.482', 'zs:VGGish -> predicted emotion(11), film-level'),
    ('0.407', 'zs:VGGish-128 direct'), ('0.421', 'zs:PCA-8(VGGish) [control]'),
    ('0.323', 'r2:wav2vec2-768'), ('0.560', 'r2:AST-768'), ('0.561', 'r2:CLAP-512'),
    ('0.558', 'r2:VGGish-128'), ('0.490', 'r2:MIR-103'),
    ('0.376', 'wg:wav2vec2 -> predicted emotion(11)'), ('0.300', 'wg:wav2vec2-768'),
]
for quoted, key in checks:
    if key not in expected:
        problems.append(f'[missing key] {key}'); continue
    actual = f'{expected[key]:.3f}'
    if actual != quoted:
        problems.append(f'[number drift] briefing says {quoted} for {key}, JSON says {actual}')
    if quoted not in cs:
        notes.append(f'[not quoted] {quoted} ({key}) absent from current_state.md')

# ------------------------------- 2b. the round-3 results, in BOTH documents --
# The briefing and the technical log both quote these; whichever is edited alone drifts.
log = io.open(ROOT / 'docs' / 'README.md', encoding='utf-8', newline='').read()
ab, cv, bo = j('emotion_ablation'), j('cross_dataset_cv'), j('box_office')

def arm(block, name, field='mean'):
    return block['arms'][name][field]

round3 = [   # (value, decimals, sign, label)
    (arm(ab['leave_one_out'], 'all 8 emotions [reference]'), 3, '', 'ablation: all 8'),
    (arm(ab['subsets'], 'fear only'), 3, '', 'ablation: fear only'),
    (arm(ab['subsets'], 'valence + energy  (2-d circumplex)'), 3, '', 'ablation: valence+energy'),
    (arm(ab['leave_one_out'], '11 features (8 + derived) [headline model]'), 3, '', 'ablation: 11 features'),
    (arm(cv['eerola_to_blockbuster'], 'VGGish -> emotion (per cue) -> genre', 'macro_f1'), 3, '', 'E->B emotion'),
    (arm(cv['eerola_to_blockbuster'], 'VGGish-128 direct', 'macro_f1'), 3, '', 'E->B direct'),
    (arm(cv['blockbuster_to_eerola'], 'VGGish -> emotion (per cue) -> genre', 'macro_f1'), 3, '', 'B->E emotion'),
    (arm(cv['blockbuster_to_eerola'], 'VGGish-128 direct', 'macro_f1'), 3, '', 'B->E direct'),
    (cv['eerola_to_blockbuster']['emotion_vs_direct']['diff'], 3, '+', 'E->B emotion vs direct'),
    (cv['blockbuster_to_eerola']['emotion_vs_direct']['diff'], 3, '+', 'B->E emotion vs direct'),
    (bo['n_films'], 0, '', 'box office: n films'),
    (bo['A_f1_vs_gross']['rho'], 2, '+', 'box office: F1 vs gross'),
    (bo['C_genre_vs_gross']['Action']['p_mw'], 3, '', 'box office: Action gross p'),
    (bo['B_emotion_vs_gross']['anger']['rho'], 2, '+', 'box office: anger rho'),
]
for value, nd, sign, label in round3:
    quoted = f'{value:{sign}.{nd}f}'
    where = [n for n, t in (('current_state.md', cs), ('README.md', log)) if quoted in t]
    if not where:
        problems.append(f'[unquoted result] {label} = {quoted} appears in neither document')
    elif len(where) == 1:
        notes.append(f'{label} = {quoted} is quoted only in docs/{where[0]}')

# ------------------------------------- 2c. the thesis-ready LaTeX snippets --
# These are pasted straight into the thesis, so a number that drifts here is a number
# that drifts into the submitted document. Only the figures at real risk are pinned:
# the zero-shot table, whose two scaling regimes were confused once already.
tex_checks = {
    'cross_dataset_transfer.tex': [
        'zs:VGGish -> predicted emotion(11), per-cue',
        'zs:VGGish -> predicted emotion(11), film-level',
        'zs:VGGish-128 direct', 'zs:PCA-8(VGGish) [control]', 'zs:loo'],
    'statistical_power.tex': [
        'sp5:ground-truth emotion(11) [ceiling]', 'sp5:VGGish -> PREDICTED emotion(11)',
        'sp5:VGGish-128 (direct)', 'sp8:ground-truth emotion(11)', 'sp8:AST-768'],
    'waveform_vs_spectrogram.tex': ['r2:wav2vec2-768', 'wg:wav2vec2-768'],
}
for fname, keys in tex_checks.items():
    p = ROOT / 'docs' / 'latex' / fname
    if not p.exists():
        problems.append(f'[missing snippet] docs/latex/{fname}'); continue
    text = io.open(p, encoding='utf-8', newline='').read()
    for key in keys:
        quoted = f'{expected[key]:.3f}'
        # the tables use the leading-dot convention (.323), the prose writes 0.323
        if quoted not in text and quoted[1:] not in text:
            problems.append(f'[tex drift] docs/latex/{fname} does not quote {quoted} '
                            f'for {key} -- the snippet and the JSON disagree')

# ------------------------------------- 3. zero_shot: regimes must be consistent --
boot = zs.get('bootstrap', {})
if set(boot) != set(zs['zero_shot']):
    problems.append(f'[zs] bootstrap regimes {set(boot)} != score regimes {set(zs[chr(34)+"zero_shot"+chr(34)])}')
else:
    strict_boot = boot['strict (source scaler)']
    for label, quoted in [('VGGish -> predicted emotion(11), per-cue vs VGGish direct', 0.106),
                          ('predicted-emotion vs PCA-8 control', 0.092)]:
        got = round(strict_boot[label]['diff_mean'], 3)
        if abs(got - quoted) > 5e-4:
            problems.append(f'[zs bootstrap] docs quote {quoted} for {label}, JSON {got}')
        if f'{quoted:.3f}' not in cs:
            problems.append(f'[zs bootstrap] {quoted} not quoted in current_state.md')

# ------------------------------------------------------- 4. dataset counts --
from src import config
from src.features import load_set1, load_set1_shared, genre_matrix, shared_genre_matrix
df = load_set1(); Y = genre_matrix(df)
if len(df) != config.SET1_EXPECTED_CLIPS: problems.append('[data] set1 clip count')
dfs = load_set1_shared(); Ys = shared_genre_matrix(dfs)
if len(dfs) != 319: problems.append(f'[data] shared clip count {len(dfs)}')
notes.append(f'set1={len(df)} clips/{df.soundtrack.nunique()} films; '
             f'shared={len(dfs)} clips/{dfs.soundtrack.nunique()} films')

# ----------------------------------------------------- 5. embedding caches --
for name, d in [('ast', config.EMBEDDINGS_DIR), ('clap', config.CLAP_EMBEDDINGS_DIR),
                ('vggish', config.VGGISH_EMBEDDINGS_DIR), ('mir', config.MIR_EMBEDDINGS_DIR),
                ('wav2vec2', config.W2V_EMBEDDINGS_DIR)]:
    n = len(list((d / 'set1').glob('*.npy'))) if (d / 'set1').is_dir() else 0
    if n != 360: problems.append(f'[cache] {name}/set1 has {n} files, expected 360')

# --------------------------------------------------------- 6. line endings --
out = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                     capture_output=True, text=True).stdout.splitlines()
for line in out:
    f = line[3:].strip()
    p = ROOT / f
    if p.is_file() and p.suffix in {'.py', '.md', '.tex', '.json'}:
        if b'\r' in p.read_bytes(): problems.append(f'[CRLF] {f}')

# --------------------------------------------------------- 7. section refs --
for doc in (ROOT / 'docs' / 'README.md', ROOT / 'docs' / 'current_state.md'):
    text = io.open(doc, encoding='utf-8', newline='').read()
    have = set(re.findall(r'^#{2,3} (\d+[a-z]?)[.b]', text, re.M)) |            set(re.findall(r'^#{2,3} (\d+\.\d+)', text, re.M)) |            set(re.findall(r'^### (\d+[a-z])\.', text, re.M))
    bases = {h.split('.')[0].rstrip('abcdef') for h in have}
    for ref in set(re.findall(r'§(\d+(?:\.\d+)?[a-z]?)', text)):
        if ref.split('.')[0].rstrip('abcdef') not in bases:
            problems.append(f'[bad section ref] {doc.name} §{ref}')

# ------------------------------- 7b. the exported briefing copies are stale --
# docs/current_state.{docx,pdf} are exports of an older revision of current_state.md.
# They cannot be regenerated here (no pandoc), so this reports rather than fails -- but
# handing someone the PDF would hand them superseded numbers.
md = ROOT / 'docs' / 'current_state.md'
for ext in ('docx', 'pdf'):
    p = md.with_suffix('.' + ext)
    if p.exists() and p.stat().st_mtime < md.stat().st_mtime:
        notes.append(f'current_state.{ext} is older than current_state.md -- an obsolete '
                     f'export; the Markdown is authoritative')

# ------------------------------------- 8. every results file is documented --
readme = io.open(R / 'README.md', encoding='utf-8', newline='').read()
for f in sorted(R.glob('*.json')):
    if f.name not in readme:
        problems.append(f'[undocumented result] {f.name} is not listed in results/README.md')
    if f.name not in log:
        problems.append(f'[undocumented result] {f.name} is not named in docs/README.md')

# ---------------------------------- 9. every results file actually renders --
# show_results.py is shape-driven, so a new experiment that invents a shape is dropped
# silently rather than failing. This is what caught the zero-shot tables going missing.
sys.path.insert(0, str(ROOT / 'experiments'))
import show_results                                             # noqa: E402
for f in sorted(R.glob('*.json')):
    blocks = []
    show_results.walk(json.loads(f.read_text(encoding='utf-8')), '', blocks,
                      {len(v): v for v in json.loads(f.read_text(encoding='utf-8')).values()
                       if isinstance(v, list) and v and all(isinstance(x, str) for x in v)})
    tables = [b for b in blocks if b.startswith('**')]
    leaves = len(re.findall(r'": \{', f.read_text(encoding='utf-8')))
    if len(tables) < 2 and leaves > 4:
        problems.append(f'[unrendered result] show_results.py emits {len(tables)} table(s) '
                        f'for {f.name} -- a block shape it does not recognise?')
    if f.stem not in show_results.DESCRIPTIONS:
        problems.append(f'[undescribed result] {f.stem} has no entry in '
                        f'show_results.DESCRIPTIONS, so the listing shows a blank line')

print('=' * 70)
print('AUDIT')
print('=' * 70)
for n in notes: print('  note   ', n)
print()
if problems:
    for p_ in problems: print('  ISSUE  ', p_)
    print(f'\n{len(problems)} issue(s)')
    sys.exit(1)
else:
    print('  no issues found -- docs and results agree')
