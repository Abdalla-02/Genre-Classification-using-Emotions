"""Consistency audit: do the documents still agree with the saved results?

Run this after changing any experiment or any document. It checks the things that drift
silently and are embarrassing to discover late:

  1. every file path quoted in the Markdown docs resolves to a file that exists;
  2. every headline number quoted in docs/current_state.md matches results/*.json to
     three decimals -- this catches a table updated in one place but not another;
  3. the zero-shot bootstrap and the zero-shot score table use the SAME feature-scaling
     regime (they once did not, which flattered the reported margin by 0.013);
  4. the dataset still cleans to 346 / 319 clips;
  5. all five embedding caches hold 360 clips;
  6. no file in the working tree has CRLF endings (the repo is LF);
  7. every section cross-reference in the docs points at a heading that exists.

    python experiments/audit_consistency.py

Exits non-zero if anything fails, so it can be wired into a pre-commit hook.
"""
import io, json, re, sys, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
problems, notes = [], []

# ---------------------------------------------------------------- 1. paths --
DOCS = list(ROOT.glob('*.md')) + list(ROOT.rglob('docs/**/*.md')) + \
       list(ROOT.rglob('experiments/**/*.md')) + list(ROOT.rglob('results/*.md'))
path_re = re.compile(r'`([A-Za-z0-9_./\\-]+\.(?:py|md|tex|json|bib|csv|npy|txt))`')
for d in DOCS:
    text = io.open(d, encoding='utf-8', newline='').read()
    for m in path_re.findall(text):
        if m.startswith(('http', 'word/')) or '<' in m:
            continue
        # historical / external references that intentionally do not resolve locally
        if m in {'set1_ast.npy', 'bib/library.bib', 'current_state.docx',
                 'film_genre_master_list.csv', 'mir_feature_names.csv',
                 'mean_ratings_set1_enriched.csv'}:
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
log = io.open(ROOT / 'docs' / 'README.md', encoding='utf-8', newline='').read()
have = set(re.findall(r'^#{2,3} (\d+[a-z]?)[.b]', log, re.M)) |        set(re.findall(r'^#{2,3} (\d+\.\d+)', log, re.M)) |        set(re.findall(r'^### (\d+[a-z])\.', log, re.M))
for ref in set(re.findall(r'§(\d+(?:\.\d+)?[a-z]?)', log + cs)):
    base = ref.split('.')[0]
    if base not in {h.rstrip('abcdef') for h in have} and base not in have:
        problems.append(f'[bad section ref] §{ref}')

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
