"""13단계(논문 v11 표 3, 3.3절, 그림 2의 검증 부분): SD 결합 모형의 판별 검증 — 단순 모형·기준 예측과 비교

검증 사례 (점포 수는 4월 1일 기준, 회차마다 계수를 추정 분포에서 뽑음)
  A_M1 / A_M0   2021 → 2025, 실제 진입 투입. 결합 모형 / 단순 모형(모든 점포 같은 평균 폐업률)
  A_FE          A_M1과 같되 위험식에 동 고정효과(임대료 수준은 동 단위라 제외)
  B_M2 / B_M0   2021 → 2025, 진입 내생(12단계 진입식, 2021년 전이까지 학습). 단순 = 평균 폐업률·동별 평균 진입(2017~2021)
  H_M2 / H_M0   2026 홀드아웃: 2025 코호트 → 2026, 위험식은 2024년 전이까지, 진입식은 2025년 전이까지 학습
  C_M1 / C_FE   2026 조건부: H와 같되 실제 2026 신규 점포를 넣음(동 고정효과 없음 / 있음)
지표  점포 수 오차, 2025 동별 점포 수 가중 절대오차, 동별 폐업률 상관(2022~2025 합산), 점포 단위 시간 밖 AUC(학습 2·3·4년)
기준  변화 없음(직전 점포 수 유지), 동별 폐업률은 동의 2016~2021년 폐업률

입력  data/yearly/cohort_*.csv, outputs/sd_feedback/dong_panel_april.csv(없으면 12단계가 만듦)
출력  outputs/sd_feedback/val_runs.csv     사례 × 회차 × 연도 점포 수·폐업·진입
      outputs/sd_feedback/val_dong.csv     사례 × 회차 × 연도 × 동 폐업·점포 수
      outputs/sd_feedback/val_auc.csv      학습 기간별 시간 밖 AUC
      outputs/sd_feedback/val_summary.csv  논문 표 3의 수치
      outputs/sd_feedback/fe_coef.csv      동 고정효과 위험식의 BIZ200·C200 계수
실행  python code/13_sd_validation.py      (NREP=100, 2코어 약 5분)
"""
import os, sys, importlib.util, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s = importlib.util.spec_from_file_location('fb', os.path.join(W, 'code', '12_sd_feedback_model.py')); fb = importlib.util.module_from_spec(_s); _s.loader.exec_module(fb)
sd, pai, OUT, D, P = fb.sd, fb.pai, fb.OUT, fb.DONGS, fb.PANEL
NREP = int(os.environ.get('NREP', 100)); CO = sd.CO

# ---------- 동 고정효과: 8단계 설계행렬에 동 더미를 더함(임대료 더미는 동 단위라 제외)
_design = sd.design; FE = {'on': False}
def design_fe(df, C, BIZ):
    X = _design(df, C, BIZ)
    if FE['on'] and 'dong' in df:
        X = X.drop(columns=['mid', 'high'], errors='ignore')
        for k in D[1:]: X['d_' + k] = (np.asarray(df['dong']) == k).astype(float)
    return X
sd.design = design_fe

def fit(years, fe=False):
    """점포-연 cloglog(8단계 sd.fit과 같은 추정, fe=True면 동 고정효과)"""
    FE['on'] = fe; rows = []
    for Y in years:
        S = pai.attrs(CO[Y]); C, B = sd.counts(S); X = sd.design(S, C, B); X['y'] = CO[Y].ev.values; X['id'] = S.id.values; rows.append(X)
    X = pd.concat(rows, ignore_index=True); y = X.pop('y'); g = pd.factorize(X.pop('id'))[0]
    keep = [c for c in X.columns if X[c].std() > 0 or c == 'const']
    m = sm.GLM(y, X[keep], family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster', cov_kwds={'groups': g})
    return m, keep

def rate(years): return float(np.mean([CO[y].ev.mean() for y in years]))
def meanE(y0, y1): return P[P.year.between(y0, y1)].groupby('dong').E.mean().to_dict()
Y4 = [2021, 2022, 2023, 2024]
CASES = {   # name: (start, steps, model, fe, entry, entry model, const_p, const_E)
    'A_M1': (2021, [2022, 2023, 2024, 2025], 'full', False, 'exo', None, None, None),
    'A_M0': (2021, [2022, 2023, 2024, 2025], 'full', False, 'exo', None, rate(sd.YEARS), None),
    'A_FE': (2021, [2022, 2023, 2024, 2025], 'full_fe', True, 'exo', None, None, None),
    'B_M2': (2021, [2022, 2023, 2024, 2025], 'full', False, 'endo', 2021, None, None),
    'B_M0': (2021, [2022, 2023, 2024, 2025], 'full', False, 'endo', 2021, rate(sd.YEARS), meanE(2017, 2021)),
    'H_M2': (2025, [2026], 'to2024', False, 'endo', 2025, None, None),
    'H_M0': (2025, [2026], 'to2024', False, 'endo', 2025, rate(Y4), meanE(2021, 2025)),
    'C_M1': (2025, [2026], 'to2024', False, 'exo', None, None, None),
    'C_FE': (2025, [2026], 'to2024_fe', True, 'exo', None, None, None),
}
M = {}
def init():
    fb.build_2026()
    M['full'] = fit(sd.YEARS); M['full_fe'] = fit(sd.YEARS, True); M['to2024'] = fit(Y4); M['to2024_fe'] = fit(Y4, True)

def job(name):
    start, steps, mk, fe, entry, emy, cp, cE = CASES[name]; m, keep = M[mk]; em = fb.entry_model(emy) if emy else None
    R, G = [], []
    for r in range(NREP):
        FE['on'] = fe
        beta = np.random.default_rng(r).multivariate_normal(m.params[keep].values, m.cov_params().loc[keep, keep].values)
        o, d = fb.run(start, steps, beta, keep, em, r, entry=entry, track_dong=True, const_p=cp, const_E=cE)
        o['rep'] = r; d['rep'] = r; o['case'] = name; d['case'] = name; R.append(o); G.append(d)
    return pd.concat(R), pd.concat(G)

# ---------- 실제 값(코호트 정의). 2026 점포 수 = 2025 생존 + 2026 신규(패널)
def actual():
    rows = []
    for Y in range(2022, 2027):
        p = CO[Y - 1]
        for k in D:
            m = p.dong.values == k
            st = (CO[Y].dong.values == k).sum() if Y <= 2025 else int((m & (p.ev.values == 0)).sum() + P[(P.year == 2026) & (P.dong == k)].E.sum())
            rows.append(dict(year=Y - 1, dong=k, clos=int((m & (p.ev.values == 1)).sum()), stock=int(m.sum()), stock_next=int(st)))
    return pd.DataFrame(rows)

def auc_table():
    """점포 단위 1년 폐업 판별력: 앞 2·3·4년 전이로 학습, 나머지 연도로 평가. 결합(전체 변수) vs 업력만"""
    Pn = {}
    for Y in sd.YEARS:
        S = sd.attrs(CO[Y]); C, B = sd.counts(S); X = sd.design(S, C, B); X['y'] = CO[Y].ev.values; Pn[Y] = X
    full = [c for c in Pn[2021].columns if c != 'y']; rows = []
    for k in (2, 3, 4):
        tr = pd.concat([Pn[y] for y in sd.YEARS[:k]]); te = pd.concat([Pn[y] for y in sd.YEARS[k:]])
        for name, cols in (('combined', full), ('age_only', ['const', 'log_age'])):
            cols = [c for c in cols if tr[c].std() > 0 or c == 'const']
            m = sm.GLM(tr.y, tr[cols], family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
            rows.append(dict(train=f'{sd.YEARS[0]}-{sd.YEARS[k - 1]}', test=f'{sd.YEARS[k]}-{sd.YEARS[-1]}', model=name, auc=roc_auc_score(te.y, m.predict(te[cols]))))
    return pd.DataFrame(rows)

def summarize(R, G, A, AUC):
    act = A.groupby('year').stock_next.sum(); act.index = act.index + 1; act[2021] = len(CO[2021])
    err = lambda c, ys: [100 * (R[(R.case == c) & (R.year == y)].stock_after.mean() / act[y] - 1) for y in ys]
    G = G.copy(); G['ay'] = G.year - 1
    m = G.groupby(['case', 'ay', 'dong']).agg(clos=('clos', 'mean'), stock=('stock', 'mean')).reset_index()
    def dong_corr(c):
        x = m[m.case == c].merge(A, left_on=['ay', 'dong'], right_on=['year', 'dong'], suffixes=('_s', '_a')); x = x[x.clos_s.notna()]
        g = x.groupby('dong')[['clos_s', 'stock_s', 'clos_a', 'stock_a']].sum(); return np.corrcoef(g.clos_s / g.stock_s, g.clos_a / g.stock_a)[0, 1]
    def dong_stock(c):
        x = m[(m.case == c) & (m.ay == 2025)].merge(A[A.year == 2024][['dong', 'stock_next']], on='dong')
        return 100 * (x.stock - x.stock_next).abs().sum() / x.stock_next.sum()
    pre = P[P.year.between(2016, 2021)].groupby('dong')[['D', 'S']].sum(); a = A[A.year.between(2021, 2024)].groupby('dong')[['clos', 'stock']].sum()
    st0 = A[A.year == 2021].set_index('dong').stock; st25 = A[A.year == 2024].set_index('dong').stock_next
    naive26 = 100 * (len(CO[2025]) / act[2026] - 1)
    rng = lambda v: f'{min(v):+.1f}~{max(v):+.1f}'
    ac = AUC[AUC.model == 'combined'].auc; aa = AUC[AUC.model == 'age_only'].auc
    rows = [
        ('Stores 2022~25, error (%)', rng(err('B_M2', range(2022, 2026))), rng(err('B_M0', range(2022, 2026))), '—'),
        ('Dong stores 2025, error (%)', f'{dong_stock("B_M2"):.1f}', f'{dong_stock("B_M0"):.1f}', f'{100 * (st0 - st25).abs().sum() / st25.sum():.1f}'),
        ('Dong closure rate, r', f'{dong_corr("A_M1"):.2f}', f'{dong_corr("A_M0"):.2f}', f'{np.corrcoef(prer := (pre.D / pre.S).reindex(a.index), a.clos / a.stock)[0, 1]:.2f}'),
        ('  with dong fixed effects', f'{dong_corr("A_FE"):.2f}', '—', '—'),
        ('Store AUC, out-of-time', f'{ac.min():.2f}~{ac.max():.2f}', f'{aa.min():.2f}~{aa.max():.2f}', '0.50'),
        ('Stores 2026 (hold-out), error (%)', f'{err("H_M2", [2026])[0]:+.1f}', f'{err("H_M0", [2026])[0]:+.1f}', f'{naive26:+.1f}'),
        ('  with actual 2026 entry', f'{err("C_M1", [2026])[0]:+.1f}', f'{100 * ((len(CO[2025]) * (1 - rate(Y4)) + P[P.year == 2026].E.sum()) / act[2026] - 1):+.1f}', f'{naive26:+.1f}'),
        ('(exogenous entry) Stores 2022~25, error (%)', rng(err('A_M1', range(2022, 2026))), rng(err('A_M0', range(2022, 2026))), '—'),
    ]
    return pd.DataFrame(rows, columns=['test', 'combined', 'simple', 'naive'])

if __name__ == '__main__':
    init()
    fc = []
    for k in ('full', 'full_fe'):
        m, _ = M[k]; fc += [dict(model=k, var=v, hr=np.exp(m.params[v]), p=m.pvalues[v]) for v in ('BIZ200', 'C200')]
    fc = pd.DataFrame(fc); fc.to_csv(os.path.join(OUT, 'fe_coef.csv'), index=False, encoding='utf-8-sig'); print(fc.round(4).to_string(index=False))
    with Pool(2, initializer=init) as pool: res = pool.map(job, list(CASES))
    R = pd.concat([r[0] for r in res]); G = pd.concat([r[1] for r in res]); A = actual(); FE['on'] = False; AUC = auc_table()
    R.to_csv(os.path.join(OUT, 'val_runs.csv'), index=False, encoding='utf-8-sig'); G.to_csv(os.path.join(OUT, 'val_dong.csv'), index=False, encoding='utf-8-sig')
    AUC.to_csv(os.path.join(OUT, 'val_auc.csv'), index=False, encoding='utf-8-sig')
    Sm = summarize(R, G, A, AUC); Sm.to_csv(os.path.join(OUT, 'val_summary.csv'), index=False, encoding='utf-8-sig')
    print(AUC.round(3).to_string(index=False)); print(Sm.to_string(index=False))
    h = R[R.case.isin(['H_M2', 'C_M1'])].groupby('case').agg(stock=('stock_after', 'mean'), lo=('stock_after', lambda s: s.quantile(.05)), hi=('stock_after', lambda s: s.quantile(.95)), clos=('closures', 'mean'), ent=('entries', 'mean'))
    print('[2026] 실제 점포', int(A[A.year == 2025].stock_next.sum()), '폐업', int(A[A.year == 2025].clos.sum()), '진입', int(P[P.year == 2026].E.sum())); print(h.round(1).to_string())
