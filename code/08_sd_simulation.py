"""8단계: SD(점포 stock-flow) 시뮬레이션: 연도별 코호트 + 시간가변 폐업 위험(이산시간 Cox형) + 정보 활용률 시나리오
입력  data/yearly/cohort_{2021..2025}_end20261013.csv  (6단계 결과)
출력  outputs/yearly/sd_params.csv     폐업 확률 계수(점포-연 패널 cloglog)
      outputs/yearly/sd_backtest.csv   검증 시뮬레이션(회차 × 연도)
      outputs/yearly/sd_scenarios.csv  정보 활용률 시나리오(α × 회차 × 연도)
실행  python code/08_sd_simulation.py all   (all | backtest | scenario, 반복 수 NREP=200, 후보지 수 KCAND=5; 검증 약 10분, 시나리오 30분 이상)
      콘솔 출력을 outputs/yearly/sd_run_log.txt로 저장해 두었다

모형 (연 단위, 기준일 4월 1일마다 한 걸음)
  S(t+1) = S(t) − D(t) + E(t)                         점포 수 stock, 폐업 flow D, 신규 flow E
  D(t)   = Σ_i p_i(t),  p_i(t) = 1 − exp(−exp(η_i(t)))  점포별 1년 폐업 확률(cloglog = 비례위험의 이산시간 형태)
  η_i(t) = a + β1·ln(1+업력) + β2·ln매출 + β3·매출결측 + 임대료수준 + 업태 + β4·C200(t)/10 + β5·BIZ200(t)/10
           C200, BIZ200 = 그 시점 영업 중 점포로 다시 센 반경 200m 동종 수, 전체 음식점 수  →  신규·폐업이 밀도를 바꾸고, 밀도가 다음 해 폐업 확률을 바꿈(피드백)
  E(t)   = 실제로 다음 기준일에 영업 중이던 신규 점포(외생). 시나리오에서는 정보를 받은 비율 α의 신규 점포가 입지를 바꿈
정보 시나리오 S1(α): 정보를 받은 신규 사업자는 무작위 후보지 k곳과 원래 입지의 예상 폐업 확률을 비교해 더 낮은 곳을 택함. α=0이 S0
검증  (1) 전체 기간 계수로 2021 기준 → 2025까지 점포 수 재현  (2) 2021~2022 전이로만 추정 → 2023 기준에서 2025까지 예측(시간 분할)
주의  계수는 이 데이터에서 추정한 값이며, 입지 선택 규칙(k, 선택 기준)과 α는 시나리오 가정이다
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.spatial import cKDTree
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(W, 'data', 'yearly'); OUT = os.path.join(W, 'outputs', 'yearly'); os.makedirs(OUT, exist_ok=True); TAG = '20261013'
YEARS = [2021, 2022, 2023, 2024, 2025]; RAD = 200
NREP = int(os.environ.get('NREP', 200)); K = int(os.environ.get('KCAND', 5))

# ---------- 데이터
tr = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True)
CO = {}
for Y in YEARS:
    d = pd.read_csv(os.path.join(DATA, f'cohort_{Y}_end{TAG}.csv'), parse_dates=['closure_date', 'license_date'])
    d['x'], d['y'] = tr.transform(d.longitude.values, d.latitude.values)
    d['cat'] = d.category_name.fillna('미분류'); d['code'] = d.category_code.fillna('미분류')
    d['ev'] = (d.closure_date.notna() & (d.closure_date < pd.Timestamp(f'{Y + 1}-04-01'))).astype(int)   # 다음 기준일 전 폐업
    CO[Y] = d
CATS = sorted(set().union(*[set(d.cat) for d in CO.values()]) - {'한식음식점'})
lsall = np.log(pd.concat([d.area_sales_per_store for d in CO.values()])).dropna(); LSMED = lsall.median()

def attrs(d):
    """점포 상태 표(시뮬레이션용): 필요한 열만"""
    ls = np.log(d.area_sales_per_store)
    return pd.DataFrame({'id': d.restaurant_id.values, 'x': d.x.values, 'y': d.y.values, 'code': d.code.values, 'cat': d.cat.values,
                         'age': d.business_age_years.values, 'ls': ls.fillna(LSMED).values, 'sm': ls.isna().astype(int).values,
                         'mid': (d.rent_level == 'mid').astype(int).values, 'high': (d.rent_level == 'high').astype(int).values,
                         'ent': 0})

def counts(S, pts=None, codes=None):
    """S의 영업 점포 기준 RAD 안 동종 수 C, 전체 수 BIZ. pts가 있으면 그 지점(자기 자신 없음), 없으면 S 점포 각각(자기 제외)"""
    P = S[['x', 'y']].values; tree = cKDTree(P)
    q = P if pts is None else pts; cd = S.code.values if codes is None else codes
    biz = tree.query_ball_point(q, RAD, return_length=True) - (1 if pts is None else 0)
    c = np.zeros(len(q), int)
    for k in np.unique(cd):
        m = cd == k; sel = S.code.values == k
        if sel.any(): c[m] = cKDTree(P[sel]).query_ball_point(q[m], RAD, return_length=True) - (1 if pts is None else 0)
    return c, biz

def design(df, C, BIZ):
    X = pd.DataFrame({'log_age': np.log1p(df.age.values), 'ls': df.ls.values, 'sm': df.sm.values, 'mid': df.mid.values, 'high': df.high.values}, index=df.index)
    for c in CATS: X['c_' + c] = (df.cat.values == c).astype(float)
    X['C200'] = np.asarray(C) / 10; X['BIZ200'] = np.asarray(BIZ) / 10
    return sm.add_constant(X, has_constant='add').astype(float)

# ---------- 추정 (점포-연 패널, 점포 단위 군집 SE)
def fit(years):
    rows = []
    for Y in years:
        S = attrs(CO[Y]); C, B = counts(S); X = design(S, C, B); X['y'] = CO[Y].ev.values; X['id'] = S.id.values; rows.append(X)
    P = pd.concat(rows, ignore_index=True)
    y = P.pop('y'); g = pd.factorize(P.pop('id'))[0]
    keep = [c for c in P.columns if P[c].std() > 0 or c == 'const']
    m = sm.GLM(y, P[keep], family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster', cov_kwds={'groups': g})
    return m, keep, int(y.sum()), len(y)

def hazard(beta, keep, df, C, BIZ):
    X = design(df, C, BIZ).reindex(columns=keep, fill_value=0.0)
    return 1 - np.exp(-np.exp(X.values @ beta))

# ---------- 시뮬레이션
ALLID = {i: k for k, i in enumerate(pd.concat([d.restaurant_id for d in CO.values()]).unique())}

def entrants(Y):
    """Y 기준일에 처음 나타난 점포(전 기준일 코호트에 없던 점포)"""
    new = CO[Y][~CO[Y].restaurant_id.isin(set(CO[Y - 1].restaurant_id))]
    return attrs(new).assign(ent=1)

def run(start, beta, keep, alpha, rng_u, rng_r, pool, steps):
    """start: 시작 연도 코호트. steps: 이후 기준 연도 리스트. rng_u: (id×step) 균등난수(공통 난수), 반환: 연도별 stock·폐업·신규 통계"""
    S = attrs(CO[start]); out = []
    for s, Y in enumerate(steps):
        C, B = counts(S); p = hazard(beta, keep, S, C, B)
        u = rng_u[S.id.map(ALLID).values, s]; closed = u < p
        out.append(dict(year=Y, stock_before=len(S), closures=int(closed.sum()), closures_ent=int(closed[S.ent.values == 1].sum()),
                        n_ent=int((S.ent.values == 1).sum()), ent_biz=float(B[S.ent.values == 1].mean()) if (S.ent.values == 1).any() else np.nan,
                        ent_p=float(p[S.ent.values == 1].mean()) if (S.ent.values == 1).any() else np.nan, mean_p=float(p.mean())))
        T = S[~closed].copy(); T['age'] += 1; T['ent'] = np.where(T.ent.values == 1, 2, 0)      # 2 = 이전 연도 신규(구분용)
        E = entrants(Y)
        if alpha > 0 and len(E):
            info = rng_r.random(len(E)) < alpha
            ET = E.copy(); Ct, Bt = counts(T, E[['x', 'y']].values, E.code.values); p0 = hazard(beta, keep, E, Ct, Bt)
            for i in np.flatnonzero(info):
                cand = pool[Y].iloc[rng_r.choice(len(pool[Y]), K, replace=False)]
                cc, bb = counts(T, cand[['x', 'y']].values, np.repeat(E.code.values[i], K))
                Z = E.iloc[[i] * K].copy()
                for col in ('x', 'y', 'ls', 'sm', 'mid', 'high'): Z[col] = cand[col].values
                pc = hazard(beta, keep, Z, cc, bb); j = int(np.argmin(pc))
                if pc[j] < p0[i]:
                    for col in ('x', 'y', 'ls', 'sm', 'mid', 'high'): ET.iloc[i, ET.columns.get_loc(col)] = cand[col].values[j]
            E = ET
        S = pd.concat([T, E], ignore_index=True)
        out[-1]['stock_after'] = len(S)
    return pd.DataFrame(out)

def monte(start, model, alpha, steps, nrep=NREP, seed=0):
    m, keep, _, _ = model; cov = m.cov_params().loc[keep, keep].values; b0 = m.params[keep].values
    pool = {Y: attrs(CO[Y]) for Y in steps}; res = []
    for r in range(nrep):
        rs = np.random.default_rng(seed + r); beta = rs.multivariate_normal(b0, cov)
        u = rs.random((len(ALLID), len(steps))); rr = np.random.default_rng(10_000 + seed + r)
        o = run(start, beta, keep, alpha, u, rr, pool, steps); o['rep'] = r; o['alpha'] = alpha; res.append(o)
    return pd.concat(res, ignore_index=True)

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    actual = {Y: len(CO[Y]) for Y in YEARS}
    full = fit(YEARS); m, keep, ne, nobs = full
    pd.DataFrame({'coef': m.params, 'hr': np.exp(m.params), 'p': m.pvalues}).to_csv(os.path.join(OUT, 'sd_params.csv'), encoding='utf-8-sig')
    print(f'[추정] 점포-연 {nobs}건, 폐업 {ne}건 | C200 10곳당 HR {np.exp(m.params["C200"]):.3f} (p={m.pvalues["C200"]:.3f}), BIZ200 10곳당 HR {np.exp(m.params["BIZ200"]):.3f} (p={m.pvalues["BIZ200"]:.3f}), 업력(log) HR {np.exp(m.params["log_age"]):.3f}')
    if stage in ('all', 'backtest'):
        rows = []
        early = fit([2021, 2022]); print(f'[시간 분할 추정] 2021~2022 전이: 폐업 {early[2]}건 / {early[3]}건 | BIZ200 HR {np.exp(early[0].params["BIZ200"]):.3f}')
        for name, start, model, steps in (('전체기간 계수, 2021 시작', 2021, full, [2022, 2023, 2024, 2025]), ('2021~22 계수, 2023 시작(시간 분할)', 2023, early, [2024, 2025])):
            o = monte(start, model, 0.0, steps); o['name'] = name; rows.append(o)
            g = o.groupby('year').stock_after.agg(['mean', lambda s: s.quantile(.05), lambda s: s.quantile(.95)]); g.columns = ['sim', 'lo', 'hi']
            g['actual'] = [actual[y] for y in g.index]; g['err_%'] = (g.sim / g.actual - 1) * 100
            c = o.groupby('year').closures.mean()
            g['closures_sim'] = c.values
            g['closures_actual'] = [int(CO[y - 1].ev.sum()) for y in g.index]
            # 비교 기준: 밀도·업력 없이 학습 기간 평균 폐업률만 쓰는 단순 모형 (신규는 같은 실제 값)
            tr_years = YEARS[:YEARS.index(start)] if start > 2021 else YEARS
            rate = np.mean([CO[y].ev.mean() for y in (tr_years if start > 2021 else YEARS)]) if start == 2021 else np.mean([CO[y].ev.mean() for y in (2021, 2022)])
            s_b, bench = len(CO[start]), []
            for y in steps: s_b = s_b * (1 - rate) + len(entrants(y)); bench.append(s_b)
            g['단순_평균폐업률'] = bench; g['단순_err_%'] = (g['단순_평균폐업률'] / g.actual - 1) * 100
            print(f'\n[{name}] 점포 수(기준일 4/1, 시뮬레이션 {NREP}회)'); print(g.round(1).to_string())
        pd.concat(rows).to_csv(os.path.join(OUT, 'sd_backtest.csv'), index=False, encoding='utf-8-sig')
    if stage in ('all', 'scenario'):
        rows = [monte(2021, full, a, [2022, 2023, 2024, 2025]) for a in (0.0, 0.4, 0.6)]
        R = pd.concat(rows); R.to_csv(os.path.join(OUT, 'sd_scenarios.csv'), index=False, encoding='utf-8-sig')
        tot = R.groupby(['alpha', 'rep']).agg(clos=('closures', 'sum'), clos_ent=('closures_ent', 'sum'), stock=('stock_after', 'last'), entp=('ent_p', 'mean'), entb=('ent_biz', 'mean')).reset_index()
        base = tot[tot.alpha == 0].set_index('rep')
        print(f'\n[시나리오] 2021→2025 누적 (평균 [5%, 95%], {NREP}회)')
        for a, g in tot.groupby('alpha'):
            d = g.set_index('rep'); line = f'α={a:.1f}: 폐업 {g.clos.mean():.0f} [{g.clos.quantile(.05):.0f},{g.clos.quantile(.95):.0f}], 신규 점포 폐업 {g.clos_ent.mean():.0f}, 2025 점포 수 {g.stock.mean():.0f}, 신규 점포 평균 예상 폐업확률 {g.entp.mean():.4f}, 신규 점포 주변 음식점 수 {g.entb.mean():.1f}'
            if a > 0: dd = d.clos - base.clos; line += f' | S0 대비 폐업 {dd.mean():+.1f} [{dd.quantile(.05):+.1f},{dd.quantile(.95):+.1f}]'
            print(line)
