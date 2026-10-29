"""12단계(논문 v11 2.4절, 그림 1): 피드백 고리를 넓힌 Cox–SD 결합 모형 — 공실 재진입(내생 진입)

8단계 SD(점포별 1년 폐업 확률 → 점포 수 stock)에 진입 고리를 더한다. 13·14단계가 이 파일을 모듈로 불러 쓴다.

고리  R1 집적       : 폐업 → 200m 음식점 수(BIZ)↓ → 다음 해 폐업 확률↑ → 폐업          (8단계와 같음)
      B1 유출       : 점포 수↑ → 폐업 flow↑ → 점포 수↓
      B2 공실 재진입: 폐업 → 공실↑ → 신규 진입↑(공실 자리에 입점) → BIZ↑ → 폐업 확률↓
진입식  E_k(t) ~ Poisson(μ_k(t)),  ln μ_k(t) = a_k + τ + b·[ln(D_k(t)+1) − 동 평균]      (동 k, 그해 폐업 D_k)
        4월 기준 17개 동 패널(2016~2026 전이)에서 동·연도 고정효과 Poisson(동 군집 SE)으로 추정.
        τ = 최근 5개 학습 연도 효과의 평균, 연도 충격 ε ~ N(0, 그 표준편차). 매 회차 b를 추정 분포에서 뽑는다.
        신규 점포의 업종·입지 원형은 시작 시점 이전 신규 점포에서 뽑고, 그 동의 공실에 먼저 입점(공실이 모자라면 원형 입지).
정보 시나리오(예측)  정보를 받은 신규 점포는 아직 비어 있는 공실 5곳(구 전체)과 현재 입지 중 예상 폐업 확률이 가장 낮은 곳을 택한다
        (공실 하나에 한 점포 → 수용 한도). 자동화형(ρ, λ)은 10단계와 같다.
진입 entry='exo'이면 8·10단계처럼 실제 신규 점포를 넣는다(검증·조건부 예측용).

입력  data/raw/성동구_일반음식점_인허가_갱신_20261019.csv   4월 기준 동 패널, 2026-04-01 실제 점포(예측 출발점)
      data/yearly/cohort_{2021..2025}_end20261013.csv     (6단계 결과)
출력  outputs/sd_feedback/dong_panel_april.csv             동 × 연도(2016~2026) 기초 점포 S, 폐업 D, 신규 E, 다음 기준일 점포 S1
실행  python code/12_sd_feedback_model.py                  동 패널을 만들고 진입식 추정치를 출력(수 초)
"""
import os, re, sys, importlib.util, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from pyproj import Transformer
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(W, 'outputs', 'sd_feedback'); os.makedirs(OUT, exist_ok=True)
LICENSE = os.path.join(W, 'data', 'raw', '성동구_일반음식점_인허가_갱신_20261019.csv')

def _load(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(W, 'code', f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sd = _load('sd', '08_sd_simulation.py'); pai = _load('pai', '10_sd_physical_ai.py')
K = 5
_ext = lambda a: (lambda m: m.group(1) if m else None)(re.search(r'([가-힣]+(?:동\d*가|동|가))(?=\s|\d|-|$)', str(a)))   # 6단계와 같은 동 추출

def read_license():
    L = pd.read_csv(LICENSE, encoding='cp949', low_memory=False); L['restaurant_id'] = L['관리번호'].astype(str)
    L['open'] = pd.to_datetime(L['인허가일자'].astype(str).str.strip(), errors='coerce')
    L['close'] = pd.to_datetime(L['폐업일자'].astype(str).str.strip(), errors='coerce')
    L['dong'] = L['지번주소'].map(_ext)
    return L

# ---------- 4월 기준 동 패널 (6단계 코호트와 같은 규칙: 기준일에 인허가 ≤ 기준일, 폐업 없음 또는 폐업일 ≥ 기준일)
def build_panel():
    L = read_license(); dongs = sorted(sd.CO[2025].dong.dropna().unique()); L = L[L.dong.isin(dongs)]
    alive = lambda Y: (L.open <= pd.Timestamp(f'{Y}-04-01')) & (L.close.isna() | (L.close >= pd.Timestamp(f'{Y}-04-01')))
    rows = []
    for Y in range(2016, 2027):
        a0, a1 = alive(Y - 1), alive(Y); D = a0 & L.close.notna() & (L.close < pd.Timestamp(f'{Y}-04-01')); E = a1 & ~a0
        for k in dongs:
            m = L.dong == k; rows.append(dict(dong=k, year=Y, S=int((a0 & m).sum()), D=int((D & m).sum()), E=int((E & m).sum()), S1=int((a1 & m).sum())))
    P = pd.DataFrame(rows); P.to_csv(os.path.join(OUT, 'dong_panel_april.csv'), index=False, encoding='utf-8-sig')
    return P

_pp = os.path.join(OUT, 'dong_panel_april.csv')
PANEL = pd.read_csv(_pp) if os.path.exists(_pp) else build_panel()
DONGS = sorted(PANEL.dong.unique())

# ---------- 2026-04-01 실제 점포(예측 출발점): 2025 코호트 생존 점포 + 인허가 원자료의 2026 신규 점포
def build_2026():
    """sd.CO[2026]을 만든다. 좌표 없는 신규 점포는 같은 동 2025 코호트 점포 위치 하나, 임대료 수준은 동 최빈값"""
    if 2026 in sd.CO: return
    L = read_license(); b = pd.Timestamp('2026-04-01'); c25 = sd.CO[2025]
    surv = c25[c25.ev == 0].copy(); surv['business_age_years'] += 1
    new = L[(L.open > pd.Timestamp('2025-04-01')) & (L.open <= b) & (L.close.isna() | (L.close >= b)) & ~L.restaurant_id.isin(set(c25.restaurant_id))].copy()
    allc = pd.concat(sd.CO.values())
    mp = allc[['restaurant_id', 'category_code', 'category_name']].merge(L[['restaurant_id', '업태구분명']], on='restaurant_id').dropna(subset=['category_code']).groupby('업태구분명')[['category_code', 'category_name']].first()
    new['category_code'] = new['업태구분명'].map(mp.category_code); new['category_name'] = new['업태구분명'].map(mp.category_name)
    new = new[new.dong.isin(set(c25.dong))]
    x = pd.to_numeric(new['좌표정보(X)'], errors='coerce'); y = pd.to_numeric(new['좌표정보(Y)'], errors='coerce')
    lo, la = Transformer.from_crs('EPSG:5174', 'EPSG:4326', always_xy=True).transform(x.fillna(0).values, y.fillna(0).values)
    new['longitude'] = np.where(x.notna(), lo, np.nan); new['latitude'] = np.where(x.notna(), la, np.nan)
    rng = np.random.default_rng(7)
    for i in np.flatnonzero(new.longitude.isna().values):
        pool = c25[c25.dong == new.dong.values[i]]; j = rng.integers(len(pool))
        new.iloc[i, new.columns.get_loc('longitude')] = pool.longitude.values[j]; new.iloc[i, new.columns.get_loc('latitude')] = pool.latitude.values[j]
    new['rent_level'] = new.dong.map(c25.groupby('dong').rent_level.agg(lambda s: s.mode().iloc[0]))
    new['business_age_years'] = (b - new.open).dt.days / 365.25; new['area_sales_per_store'] = np.nan
    new['x'], new['y'] = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True).transform(new.longitude.values, new.latitude.values)
    new['cat'] = new.category_name.fillna('미분류'); new['code'] = new.category_code.fillna('미분류'); new['ev'] = 0
    new = new.reindex(columns=list(surv.columns))
    sd.CO[2026] = pd.concat([surv, new], ignore_index=True)
    D2 = pd.concat([pai.DONG, pd.Series(new.dong.values, index=new.restaurant_id.values)]); pai.DONG = D2[~D2.index.duplicated()]

# ---------- 진입식
def entry_model(ymax, w=5):
    """ymax 전이(그해 4월 기준)까지로 학습. 반환: 동별 수준 a(τ 포함), 동 평균 c, 탄력성 b·SE, 연도 충격 sig"""
    x = PANEL[PANEL.year <= ymax].copy(); x['lD'] = np.log(x.D + 1); c = x.groupby('dong').lD.mean(); x['lDc'] = x.lD - x.dong.map(c)
    m = smf.glm('E ~ lDc + C(dong) + C(year)', x, family=sm.families.Poisson()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.dong)[0]})
    yrs = sorted(x.year.unique()); yfe = np.array([0.0] + [m.params[f'C(year)[T.{y}]'] for y in yrs[1:]])
    tau = yfe[-w:].mean(); sig = np.std(yfe[-w:], ddof=1)
    a = {k: m.params['Intercept'] + m.params.get(f'C(dong)[T.{k}]', 0.0) + tau for k in DONGS}
    return dict(a=a, c=c.to_dict(), b=m.params['lDc'], bse=m.bse['lDc'], sig=sig, ymax=ymax, disp=m.pearson_chi2 / m.df_resid, p=m.pvalues['lDc'])

# ---------- 신규 점포 원형(업종·입지 표본): 시작 시점 이전에 관측된 신규 점포
def templates(start):
    start = min(start, 2025)
    if start == 2021:
        d = sd.CO[2021]; T = pai.attrs(d[d.business_age_years < 2])
    else:
        T = pd.concat([pai.attrs(sd.CO[Y][~sd.CO[Y].restaurant_id.isin(set(sd.CO[Y - 1].restaurant_id))]) for Y in range(2022, start + 1)], ignore_index=True)
    T['ent'] = 1
    return {k: g.reset_index(drop=True) for k, g in T.groupby('dong')}, T

SITE = ['x', 'y', 'mid', 'high', 'dong', 'ls', 'sm']

def run(start, steps, beta, keep, em, seed, entry='endo', alpha=0.0, rho=0.0, lam=0.5, track_dong=False, const_p=None, const_E=None):
    """start 코호트에서 steps 기준일까지 한 해씩. em: entry_model 결과(entry='exo'이면 None 가능).
    const_p / const_E: 단순 모형(모든 점포 같은 폐업 확률 / 동별 평균 진입). 같은 seed면 시나리오 간 공통 난수."""
    U = np.random.default_rng(seed).random((20000, len(steps)))          # 폐업 공통 난수(점포 uid × 연도)
    rng = np.random.default_rng(10_000 + seed); ra = np.random.default_rng(20_000 + seed); ri = np.random.default_rng(30_000 + seed)
    S = pai.attrs(sd.CO[start]); S['orig'] = 1; S['uid'] = np.arange(len(S))
    TPL, TALL = templates(start); nid = len(S); out = []; dcl = []
    b = rng.normal(em['b'], em['bse']) if em else 0.0
    for s, Y in enumerate(steps):
        C, B = sd.counts(S); A = pai.auto_near(S) if rho > 0 else 0; Bs = B - lam * A
        p = sd.hazard(beta, keep, S, C, Bs) if const_p is None else np.full(len(S), const_p); closed = U[S.uid.values, s] < p
        dense = B >= pai.BIZ_HI
        rec = dict(year=Y, stock_before=len(S), closures=int(closed.sum()), closures_dense=int((closed & dense).sum()), mean_p=float(p.mean()))
        if track_dong:
            for k in DONGS: dcl.append(dict(year=Y, dong=k, clos=int((closed & (S.dong.values == k)).sum()), stock=int((S.dong.values == k).sum())))
        vac = S.loc[closed, SITE].reset_index(drop=True)              # 공실(그해 폐업 점포 자리)
        T = S[~closed].copy(); T['age'] += 1; T['ent'] = np.where(T.ent.values == 1, 2, 0)
        if entry == 'exo':
            E = pai.attrs(sd.CO[Y][~sd.CO[Y].restaurant_id.isin(set(sd.CO[Y - 1].restaurant_id))]); E['ent'] = 1
        else:
            Dk = vac.dong.value_counts().reindex(DONGS, fill_value=0)
            eps = rng.normal(0, em['sig'])
            mu = np.array([np.exp(em['a'][k] + b * (np.log(Dk[k] + 1) - em['c'][k]) + eps) for k in DONGS]) if const_E is None else np.array([const_E[k] for k in DONGS])
            nk = rng.poisson(mu)
            parts = []; used = np.zeros(len(vac), bool)
            for k, n in zip(DONGS, nk):
                if n == 0: continue
                pool = TPL.get(k, TALL); Z = pool.iloc[rng.integers(0, len(pool), n)].copy().reset_index(drop=True)
                vk = np.flatnonzero((vac.dong.values == k) & ~used); rng.shuffle(vk); m = min(n, len(vk))
                Z['vac'] = -1
                if m:
                    for col in SITE: Z.loc[:m - 1, col] = vac[col].values[vk[:m]]
                    used[vk[:m]] = True; Z.loc[:m - 1, 'vac'] = vk[:m]
                parts.append(Z)
            E = pd.concat(parts, ignore_index=True) if parts else TALL.iloc[:0].copy()
            E['age'] = rng.random(len(E)); E['ent'] = 1; E['orig'] = 0; E['auto'] = 0
        E['uid'] = np.arange(nid, nid + len(E)); nid += len(E)
        if rho > 0 and len(E):
            wv = E.cat.map(pai.W_CAT).fillna(1.0).values; E['auto'] = (ra.random(len(E)) < np.minimum(1.0, rho * wv / wv.mean())).astype(int)
        if alpha > 0 and len(E) and entry == 'endo':
            info = np.flatnonzero(ri.random(len(E)) < alpha)
            Ct, Bt = sd.counts(T, E[['x', 'y']].values, E.code.values); p0 = sd.hazard(beta, keep, E, Ct, Bt - lam * (pai.auto_near(T, E[['x', 'y']].values) if rho > 0 else 0))
            for i in info:
                free = np.flatnonzero(~used)
                if len(free) == 0: break
                ci = ri.choice(free, min(K, len(free)), replace=False); cand = vac.iloc[ci]
                cc, bb = sd.counts(T, cand[['x', 'y']].values, np.repeat(E.code.values[i], len(ci)))
                if rho > 0: bb = bb - lam * pai.auto_near(T, cand[['x', 'y']].values)
                Z = E.iloc[[i] * len(ci)].copy()
                for col in SITE: Z[col] = cand[col].values
                pc = sd.hazard(beta, keep, Z, cc, bb); j = int(np.argmin(pc))
                if pc[j] < p0[i]:
                    old = int(E.vac.values[i])
                    if old >= 0: used[old] = False
                    used[ci[j]] = True
                    for col in SITE: E.iat[i, E.columns.get_loc(col)] = cand[col].values[j]
                    E.iat[i, E.columns.get_loc('vac')] = ci[j]
        rec['entries'] = len(E); rec['entries_dense'] = int((sd.counts(T, E[['x', 'y']].values, E.code.values)[1] >= pai.BIZ_HI).sum()) if len(E) else 0
        S = pd.concat([T, E.drop(columns=['vac'], errors='ignore')], ignore_index=True)
        rec['stock_after'] = len(S); rec['n_auto'] = int(S.auto.sum()); out.append(rec)
    if track_dong:
        for k in DONGS: dcl.append(dict(year=steps[-1] + 1, dong=k, clos=np.nan, stock=int((S.dong.values == k).sum())))
    return pd.DataFrame(out), pd.DataFrame(dcl)

def draw_beta(model, rs):
    m, keep = model[0], model[1]
    return rs.multivariate_normal(m.params[keep].values, m.cov_params().loc[keep, keep].values), keep

if __name__ == '__main__':
    P = build_panel()
    print('[동 패널] 4월 기준 합계'); print(P.groupby('year')[['S', 'D', 'E', 'S1']].sum().to_string())
    for y in (2021, 2025, 2026):
        e = entry_model(y); print(f'[진입식, {y}년 전이까지 학습] b = {e["b"]:.3f} (SE {e["bse"]:.3f}, p = {e["p"]:.4f}), 연도 충격 σ = {e["sig"]:.3f}, 과산포 {e["disp"]:.2f}')
