"""
10단계(논문 표 5, 그림 4): 피지컬 AI 대체 시나리오 — "빈자리를 누가 채우는가"

질문  2021~2025년에 실제로 들어온 신규 점포 중 비율 ρ가 자동화형(키오스크·무인 주문·포장·배달 중심)이었다면,
      일반 음식점의 폐업은 얼마나 달라졌을까? AI 입지 정보(정보 활용률 α)는 그 손실을 얼마나 상쇄할까?

모형  8단계 SD(code/08_sd_simulation.py)를 그대로 쓰고 두 곳만 바꾼다.
  (1) 신규 점포 표시: 매년 실제 신규 점포마다 자동화형 여부 uᵢ를 무작위로 정한다.
        P(uᵢ = 1) = min(1, ρ · w_업종(i) / 그해 신규 점포의 평균 w)
        w_업종 = 리뷰에서 업종별로 구한 비대면 주문·포장·배달 언급률 / 전체 언급률 (data/interim/category_automation_rates.csv)
        → 전체 신규 점포의 약 ρ가 자동화형이 되고, 언급률이 높은 업종(패스트푸드·분식 등)이 더 자주 뽑힌다.
        리뷰는 점포 점수가 아니라 업종 성향으로만 쓰므로 2026년 리뷰를 과거 시뮬레이션에 넣어도 시점 문제가 없다.
  (2) 유효 집적: 자동화형 점포는 주변에 주던 유동인구 효과를 λ만큼 잃는다.
        BIZ*ᵢ(t) = BIZ₂₀₀ᵢ(t) − λ · (반경 200m 안 자동화형 점포 수)
        ηᵢ(t) = a + θ'Zᵢ + β_C·C₂₀₀ᵢ(t)/10 + β_B·BIZ*ᵢ(t)/10      (계수는 8단계와 같은 점포-연 추정치)
      동종 수 C는 그대로 센다(자동화형도 같은 업종 손님을 나눠 가짐). 자동화형 점포 자신의 폐업 확률 식은 같다.
  정보 활용률 α: 8단계와 같다. 정보를 받은 신규 점포는 후보지 5곳과 원래 입지 중 예상 폐업 확률이 가장 낮은 곳을 택하며,
      이때 예상 확률도 BIZ*로 계산한다(AI 플랫폼이 자동화형 점포 분포까지 알려 준다고 가정).
  ρ, λ, α와 업종 가중치 방식은 추정값이 아닌 시나리오 가정이다. ρ = 0이면 8단계 결과와 같다(공통 난수).

입력  data/yearly/cohort_*.csv, data/interim/category_automation_rates.csv
출력  outputs/physical_ai/pai_runs.csv          조합 × 회차 × 연도별 폐업·점포 수 등
      outputs/physical_ai/pai_dong.csv          조합 × 회차별 동 단위 누적 폐업
      outputs/physical_ai/pai_summary.csv       조합별 요약(평균, 5~95%, 기준 대비 차이)
실행  python code/10_sd_physical_ai.py                 기본 8개 조합, 200회 (2코어 병렬, 1~2시간)
      NREP=20 python code/10_sd_physical_ai.py         빠른 확인
      python code/10_sd_physical_ai.py check            ρ = 0 결과가 8단계와 같은지 확인(NREP=3)
      python code/10_sd_physical_ai.py summary          저장된 결과로 요약표만 다시 생성
      python code/10_sd_physical_ai.py fig              요약표로 그림 생성 → outputs/physical_ai/fig_physical_ai.png
"""
import os, sys, importlib.util, warnings
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('sd', os.path.join(W, 'code', '08_sd_simulation.py'))
sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
OUT = os.path.join(W, 'outputs', 'physical_ai'); os.makedirs(OUT, exist_ok=True)
NREP = int(os.environ.get('NREP', 200)); K = sd.K; RAD = sd.RAD
STEPS = [2022, 2023, 2024, 2025]
# (α, ρ, λ): 기본 λ = 0.5, 민감도 λ = 1.0
COMBOS = [(a, r, 0.5) for a in (0.0, 0.4) for r in (0.0, 0.2, 0.4)] + [(0.0, 0.2, 1.0), (0.0, 0.4, 1.0)]

W_CAT = pd.read_csv(os.path.join(W, 'data', 'interim', 'category_automation_rates.csv')).set_index('category_name').weight
DONG = pd.concat([d[['restaurant_id', 'dong']] for d in sd.CO.values()]).drop_duplicates('restaurant_id').set_index('restaurant_id').dong
BIZ_HI = np.quantile(sd.counts(sd.attrs(sd.CO[2021]))[1], 2 / 3)        # 2021년 200m 음식점 수 상위 1/3 경계 = 밀집지

def attrs(d):
    a = sd.attrs(d); a['dong'] = a.id.map(DONG).values; a['auto'] = 0; a['orig'] = 0; return a

def auto_near(S, pts=None):
    """S의 자동화형 점포 중 반경 RAD 안 개수(pts 없으면 S 각 점포 기준, 자기 자신 제외)"""
    q = S[['x', 'y']].values if pts is None else pts
    A = S.auto.values == 1
    if not A.any(): return np.zeros(len(q))
    n = cKDTree(S[['x', 'y']].values[A]).query_ball_point(q, RAD, return_length=True)
    return n - (A.astype(int) if pts is None else 0)

def run(beta, keep, alpha, rho, lam, rng_u, rng_r, rng_a, pool):
    S = attrs(sd.CO[2021]); S['orig'] = 1; out = []; dong_cl = {}       # orig = 2021년에 이미 있던 점포(모든 시나리오에서 같은 집합)
    for s, Y in enumerate(STEPS):
        C, B = sd.counts(S); Bs = B - lam * auto_near(S); p = sd.hazard(beta, keep, S, C, Bs)
        u = rng_u[S.id.map(sd.ALLID).values, s]; closed = u < p
        conv = S.auto.values == 0; ent = S.ent.values == 1
        for dg, n in pd.Series(S.dong.values[closed]).value_counts().items(): dong_cl[dg] = dong_cl.get(dg, 0) + int(n)
        orig = S.orig.values == 1; dense = B >= BIZ_HI
        out.append(dict(year=Y, stock_before=len(S), n_auto=int((~conv).sum()), closures=int(closed.sum()),
                        closures_dense=int((closed & dense).sum()), closures_sparse=int((closed & ~dense).sum()),
                        closures_orig=int((closed & orig).sum()), closures_conv=int((closed & conv).sum()), closures_ent=int((closed & ent).sum()),
                        n_dense=int(dense.sum()), mean_biz=float(B.mean()), mean_biz_eff=float(Bs.mean()), mean_p=float(p.mean()),
                        ent_biz=float(B[ent].mean()) if ent.any() else np.nan))
        T = S[~closed].copy(); T['age'] += 1; T['ent'] = np.where(T.ent.values == 1, 2, 0)
        E = attrs(sd.CO[Y][~sd.CO[Y].restaurant_id.isin(set(sd.CO[Y - 1].restaurant_id))]); E['ent'] = 1
        # (1) 자동화형 표시 — α 추첨(rng_r)과 따로 뽑아 ρ와 무관하게 같은 회차의 α 추첨이 유지되도록 함
        wv = E.cat.map(W_CAT).fillna(1.0).values
        pa = np.minimum(1.0, rho * wv / wv.mean()) if rho > 0 else np.zeros(len(E))
        E['auto'] = (rng_a.random(len(E)) < pa).astype(int)
        if alpha > 0 and len(E):
            info = rng_r.random(len(E)) < alpha
            ET = E.copy(); Ct, Bt = sd.counts(T, E[['x', 'y']].values, E.code.values)
            p0 = sd.hazard(beta, keep, E, Ct, Bt - lam * auto_near(T, E[['x', 'y']].values))
            for i in np.flatnonzero(info):
                cand = pool[Y].iloc[rng_r.choice(len(pool[Y]), K, replace=False)]
                cc, bb = sd.counts(T, cand[['x', 'y']].values, np.repeat(E.code.values[i], K))
                bb = bb - lam * auto_near(T, cand[['x', 'y']].values)
                Z = E.iloc[[i] * K].copy()
                for col in ('x', 'y', 'ls', 'sm', 'mid', 'high', 'dong'): Z[col] = cand[col].values
                pc = sd.hazard(beta, keep, Z, cc, bb); j = int(np.argmin(pc))
                if pc[j] < p0[i]:
                    for col in ('x', 'y', 'ls', 'sm', 'mid', 'high', 'dong'): ET.iloc[i, ET.columns.get_loc(col)] = cand[col].values[j]
            E = ET
        S = pd.concat([T, E], ignore_index=True); out[-1]['stock_after'] = len(S)
    return pd.DataFrame(out), dong_cl

MODEL = None
def init():
    global MODEL
    m, keep, _, _ = sd.fit(sd.YEARS)
    MODEL = (m.params[keep].values, m.cov_params().loc[keep, keep].values, keep)

def one(args):
    (alpha, rho, lam), nrep = args
    b0, cov, keep = MODEL; pool = {Y: attrs(sd.CO[Y]) for Y in STEPS}; R, Dg = [], []
    for r in range(nrep):
        rs = np.random.default_rng(r); beta = rs.multivariate_normal(b0, cov)          # 8단계와 같은 시드 규칙(공통 난수)
        u = rs.random((len(sd.ALLID), len(STEPS))); rr = np.random.default_rng(10_000 + r); ra = np.random.default_rng(20_000 + r)
        o, dc = run(beta, keep, alpha, rho, lam, u, rr, ra, pool)
        o['rep'] = r; o['alpha'] = alpha; o['rho'] = rho; o['lam'] = lam; R.append(o)
        Dg += [dict(alpha=alpha, rho=rho, lam=lam, rep=r, dong=k, closures=v) for k, v in dc.items()]
    return pd.concat(R, ignore_index=True), pd.DataFrame(Dg)

def summarize(R):
    """조합별 2021→2025 누적. 주 지표는 전체 점포 폐업(점포 집합은 모든 시나리오에서 같고 유효 집적만 다름)"""
    tot = R.groupby(['alpha', 'rho', 'lam', 'rep']).agg(clos=('closures', 'sum'), n_dense=('closures_dense', 'sum'), n_sparse=('closures_sparse', 'sum'),
          clos_orig=('closures_orig', 'sum'), stock=('stock_after', 'last'), n_auto=('n_auto', 'last'),
          biz_eff=('mean_biz_eff', 'last'), ent_biz=('ent_biz', 'mean')).reset_index()
    base = tot[(tot.alpha == 0) & (tot.rho == 0)].set_index('rep')
    rows = []
    for (a, r, l), g in tot.groupby(['alpha', 'rho', 'lam']):
        g = g.set_index('rep'); d = g.clos - base.clos.reindex(g.index); do = g.clos_orig - base.clos_orig.reindex(g.index)
        rows.append(dict(alpha=a, rho=r, lam=l, closures=g.clos.mean(), lo=g.clos.quantile(.05), hi=g.clos.quantile(.95),
                         diff_vs_S0=d.mean(), diff_lo=d.quantile(.05), diff_hi=d.quantile(.95), diff_pct=d.mean() / base.clos.mean() * 100,
                         closures_orig=g.clos_orig.mean(), orig_diff=do.mean(), orig_diff_lo=do.quantile(.05), orig_diff_hi=do.quantile(.95),
                         dense=g.n_dense.mean(), sparse=g.n_sparse.mean(), dense_diff=(g.n_dense - base.n_dense.reindex(g.index)).mean(),
                         sparse_diff=(g.n_sparse - base.n_sparse.reindex(g.index)).mean(),
                         dense_diff_pct=(g.n_dense - base.n_dense.reindex(g.index)).mean() / base.n_dense.mean() * 100,
                         sparse_diff_pct=(g.n_sparse - base.n_sparse.reindex(g.index)).mean() / base.n_sparse.mean() * 100,
                         stock_2025=g.stock.mean(), n_auto_2025=g.n_auto.mean(), biz_eff_2025=g.biz_eff.mean(), entrant_biz=g.ent_biz.mean(), nrep=len(g)))
    return pd.DataFrame(rows)

if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] == 'fig'):
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':      # 저장된 pai_runs.csv로 요약만 다시
        Sm = summarize(pd.read_csv(os.path.join(OUT, 'pai_runs.csv'))); Sm.to_csv(os.path.join(OUT, 'pai_summary.csv'), index=False, encoding='utf-8-sig')
        pd.set_option('display.width', 250); print(Sm.round(1).to_string(index=False)); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        init(); R, _ = one(((0.0, 0.0, 0.5), 3)); R4, _ = one(((0.4, 0.0, 0.5), 3))
        ref = pd.read_csv(os.path.join(W, 'outputs', 'yearly', 'sd_scenarios.csv'))
        for a, X in ((0.0, R), (0.4, R4)):
            m = X.merge(ref[ref.alpha == a], on=['rep', 'year'], suffixes=('', '_ref'))
            print(f'α={a}: 8단계와 폐업 수 일치 {(m.closures == m.closures_ref).mean():.0%} ({len(m)}행)')
        sys.exit()
    with Pool(2, initializer=init) as P:
        res = P.map(one, [(c, NREP) for c in COMBOS])
    R = pd.concat([r[0] for r in res], ignore_index=True); D = pd.concat([r[1] for r in res], ignore_index=True)
    R.to_csv(os.path.join(OUT, 'pai_runs.csv'), index=False, encoding='utf-8-sig')
    D.to_csv(os.path.join(OUT, 'pai_dong.csv'), index=False, encoding='utf-8-sig')
    Sm = summarize(R); Sm.to_csv(os.path.join(OUT, 'pai_summary.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); print(Sm.round(1).to_string(index=False))


def figure():
    """outputs/physical_ai/fig_physical_ai.png: (a) ρ·α별 일반 점포 누적 폐업, (b) 밀집지·비밀집지 증가분 (λ = 0.5)"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if any(k in f for k in ('malgun', 'AppleSDGothicNeo', 'NotoSansCJK', 'NanumGothic')):
            font_manager.fontManager.addfont(f); plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name(); break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    S = pd.read_csv(os.path.join(OUT, 'pai_summary.csv')); S = S[S.lam == 0.5]
    BLUE, GRAY, INK = '#2a5d9f', '#8a8f98', '#1b1b1b'
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=220)
    ax = axs[0]
    for a, c, mk, dx, lab in ((0.0, GRAY, 'o', -0.08, 'AI 입지 정보 없음 (α = 0%)'), (0.4, BLUE, 's', 0.08, 'AI 입지 정보 활용 (α = 40%)')):
        g = S[S.alpha == a].sort_values('rho'); x = np.arange(len(g)) + dx
        ax.errorbar(x, g.closures, yerr=[g.closures - g.lo, g.hi - g.closures], fmt=mk, color=c, ecolor=c, capsize=3, ms=4.5, lw=1.1, label=lab)
        for xi, v, d in zip(x, g.closures, g.diff_pct):
            ax.text(xi + (0.1 if a else -0.1), v, f'{v:,.0f}\n({d:+.1f}%)' if (a or d) else f'{v:,.0f}', fontsize=6.3, ha='left' if a else 'right', va='center', color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(['0%', '20%', '40%']); ax.set_xlabel('신규 점포 중 자동화형 비율 ρ'); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel('누적 폐업 2021→2025(곳)'); ax.set_title('(a) 누적 폐업 (평균, 5~95%)', loc='left', fontsize=8)
    ax.grid(axis='y', color='#e6e6e6', lw=.5); ax.set_ylim(1640, 1935); ax.legend(frameon=False, fontsize=6.5, loc='upper center', ncol=2)
    ax = axs[1]; g = S[S.alpha == 0].sort_values('rho'); x = np.arange(len(g)); w = 0.36
    ax.bar(x - w / 2, g.dense_diff_pct, w, color=BLUE, label='밀집지 (200m 음식점 상위 1/3)'); ax.bar(x + w / 2, g.sparse_diff_pct, w, color=GRAY, label='그 외 지역')
    for xi, v1, v2 in zip(x, g.dense_diff_pct, g.sparse_diff_pct):
        ax.text(xi - w / 2, v1 + 0.03, f'{v1:+.1f}%', ha='center', va='bottom', fontsize=6.3); ax.text(xi + w / 2, v2 + 0.03, f'{v2:+.1f}%', ha='center', va='bottom', fontsize=6.3)
    ax.set_xticks(x); ax.set_xticklabels(['0%', '20%', '40%']); ax.set_xlabel('신규 점포 중 자동화형 비율 ρ'); ax.axhline(0, color='#999', lw=.6)
    ax.set_ylabel('기준(ρ = 0) 대비 폐업 증가율(%)'); ax.set_title('(b) 지역별 폐업 증가율 (α = 0%, λ = 0.5)', loc='left', fontsize=8); ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.grid(axis='y', color='#e6e6e6', lw=.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_physical_ai.png')); plt.close(fig); print('그림 저장')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'fig':
    figure()
