"""14단계(논문 v11 3.6절, 그림 2): 2027~2030 정책 예측 — 2026-04-01 실제 점포 3,947곳에서 출발, 진입 내생(R1·B1·B2)

시나리오 (공통 난수로 비교, 위험식은 2021~2025 전이, 진입식은 2026 전이까지 학습)
  S0          기준
  P1_info     AI 입지 정보 α = 40%: 정보를 받은 신규 점포가 빈 공실 5곳과 현재 입지 중 예상 폐업 확률이 가장 낮은 곳을 택함(공실당 한 점포)
  P2_auto     자동화형 신규 점포 ρ = 40%, λ = 0.5 (10단계와 같은 정의)
  P3_both     P1 + P2
  P4_vacancy  공실 재입점 지원: 진입식의 공실 탄력성 b를 두 배로(B2 고리 강화)

입력  12단계 모형, data/yearly/cohort_*.csv, data/raw/성동구_일반음식점_인허가_갱신_20261019.csv
      outputs/sd_feedback/val_runs.csv (그림의 검증 부분, 13단계 결과)
출력  outputs/sd_feedback/fc_runs.csv       시나리오 × 회차 × 연도
      outputs/sd_feedback/fc_summary.csv    시나리오별 2030 점포 수·4년 누적 폐업·기준 대비 차이(평균, 5~95%)
      outputs/sd_feedback/fig_validation_forecast.png   논문 그림 2
실행  python code/14_sd_forecast.py         (NREP=100, 2코어 약 40분 — 정보 시나리오가 느림)
      python code/14_sd_forecast.py fig     저장된 결과로 요약·그림만 다시
"""
import os, sys, importlib.util, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s = importlib.util.spec_from_file_location('fb', os.path.join(W, 'code', '12_sd_feedback_model.py')); fb = importlib.util.module_from_spec(_s); _s.loader.exec_module(fb)
sd, OUT = fb.sd, fb.OUT
NREP = int(os.environ.get('NREP', 100)); STEPS = [2027, 2028, 2029, 2030]
SC = {'S0': dict(), 'P1_info': dict(alpha=0.4), 'P2_auto': dict(rho=0.4, lam=0.5), 'P3_both': dict(alpha=0.4, rho=0.4, lam=0.5), 'P4_vacancy': dict(bmult=2.0)}
M = {}
def init(): fb.build_2026(); M['m'] = sd.fit(sd.YEARS); M['em'] = fb.entry_model(2026)
def job(name):
    kw = dict(SC[name]); bm = kw.pop('bmult', 1.0); em = dict(M['em']); em['b'] *= bm; em['bse'] *= bm; R = []
    for r in range(NREP):
        beta, keep = fb.draw_beta(M['m'], np.random.default_rng(r))
        o, _ = fb.run(2026, STEPS, beta, keep, em, r, entry='endo', **kw); o['rep'] = r; o['sc'] = name; R.append(o)
    return pd.concat(R)

def summarize(F):
    T = F.groupby(['sc', 'rep']).agg(clos=('closures', 'sum'), dense=('closures_dense', 'sum'), stock=('stock_after', 'last'), ent=('entries', 'sum')).reset_index()
    b = T[T.sc == 'S0'].set_index('rep'); rows = []
    for s, g in T.groupby('sc'):
        g = g.set_index('rep'); d = g.clos - b.clos; ds = g.stock - b.stock
        rows.append(dict(scenario=s, stock_2030=g.stock.mean(), stock_lo=g.stock.quantile(.05), stock_hi=g.stock.quantile(.95), closures=g.clos.mean(),
                         d_closures=d.mean(), d_lo=d.quantile(.05), d_hi=d.quantile(.95), d_pct=100 * d.mean() / b.clos.mean(),
                         d_stock=ds.mean(), d_stock_lo=ds.quantile(.05), d_stock_hi=ds.quantile(.95), d_dense=(g.dense - b.dense).mean(), entries=g.ent.mean()))
    return pd.DataFrame(rows)

def figure(F, V):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    act = {2021: 3775, 2022: 3866, 2023: 3908, 2024: 4067, 2025: 4009, 2026: 3947}
    fig, ax = plt.subplots(figsize=(4.6, 2.45), dpi=300)
    def band(df, color, label, ls='-', fill=True, start=None):
        g = df.groupby('year').stock_after; m, lo, hi = g.mean(), g.quantile(.05), g.quantile(.95)
        x, y = [start[0]] + list(m.index), [start[1]] + list(m.values)
        lo = pd.concat([pd.Series({start[0]: start[1]}), lo]); hi = pd.concat([pd.Series({start[0]: start[1]}), hi])
        if fill: ax.fill_between(lo.index, lo.values, hi.values, color=color, alpha=.18, lw=0)
        ax.plot(x, y, ls, color=color, lw=1.4, label=label)
    band(V[V.case == 'B_M2'], '#2b6cb0', 'Back-test 2022–25 (endogenous entry)', start=(2021, 3775))
    h = V[V.case == 'H_M2'].stock_after
    ax.errorbar([2026], [h.mean()], yerr=[[h.mean() - h.quantile(.05)], [h.quantile(.95) - h.mean()]], fmt='s', color='#2b6cb0', ms=4, capsize=2, lw=1, label='Hold-out 2026 (data to 2025)')
    cols = {'S0': ('#555555', 'Forecast S0 from actual 2026 (90% band)', '-'), 'P1_info': ('#2f855a', 'P1 AI location info (α = 40%)', '--'),
            'P2_auto': ('#c05621', 'P2 automated entrants (ρ = 40%)', '--'), 'P3_both': ('#9b2c2c', 'P3 = P1 + P2', ':'), 'P4_vacancy': ('#6b46c1', 'P4 vacancy re-entry support (b × 2)', '-.')}
    for k, (c, lab, ls) in cols.items(): band(F[F.sc == k], c, lab, ls=ls, fill=(k == 'S0'), start=(2026, 3947))
    ax.plot(list(act), list(act.values()), 'o-', color='black', ms=3.5, lw=1.2, label='Actual', zorder=5)
    ax.axvline(2026.0, color='#999', lw=.6, ls=':')
    ax.set_xticks(range(2021, 2031)); ax.set_xticklabels([str(y)[2:] if y > 2021 else '2021' for y in range(2021, 2031)])
    ax.set_ylabel('Operating restaurants (April 1)'); ax.set_xlabel('Year'); ax.legend(fontsize=6.3, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False); ax.set_xlim(2020.7, 2030.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_validation_forecast.png'), dpi=300)

if __name__ == '__main__':
    if not (len(sys.argv) > 1 and sys.argv[1] == 'fig'):
        with Pool(2, initializer=init) as p: F = pd.concat(p.map(job, list(SC)))
        F.to_csv(os.path.join(OUT, 'fc_runs.csv'), index=False, encoding='utf-8-sig')
    F = pd.read_csv(os.path.join(OUT, 'fc_runs.csv'))
    Sm = summarize(F); Sm.to_csv(os.path.join(OUT, 'fc_summary.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); print(Sm.round(1).to_string(index=False))
    figure(F, pd.read_csv(os.path.join(OUT, 'val_runs.csv')))
