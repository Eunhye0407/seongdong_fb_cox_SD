"""9단계: SD 결과 그림 2개 — (A) 점포 수 검증 곡선, (B) 정보 활용률 시나리오
입력  outputs/yearly/sd_backtest.csv, sd_scenarios.csv (8단계 결과), data/yearly/cohort_*.csv
출력  outputs/yearly/figA_sd_validation.png, figB_sd_scenarios.png
실행  python code/09_sd_figures.py
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(W, 'outputs', 'yearly')
spec = importlib.util.spec_from_file_location('sd', os.path.join(W, 'code', '08_sd_simulation.py')); sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)
from matplotlib import font_manager
for _f in font_manager.findSystemFonts():   # 한글 글꼴: Windows 맑은 고딕, macOS Apple SD Gothic Neo, Linux Noto/Nanum
    if any(k in _f for k in ('malgun', 'AppleSDGothicNeo', 'NotoSansCJK', 'NanumGothic')):
        font_manager.fontManager.addfont(_f); plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, GRAY, DARK, RED = '#2a5d9f', '#8a8f98', '#1b1b1b', '#c0504d'
bt = pd.read_csv(os.path.join(OUT, 'sd_backtest.csv')); sc = pd.read_csv(os.path.join(OUT, 'sd_scenarios.csv'))
actual = {Y: len(sd.CO[Y]) for Y in sd.YEARS}

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=220, sharey=True)
cfg = [('전체기간 계수, 2021 시작', 2021, [2022, 2023, 2024, 2025], '(a) 2021년 시작, 전체 기간 계수'), ('2021~22 계수, 2023 시작(시간 분할)', 2023, [2024, 2025], '(b) 2023년 시작, 2021~22년 계수만 사용')]
for ax, (name, start, steps, title) in zip(axs, cfg):
    o = bt[bt.name == name]; g = o.groupby('year').stock_after.agg(['mean', lambda s: s.quantile(.05), lambda s: s.quantile(.95)]); g.columns = ['m', 'lo', 'hi']
    xs = [start] + steps; yr = list(range(start, 2026))
    ax.plot(yr, [actual[y] for y in yr], color=DARK, marker='o', ms=4, lw=1.6, label='실제')
    ax.plot(xs, [actual[start]] + g.m.tolist(), color=BLUE, marker='s', ms=3.5, lw=1.4, label='SD 시뮬레이션 평균')
    ax.fill_between(xs, [actual[start]] + g.lo.tolist(), [actual[start]] + g.hi.tolist(), color=BLUE, alpha=.18, lw=0, label='5~95% 구간')
    rate = np.mean([sd.CO[y].ev.mean() for y in (sd.YEARS if start == 2021 else (2021, 2022))]); s_b = actual[start]; nb = [s_b]
    for y in steps: s_b = s_b * (1 - rate) + len(sd.entrants(y)); nb.append(s_b)
    ax.plot(xs, nb, color=GRAY, ls='--', lw=1.2, label='단순 평균 폐업률 모형')
    ax.set_title(title, fontsize=9, loc='left'); ax.set_xticks(yr); ax.set_xticklabels([f'{y}.4' for y in yr]); ax.grid(axis='y', color='#e6e6e6', lw=.6)
axs[0].set_ylabel('영업 점포 수(기준일 4월 1일)'); axs[0].set_ylim(3700, 4150); axs[1].legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figA_sd_validation.png')); plt.close(fig)

tot = sc.groupby(['alpha', 'rep']).agg(clos=('closures', 'sum'), entb=('ent_biz', 'mean'), entp=('ent_p', 'mean')).reset_index()
al = [0.0, 0.4, 0.6]; lab = ['0% (S0)', '40%', '60%']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=220)
m = np.array([tot[tot.alpha == a].clos.mean() for a in al]); lo = np.array([tot[tot.alpha == a].clos.quantile(.05) for a in al]); hi = np.array([tot[tot.alpha == a].clos.quantile(.95) for a in al])
axs[0].errorbar(range(3), m, yerr=[m - lo, hi - m], fmt='o', color=BLUE, ecolor=BLUE, capsize=4, ms=6, lw=1.4)
for i, v in enumerate(m): axs[0].text(i, hi[i] + 8, f'{v:,.0f}' + ('' if i == 0 else f'\n({v - m[0]:+.0f}, {(v / m[0] - 1) * 100:+.1f}%)'), ha='center', va='bottom', fontsize=8)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(lab); axs[0].set_xlim(-.5, 2.5); axs[0].set_ylim(1550, 1950)
axs[0].set_ylabel('누적 폐업 점포 수 (평균, 5~95%)'); axs[0].set_xlabel('정보 활용률 α'); axs[0].set_title('(a) 누적 폐업', fontsize=9, loc='left'); axs[0].grid(axis='y', color='#e6e6e6', lw=.6); axs[0].set_axisbelow(True)
b = [tot[tot.alpha == a].entb.mean() for a in al]; p = [tot[tot.alpha == a].entp.mean() for a in al]
axs[1].bar(range(3), b, color=[GRAY, BLUE, BLUE], width=.5); axs[1].set_axisbelow(True)
for i, v in enumerate(b): axs[1].text(i, v + 4, f'{v:.0f}곳\n폐업확률 {p[i]:.3f}', ha='center', va='bottom', fontsize=8)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(lab); axs[1].set_xlim(-.6, 2.6)
axs[1].set_ylim(0, 235); axs[1].set_ylabel('신규 점포 200m 안 음식점 수(평균)'); axs[1].set_xlabel('정보 활용률 α'); axs[1].set_title('(b) 신규 점포 입지의 밀집도와 예상 폐업확률', fontsize=9, loc='left'); axs[1].grid(axis='y', color='#e6e6e6', lw=.6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figB_sd_scenarios.png')); plt.close(fig)
print('저장 완료')
