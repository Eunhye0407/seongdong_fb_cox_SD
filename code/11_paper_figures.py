"""
11단계: 논문(KIIT 2026 추계) 그림 1~4 — 논문에 넣은 그림을 결과 파일에서 그대로 다시 그린다

입력  outputs/yearly/sd_backtest.csv, sd_scenarios.csv (8단계), outputs/physical_ai/pai_summary.csv (10단계),
      data/yearly/cohort_*.csv (실제 점포 수, 단순 평균 폐업률 모형)
출력  outputs/paper/fig1_model.png            그림 1. 연구 모형
      outputs/paper/fig2_validation.png       그림 2. 점포 수 검증(2021년 시작, 전체 기간 계수)
      outputs/paper/fig3_scenarios.png        그림 3. 정보 활용률별 누적 폐업과 신규 점포 입지 밀집도
      outputs/paper/fig4_physical_ai.png      그림 4. 피지컬 AI 대체 시나리오의 폐업 변화와 위치
실행  python code/11_paper_figures.py
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(W, 'outputs', 'paper'); os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location('sd', os.path.join(W, 'code', '08_sd_simulation.py')); sd = importlib.util.module_from_spec(spec); spec.loader.exec_module(sd)

for _f in sorted(font_manager.findSystemFonts()):   # 한글 글꼴: Noto Sans CJK 우선, 없으면 맑은 고딕·Apple SD Gothic Neo·나눔고딕
    if any(k in _f for k in ('NotoSansCJK-Regular', 'NotoSansKR', 'malgun', 'AppleSDGothicNeo', 'NanumGothic')):
        font_manager.fontManager.addfont(_f); plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name(); break
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BLUE, GRAY, INK, LIGHT, BOXB = '#2a5d9f', '#8a8f98', '#1b1b1b', '#dfe5ef', '#e3eaf4'
GRID = dict(axis='y', color='#e6e6e6', lw=.6)

# ---------- 그림 1. 연구 모형
fig, ax = plt.subplots(figsize=(3.25, 2.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fc, ec=INK, lw=1.0))
    ax.text(x + w / 2, y + h - 2.8, title, ha='center', va='top', fontsize=5.6, weight='bold')
    ax.text(x + w / 2, y + h - 8.6, body, ha='center', va='top', fontsize=4.5, linespacing=1.5)
def arrow(p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, color=kw.pop('color', INK), lw=kw.pop('lw', 1.0), **kw))
box(0.5, 51, 27, 17, '통제 변수 Z', '업력 · 임대료 수준\n매출(log, 본 분석) · 업태', '#f0f0f0')
box(0.5, 27, 27, 18, '반경 경쟁 K(r)', '동종 C · 전체 BIZ\n비율 R (r = 100~500m)', BOXB)
box(0.5, 3, 27, 18, '경쟁 네트워크 N(r)', '연결강도 · 매개중심성\n군집계수', BOXB)
box(38, 25, 22, 21, '폐업 위험', 'Cox 비례위험\n기준일 2021~2025', 'white')
box(69, 9, 30.5, 57, 'System Dynamics', '영업 점포 수 S(t)\nS(t+1) = S(t)\n− 폐업 D(t) + 신규 E(t)\n\nD(t) = 점포별\n1년 폐업확률의 합\n\n정보 활용률 α\n자동화형 ρ 시나리오', 'white')
arrow((24, 51), (39.5, 46)); ax.text(34, 54, '통제', fontsize=5)
arrow((27.5, 35.5), (38, 35.5)); ax.text(28.2, 37.5, 'H1–H3', fontsize=4.8)
arrow((24, 21), (39.5, 25)); ax.text(34, 16, 'H4', fontsize=5)
arrow((60, 35.5), (69, 35.5)); ax.text(61.5, 37.5, '계수', fontsize=5)
arrow((78, 9), (50, 24), color='#999999', lw=.8, ls='--', connectionstyle='arc3,rad=-0.25')
ax.text(53, 4.5, '신규·폐업 → 밀도(C200·BIZ200) 재계산\n→ 다음 해 폐업 위험 (피드백)', fontsize=4.4, color='#555555', ha='center', va='center')
fig.savefig(os.path.join(OUT, 'fig1_model.png'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------- 그림 2. 점포 수 검증 (2021 시작, 전체 기간 계수)
bt = pd.read_csv(os.path.join(W, 'outputs', 'yearly', 'sd_backtest.csv'))
actual = {Y: len(sd.CO[Y]) for Y in sd.YEARS}
o = bt[bt.name == '전체기간 계수, 2021 시작']
g = o.groupby('year').stock_after.agg(m='mean', lo=lambda s: s.quantile(.05), hi=lambda s: s.quantile(.95))
yr = sd.YEARS; xs = list(range(len(yr)))
rate = np.mean([sd.CO[y].ev.mean() for y in sd.YEARS]); s_b = actual[2021]; nb = [s_b]
for y in yr[1:]: s_b = s_b * (1 - rate) + len(sd.entrants(y)); nb.append(s_b)
fig, ax = plt.subplots(figsize=(3.25, 2.0))
ax.fill_between(xs, [actual[2021]] + g.lo.tolist(), [actual[2021]] + g.hi.tolist(), color=BLUE, alpha=.15, lw=0, label='SD 5~95% 구간')
ax.plot(xs, [actual[y] for y in yr], color=INK, marker='o', ms=4, lw=1.6, label='실제')
ax.plot(xs, [actual[2021]] + g.m.tolist(), color=BLUE, marker='s', ms=3.5, lw=1.4, label='SD 평균(200회)')
ax.plot(xs, nb, color=GRAY, ls='--', lw=1.2, label='단순 평균 폐업률 모형')
ax.set_xticks(xs); ax.set_xticklabels([f'{y}.4' for y in yr]); ax.set_ylabel('영업 점포 수(곳)'); ax.set_ylim(3650, 4120)
ax.grid(**GRID); ax.legend(frameon=False, fontsize=6.5, loc='lower right')
fig.savefig(os.path.join(OUT, 'fig2_validation.png'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------- 그림 3. 정보 활용률 시나리오
sc = pd.read_csv(os.path.join(W, 'outputs', 'yearly', 'sd_scenarios.csv'))
tot = sc.groupby(['alpha', 'rep']).agg(clos=('closures', 'sum'), entb=('ent_biz', 'mean')).reset_index()
al = [0.0, 0.4, 0.6]; lab = ['0%\n(기준)', '40%', '60%']
m = np.array([tot[tot.alpha == a].clos.mean() for a in al]); lo = np.array([tot[tot.alpha == a].clos.quantile(.05) for a in al]); hi = np.array([tot[tot.alpha == a].clos.quantile(.95) for a in al])
b = [tot[tot.alpha == a].entb.mean() for a in al]
fig, axs = plt.subplots(1, 2, figsize=(3.25, 1.9))
ax = axs[0]
ax.errorbar(range(3), m, yerr=[m - lo, hi - m], fmt='o', color=BLUE, ecolor=BLUE, capsize=3, ms=4.5, lw=1.2)
for i, v in enumerate(m): ax.text(i, hi[i] + 6, f'{v:,.0f}' + ('' if i == 0 else f'\n({(v / m[0] - 1) * 100:.1f}%)'), ha='center', va='bottom', fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_xlim(-.6, 2.6); ax.set_ylim(1580, 1960)
ax.set_ylabel('누적 폐업 2021→2025(곳)'); ax.set_title('(a) 누적 폐업(평균, 5~95%)', fontsize=6.8); ax.grid(**GRID); ax.set_axisbelow(True)
ax = axs[1]
ax.bar(range(3), b, color=[GRAY, BLUE, BLUE], width=.55); ax.set_axisbelow(True)
for i, v in enumerate(b): ax.text(i, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylim(0, 200); ax.set_ylabel('신규 점포 200m 내 음식점 수')
ax.set_title('(b) 신규 점포 입지 밀집도', fontsize=6.8); ax.grid(**GRID)
fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1)); fig.text(0.5, 0.02, '정보 활용률 α', ha='center', fontsize=7)
fig.savefig(os.path.join(OUT, 'fig3_scenarios.png'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)

# ---------- 그림 4. 피지컬 AI 대체 시나리오 (λ = 0.5)
S = pd.read_csv(os.path.join(W, 'outputs', 'physical_ai', 'pai_summary.csv')); S = S[S.lam == 0.5]
fig, axs = plt.subplots(1, 2, figsize=(3.25, 1.95))
ax = axs[0]; ax.axhline(0, color=GRAY, lw=.9)
for a, c, mk, dx, labl in ((0.0, GRAY, 'o', -0.08, 'α = 0%'), (0.4, BLUE, 's', 0.08, 'α = 40%')):
    gg = S[S.alpha == a].sort_values('rho'); x = np.arange(len(gg)) + dx
    ax.errorbar(x, gg.diff_vs_S0, yerr=[gg.diff_vs_S0 - gg.diff_lo, gg.diff_hi - gg.diff_vs_S0], fmt=mk, color=c, ecolor=c, capsize=2.5, ms=4, lw=1.1, label=labl)
    for xi, v in zip(x, gg.diff_vs_S0):
        ax.text(xi + (0.1 if a else -0.1), v, f'{v:+.0f}', fontsize=5.6, ha='left' if a else 'right', va='center', color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(['0%', '20%', '40%']); ax.set_xlim(-0.6, 2.6)
ax.set_xlabel('신규 점포 중 자동화형 비율 ρ (λ = 0.5)', fontsize=6.2); ax.set_ylabel('기준 대비 누적 폐업(곳)')
ax.set_title('(a) 누적 폐업 변화', fontsize=6.8); ax.grid(**GRID); ax.legend(frameon=False, fontsize=5.8, loc='center left', bbox_to_anchor=(0.0, 0.6))
ax = axs[1]
k = lambda a, r: S[(S.alpha == a) & (S.rho == r)].iloc[0]
dense = [k(0.0, 0.4)['dense'] - k(0.0, 0.0)['dense'], k(0.4, 0.4)['dense'] - k(0.4, 0.0)['dense']]
spars = [k(0.0, 0.4)['sparse'] - k(0.0, 0.0)['sparse'], k(0.4, 0.4)['sparse'] - k(0.4, 0.0)['sparse']]
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, dense, w, color=BLUE, label='밀집지'); ax.bar(x + w / 2, spars, w, color=GRAY, label='그 외')
for xi, v1, v2 in zip(x, dense, spars):
    ax.text(xi - w / 2, v1 + 0.3, f'{v1:.0f}', ha='center', va='bottom', fontsize=6); ax.text(xi + w / 2, v2 + 0.3, f'{v2:.0f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(['정보 없음\n(α = 0%)', 'AI 정보\n(α = 40%)'], fontsize=6.2); ax.set_ylim(0, max(dense + spars) * 1.2)
ax.set_ylabel('ρ = 40%의 추가 폐업(곳)'); ax.set_title('(b) 추가 폐업 위치', fontsize=6.8); ax.grid(**GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.8, loc='upper left')
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(OUT, 'fig4_physical_ai.png'), bbox_inches='tight', pad_inches=0.02); plt.close(fig)

print('저장: outputs/paper/fig1_model.png, fig2_validation.png, fig3_scenarios.png, fig4_physical_ai.png')
print(f'그림 4(b) ρ=40%, λ=0.5 추가 폐업 — 정보 없음: 밀집지 {dense[0]:.1f} / 그 외 {spars[0]:.1f} | AI 정보: 밀집지 {dense[1]:.1f} / 그 외 {spars[1]:.1f}')
