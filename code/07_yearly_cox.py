"""
7단계: 연도별 코호트 Cox — 기준일 2021~2025년 코호트마다 같은 Cox 모형을 추정 (보고서 표 1)

입력  data/yearly/cohort_{연도}_end20261013.csv   (6단계 결과)
출력  outputs/yearly/yearly_cox_summary.csv       코호트 × 반경 × 모형별 HR, 95% CI, p, C-index, 네트워크 우도비 검정
모형  A: C·BIZ(10곳 단위), B: R·BIZ, C: A + 네트워크 지표(우도비 검정)
      통제: 업력, 임대료 수준, log 상권 매출 + 매출 결측 더미(결측은 중앙값 대입), 업태 / 상권 군집 강건 SE
      모형 코드는 code/yearly_cox_spec.py (4단계 Cox의 2026-10-15 버전). 현재 4단계와 달리 매출 결측 점포를 제외하지 않고
      전체 점포를 쓴다 → 2021 코호트 결과가 outputs/tables(4단계)와 조금 다르다. README의 'Cox 결과가 두 가지인 이유' 참고
실행  python code/07_yearly_cox.py   (약 1분)
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, 'code'))
import yearly_cox_spec as cx
TAG = os.environ.get('TAG', '20261013')
rows = []
for Y in range(2021, 2026):
    cx.DATA_PATH = os.path.join(W, 'data', 'yearly', f'cohort_{Y}_end{TAG}.csv')
    df = cx.load_data()
    rm, fits = cx.radius_models(df)
    nm = cx.network_models(df, fits)
    rm['cohort'] = Y; rm['n'] = len(df); rm['events'] = int(df.event.sum()); rm['table'] = 'radius'
    nm['cohort'] = Y; nm['n'] = len(df); nm['events'] = int(df.event.sum()); nm['table'] = 'network'
    rows += [rm, nm]
    print(Y, len(df), int(df.event.sum()))
S = pd.concat(rows, ignore_index=True)
os.makedirs(os.path.join(W, 'outputs', 'yearly'), exist_ok=True)
S.to_csv(os.path.join(W, 'outputs', 'yearly', 'yearly_cox_summary.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 220)
a = S[(S.table == 'radius') & (S.model == 'A')]
for v in ('competitor_c', 'adjacent_biz'):
    t = a[a.variable.str.startswith(v)].pivot(index='cohort', columns='radius_m', values='hr_per_sd')
    p = a[a.variable.str.startswith(v)].pivot(index='cohort', columns='radius_m', values='p')
    print(f'\n[모형 A] {v} HR(SD 1 증가당)'); print(t.round(3).to_string()); print('p'); print(p.round(3).to_string())
b = S[(S.table == 'radius') & (S.model == 'B') & S.variable.str.startswith('competitor_r')]
print('\n[모형 B] competitor_r HR(SD당)'); print(b.pivot(index='cohort', columns='radius_m', values='hr_per_sd').round(3).to_string()); print(b.pivot(index='cohort', columns='radius_m', values='p').round(3).to_string())
n = S[S.table == 'network']
print('\n[네트워크 증분] LR p (cohort×radius)'); print(n.drop_duplicates(['cohort', 'radius_m']).pivot(index='cohort', columns='radius_m', values='lr_p').round(3).to_string())
ns = n[n.variable.str.startswith('net_strength')]
print('연결강도 HR(SD당) / p'); print(ns.pivot(index='cohort', columns='radius_m', values='hr').round(3).to_string()); print(ns.pivot(index='cohort', columns='radius_m', values='p').round(3).to_string())
