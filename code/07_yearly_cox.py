"""
7단계: 연도별 코호트 Cox — 기준일 2021~2025년 코호트마다 같은 Cox 모형을 추정 (보고서 표 1)

입력  data/yearly/cohort_{연도}_end20261013.csv   (6단계 결과)
출력  outputs/yearly/yearly_cox_summary.csv       코호트 × 반경 × 모형별 HR, 95% CI, p, C-index, 네트워크 우도비 검정
모형  A: C·BIZ(10곳 단위), B: R·BIZ, C: A + 네트워크 지표(우도비 검정)
      통제: 업력, 임대료 수준, 업태 / 상권 군집 강건 SE. 전체 점포 사용
      추정매출(log 매출)은 통제 변수에서 뺀다(2026-10-20 팀 합의: 매출은 유의하지 않고, 808곳이 결측이라 대입·더미·표본 제외 없이
      모든 점포를 쓰기 위함). 4단계 강건성 검토의 '매출 변수 제외(전체 3,775곳)' 설정과 같은 사양이다.
      모형 코드는 code/yearly_cox_spec.py(4단계 Cox의 2026-10-15 버전)의 함수를 쓰고 통제 변수만 바꾼다.
실행  python code/07_yearly_cox.py              (약 2분)
      NOSALES=0 python code/07_yearly_cox.py     이전 사양(log 매출 + 결측 더미) → yearly_cox_summary_with_sales.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, 'code'))
import yearly_cox_spec as cx
TAG = os.environ.get('TAG', '20261013')
NOSALES = os.environ.get('NOSALES', '1') == '1'          # 기본: 추정매출 통제변수 제외
SFX = '' if NOSALES else '_with_sales'
if NOSALES:
    _orig = cx.controls
    cx.controls = lambda df, rent='level': _orig(df, rent).drop(columns=['log_sales', 'sales_missing'], errors='ignore')
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
S.to_csv(os.path.join(W, 'outputs', 'yearly', f'yearly_cox_summary{SFX}.csv'), index=False, encoding='utf-8-sig')
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
