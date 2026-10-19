"""
6단계: 연도별 기준일 코호트(2021~2025년 4월 1일) 데이터셋 구축 — 연도별 Cox(7단계)와 SD(8단계)의 입력

코호트 정의  매년 4월 1일에 영업 중인 점포. 관찰 종료일(END, 기본 2026-10-13)까지 폐업하면 event=1, 아니면 중도절단
변수 정의    3단계(code/03_build_features.py)와 같은 함수로 임대료 범주·반경 경쟁·네트워크 변수를 코호트마다 다시 계산

입력  data/raw/성동구_일반음식점_인허가_갱신_20261019.csv         서울 열린데이터광장, 2026-10-19 내려받음(개업 ~10/14, 폐업 ~10/16)
      data/raw/서울시 상권분석서비스(영역-상권).shp                상권 경계
      data/raw/서울시_상권분석서비스(점포-상권)_{연도}년.csv.zip     2021~2025 (저장소에 있음)
      data/raw/서울시_상권분석서비스(추정매출-상권)_{연도}년.csv.zip  2021~2025 (※ 2022~2025는 용량 때문에 저장소에 없음 → README 참고)
      data/raw/서울특별시_성동구 상가임대료 정보_20211201.csv
      data/interim/yearly_snapshot_20260912/restaurants_with_reviews_raw.csv
                 2021 코호트 3,775곳의 좌표·상권·업종코드·임대료·매출. 2026-09-12 기준 2단계 결과의 사본(이 값으로 코호트를 만들었음)
      data/interim/yearly_snapshot_20260912/geocode_vworld_{full,pilot}.csv
                 2021 코호트 밖 새 점포 중 좌표가 없는 곳의 VWorld 지오코딩 결과(인허가 좌표계 EPSG:5174)
출력  data/yearly/cohort_{연도}_end{종료일}.csv   점포 1행. 3단계 최종 데이터와 같은 열 이름(리뷰 열 제외)

실행  python code/06_yearly_cohorts.py                    종료일 2026-10-13 (환경변수 END로 변경)
      END=2026-09-12 python code/06_yearly_cohorts.py     2021 코호트가 9/12 기준 3단계 결과와 일치하는지 검증

방침  - 2021 코호트는 3단계 표본(3,775곳)과 같고 폐업 갱신만 반영한다. 이 점포들의 좌표·상권·업종코드·임대료·매출은 위 사본 값을 쓰고,
        2022~2025년에 새로 들어온 점포만 같은 방법으로 계산한다.
      - 좌표: 인허가 X,Y를 EPSG:5174로 보고 WGS84로 변환(1단계와 같은 방식)
      - 임대료는 2021년 1분기 동별 값으로 고정(2021-11 이후 자료 없음), 상권 점포당 매출은 기준 연도 1분기 값
      - 업태→업종코드 매핑은 사본 값에서 복원(매핑이 없는 업태는 미분류)
"""
import os, re, sys, glob
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(W, 'data', 'raw'); SNAP = os.path.join(W, 'data', 'interim', 'yearly_snapshot_20260912')
OUT = os.path.join(W, 'data', 'yearly'); os.makedirs(OUT, exist_ok=True)
import importlib.util, unicodedata
_spec = importlib.util.spec_from_file_location('bf', os.path.join(W, 'code', '03_build_features.py'))
bf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bf)   # 3단계 변수 정의를 그대로 사용

def raw_file(pattern):
    """data/raw에서 이름이 pattern(glob)에 맞는 파일. macOS에서 올린 파일명(NFD)도 찾는다."""
    hits = [f for f in glob.glob(os.path.join(RAW, '*')) if glob.fnmatch.fnmatch(unicodedata.normalize('NFC', os.path.basename(f)), unicodedata.normalize('NFC', pattern))]
    if not hits: raise FileNotFoundError(f'data/raw에 {pattern} 파일이 없습니다 (README의 6단계 준비 참고)')
    return sorted(hits)[0]

def read_raw_csv(path, **kw):
    """csv 또는 csv 1개가 든 zip을 읽는다(zip 안의 __MACOSX 항목은 무시, 1단계와 같은 방식)"""
    if not path.lower().endswith('.zip'): return pd.read_csv(path, **kw)
    import zipfile
    with zipfile.ZipFile(path) as zf:
        mem = [m for m in zf.namelist() if not m.startswith('__MACOSX') and m.lower().endswith('.csv')]
        assert len(mem) == 1, f'{path}: csv가 1개여야 합니다 {mem}'
        with zf.open(mem[0]) as f: return pd.read_csv(f, **kw)

END = pd.Timestamp(os.environ.get('END', '2026-10-13'))
YEARS = [2021, 2022, 2023, 2024, 2025]
B = {y: pd.Timestamp(f'{y}-04-01') for y in YEARS}
TAG = END.strftime('%Y%m%d')

# ---------- 1. 원자료
L = pd.read_csv(raw_file('성동구_일반음식점_인허가_갱신_20261019.csv'), encoding='cp949', low_memory=False)
L['restaurant_id'] = L['관리번호'].astype(str)
L['open'] = pd.to_datetime(L['인허가일자'], errors='coerce'); L['close'] = pd.to_datetime(L['폐업일자'], errors='coerce')
assert L.restaurant_id.is_unique
R = pd.read_csv(os.path.join(SNAP, 'restaurants_with_reviews_raw.csv'), encoding='utf-8-sig')
R['restaurant_id'] = R['restaurant_id'].astype(str)
print(f'[1] 인허가 갱신본 {len(L)}행 (개업 최대 {L.open.max().date()}, 폐업 최대 {L.close.max().date()}) | 종료일 {END.date()} | 2021 코호트 {len(R)}곳')

# 업태 → 서비스 업종 코드 매핑(저장소 값에서 복원, 업태마다 코드가 하나로 정해지는지 확인)
tmp = R.merge(L[['restaurant_id', '업태구분명']], on='restaurant_id', how='left')
mp = tmp.dropna(subset=['서울시_서비스_업종_코드']).groupby('업태구분명').agg(code=('서울시_서비스_업종_코드', 'first'), name=('서울시_서비스_업종_코드_명', 'first'), k=('서울시_서비스_업종_코드', 'nunique'))
assert (mp.k == 1).all(), '업태-코드 매핑이 유일하지 않음'
print(f'[1] 업태→업종코드 매핑 {len(mp)}개 업태 복원 (매핑 없는 업태는 3단계와 같이 미분류)')

# ---------- 2. 점포 마스터: 2025 기준일까지 한 번이라도 코호트에 들어갈 수 있는 점포
M = L[(L.open <= B[2025]) & (L.close.isna() | (L.close >= B[2021]))].copy()
M = M.merge(R[['restaurant_id', 'longitude', 'latitude', 'TRDAR_CD', 'TRDAR_CD_N', 'rent_dong', 'rent_median', 'area_category_sales_per_store_final']].rename(columns={'longitude': 'lon_eh', 'latitude': 'lat_eh'}), on='restaurant_id', how='left')
M['in_eh'] = M.lon_eh.notna()
x = pd.to_numeric(M['좌표정보(X)'], errors='coerce'); y = pd.to_numeric(M['좌표정보(Y)'], errors='coerce')
# 좌표가 비어 있는 새 점포는 주소 지오코딩 캐시(VWorld, 인허가 좌표계 X,Y로 저장됨)로 채운다
gc = pd.concat([pd.read_csv(os.path.join(SNAP, f), encoding='utf-8-sig') for f in ('geocode_vworld_full.csv', 'geocode_vworld_pilot.csv')])
gc = gc[gc.success == 1].drop_duplicates('관리번호').set_index('관리번호')
fx = M['관리번호'].map(gc.X_5174); fy = M['관리번호'].map(gc.Y_5174)
fill = x.isna() & ~M.in_eh & fx.notna(); x = x.where(~fill, fx); y = y.where(~fill, fy)
print(f'[2] 좌표 지오코딩 캐시로 채움 {int(fill.sum())}곳')
lo, la = Transformer.from_crs('EPSG:5174', 'EPSG:4326', always_xy=True).transform(x.fillna(0).values, y.fillna(0).values)
M['longitude'] = np.where(M.in_eh, M.lon_eh, np.where(x.notna() & y.notna(), lo, np.nan))
M['latitude'] = np.where(M.in_eh, M.lat_eh, np.where(x.notna() & y.notna(), la, np.nan))
print(f'[2] 마스터 {len(M)}곳 (2021 코호트 {int(M.in_eh.sum())} + 새 점포 {int((~M.in_eh).sum())}) | 좌표 없음 {int(M.longitude.isna().sum())} → 제외')
M = M[M.longitude.notna()].copy()
M['category_code'] = M['업태구분명'].map(mp.code); M['category_name'] = M['업태구분명'].map(mp.name)

# 상권(공식 상권 안쪽만, 밖은 NaN)
shp = gpd.read_file(raw_file('*영역-상권*.shp'))
sd = shp[shp['SIGNGU_CD'].astype(str) == '11200'].copy()
pts = gpd.GeoDataFrame(M[['restaurant_id']], geometry=gpd.points_from_xy(M.longitude, M.latitude), crs='EPSG:4326').to_crs(shp.crs)
j = gpd.sjoin(pts, sd[['TRDAR_CD', 'TRDAR_CD_N', 'geometry']], how='left', predicate='within').drop_duplicates('restaurant_id')
M = M.merge(j[['restaurant_id', 'TRDAR_CD', 'TRDAR_CD_N']].rename(columns={'TRDAR_CD': 'td', 'TRDAR_CD_N': 'tdn'}), on='restaurant_id', how='left')
chk = M[M.in_eh]; same = (pd.to_numeric(chk.td, errors='coerce') == pd.to_numeric(chk.TRDAR_CD, errors='coerce')) | (chk.td.isna() & chk.TRDAR_CD.isna())
print(f'[2] 상권 배정 재현: 2021 코호트 점포 {len(chk)}곳 중 일치 {same.mean():.1%} (불일치 시 저장소 값 사용)')
# 매출 배정용 상권: 공식 상권 밖 점포는 가장 가까운 상권(1단계 방식). 군집(cluster)용 trdar_code는 안쪽만
out = M.td.isna()
nr = gpd.sjoin_nearest(pts[out.values], sd[['TRDAR_CD', 'geometry']], how='left').drop_duplicates('restaurant_id').set_index('restaurant_id')['TRDAR_CD']
M['td_sales'] = pd.to_numeric(M.td.where(~out, M.restaurant_id.map(nr)), errors='coerce')
M['trdar_code'] = np.where(M.in_eh, M.TRDAR_CD, M.td); M['trdar_name'] = np.where(M.in_eh, M.TRDAR_CD_N, M.tdn)
M['trdar_code'] = pd.to_numeric(M['trdar_code'], errors='coerce')

# 임대료: 2021년 1분기 동별 중위값(고정)
rent = pd.read_csv(raw_file('*상가임대료*.csv'), encoding='cp949', low_memory=False)
rent['ym'] = pd.to_datetime(rent['기준년월'], errors='coerce'); rent['평당전세가'] = pd.to_numeric(rent['평당전세가'], errors='coerce')
rq = rent[rent.ym.between('2021-01-01', '2021-03-31')].groupby('읍면동')['평당전세가'].median()
ext = lambda a: (lambda m: m.group(1) if m else None)(re.search(r'([가-힣]+(?:동\d*가|동|가))(?=\s|\d|-|$)', str(a)))
M['dong'] = np.where(M.in_eh, M.rent_dong, M['지번주소'].map(ext))
M['rent_median'] = np.where(M.in_eh, M.rent_median, M['dong'].map(rq))
print(f'[2] 임대료 결측 {int(M.rent_median.isna().sum())}곳, 동 {M.dong.nunique()}개')

# ---------- 3. 연도별 상권×업종 점포당 매출(기준 연도 1분기) + 직접값 없는 점포는 가까운 상권 3곳 역거리가중(1단계와 같은 방법)
food = [f'CS1000{i:02d}' for i in range(1, 11)]
codes_sd = set(sd['TRDAR_CD'].astype(str))
cent = sd[['TRDAR_CD', 'geometry']].copy(); cent['TRDAR_CD'] = cent.TRDAR_CD.astype(str); cent['geometry'] = cent.geometry.centroid
pg = gpd.GeoDataFrame(M[['restaurant_id']], geometry=gpd.points_from_xy(M.longitude, M.latitude), crs='EPSG:4326').to_crs(shp.crs)
M['geom'] = pg.geometry.values

def sales_year(Y):
    sp = raw_file(f'*점포-상권*{Y}년.csv*'); ep = raw_file(f'*추정매출-상권*{Y}년.csv*')   # csv 또는 csv.zip
    s = read_raw_csv(sp, encoding='cp949', low_memory=False).rename(columns={'stdr_yyqu_cd': '기준_년분기_코드', 'trdar_cd': '상권_코드', 'svc_induty_cd': '서비스_업종_코드', 'stor_co': '점포_수'})
    e = read_raw_csv(ep, encoding='cp949', low_memory=False)
    q = Y * 10 + 1
    s = s[(s.기준_년분기_코드 == q)]; e = e[(e.기준_년분기_코드 == q)]
    for d in (s, e): d['상권_코드'] = d['상권_코드'].astype(str); d['서비스_업종_코드'] = d['서비스_업종_코드'].astype(str)
    s = s[s.상권_코드.isin(codes_sd) & s.서비스_업종_코드.isin(food)][['상권_코드', '서비스_업종_코드', '점포_수']]
    e = e[e.상권_코드.isin(codes_sd) & e.서비스_업종_코드.isin(food)][['상권_코드', '서비스_업종_코드', '당월_매출_금액']]
    a = s.merge(e, on=['상권_코드', '서비스_업종_코드'], how='left', validate='one_to_one')
    a['v'] = np.where(a.점포_수 > 0, a.당월_매출_금액 / a.점포_수, np.nan)
    return a

def store_sales(Y):
    a = sales_year(Y); d = a.dropna(subset=['v'])
    key = M.td_sales.astype('Int64').astype('string') + '|' + M.category_code.astype('string')
    direct = key.map(dict(zip(d.상권_코드 + '|' + d.서비스_업종_코드, d.v)))
    out = direct.astype(float).copy()
    ref = d.merge(cent, left_on='상권_코드', right_on='TRDAR_CD')
    miss = M.td_sales.notna() & M.category_code.notna() & out.isna()
    for i in np.flatnonzero(miss.values):
        r = ref[ref.서비스_업종_코드 == M.category_code.iloc[i]]
        if r.empty: continue
        dist = gpd.GeoSeries(r.geometry.values).distance(M.geom.iloc[i]).values; v = r.v.values
        o = np.argsort(dist); o = o[dist[o] <= 1000][:3]
        if len(o) == 0: continue
        dd, vv = dist[o], v[o]
        out.iloc[i] = vv[np.argmin(dd)] if (dd == 0).any() else (np.average(vv, weights=1 / dd) if len(o) >= 2 else vv[0])
    return out.values

SALES = {Y: store_sales(Y) for Y in YEARS}
chk = M.in_eh.values
v0 = SALES[2021][chk]; v1 = M.area_category_sales_per_store_final.values[chk]
ok = (np.isnan(v0) & np.isnan(v1)) | np.isclose(np.nan_to_num(v0, nan=-1), np.nan_to_num(v1, nan=-1), rtol=1e-6)
print(f'[3] 2021 매출 재현: 2021 코호트 점포 {chk.sum()}곳 중 일치 {ok.mean():.1%} (불일치는 모두 공식 상권 밖 점포의 보간값 → 2021 코호트는 저장소 값 사용)')
SALES[2021] = np.where(chk, v1_full := M.area_category_sales_per_store_final.values, SALES[2021])

# ---------- 4. 코호트별 데이터셋
def cohort(Y):
    b = B[Y]
    d = M[(M.open <= b) & (M.close.isna() | (M.close >= b))].copy()
    if Y == 2021: d = d[d.in_eh]          # 2021 코호트는 3단계 표본(3,775곳)과 동일하게 유지, 폐업 갱신만 반영
    d['event'] = (d.close.notna() & (d.close >= b) & (d.close <= END)).astype(int)
    d['observation_end'] = d.close.where(d.event == 1, END)
    d['duration_days'] = (d.observation_end - b).dt.days
    d['business_age_years'] = (b - d.open).dt.days / 365.25
    d['area_sales_per_store'] = pd.Series(SALES[Y], index=M.index).loc[d.index]
    d = d.rename(columns={'사업장명': 'store_name', 'open': 'license_date', 'close': 'closure_date', '지번주소': 'address_jibun'})
    d = d[['restaurant_id', 'store_name', 'license_date', 'closure_date', 'address_jibun', 'longitude', 'latitude', 'category_code', 'category_name', 'trdar_code', 'trdar_name',
           'area_sales_per_store', 'dong', 'rent_median', 'business_age_years', 'event', 'observation_end', 'duration_days']].reset_index(drop=True)
    d['trdar_code'] = d['trdar_code'].astype('Int64')
    d = bf.add_rent_level(d)
    dist = bf.pairwise_distance(d)
    cat = d['category_code'].fillna(bf.UNMAPPED).to_numpy(dtype=object)
    same = cat[:, None] == cat[None, :]
    d = bf.add_radius_competition(d, dist, same)
    d = bf.add_network(d, dist, same)
    return d

for Y in YEARS:
    print(f'\n[4] 코호트 {Y}-04-01')
    d = cohort(Y)
    d.to_csv(os.path.join(OUT, f'cohort_{Y}_end{TAG}.csv'), index=False, encoding='utf-8-sig')
    print(f'    n={len(d)}, 폐업 {int(d.event.sum())} ({d.event.mean():.1%}), 중도절단 {int((1 - d.event).sum())}, 평균 업력 {d.business_age_years.mean():.1f}년, 평균 이웃(300m) {d.adjacent_biz_300m.mean():.0f}')
    if Y == 2021 and END == pd.Timestamp('2026-09-12'):
        F = pd.read_csv(os.path.join(SNAP, 'final_dataset_with_reviews.csv'), encoding='utf-8-sig'); F['restaurant_id'] = F['restaurant_id'].astype(str)
        mm = d.merge(F, on='restaurant_id', suffixes=('', '_eh'))
        print(f'    [검증] 9/12 기준 3단계 결과와 점포 {len(mm)}/{len(F)} 일치(교집합), 내 코호트 {len(d)}곳')
        for c in ['event', 'duration_days', 'competitor_c_100m', 'adjacent_biz_300m', 'competitor_c_300m', 'adjacent_biz_500m', 'net_strength_200m', 'rent_level', 'area_sales_per_store', 'category_code']:
            a, b2 = mm[c], mm[c + '_eh']
            eq = ((a == b2) | (a.isna() & b2.isna())) if a.dtype == object or str(a.dtype).startswith('string') else (np.isclose(a.astype(float), b2.astype(float), rtol=1e-4, equal_nan=True))
            print(f'      {c}: 일치 {eq.mean():.2%}')
