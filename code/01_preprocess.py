# -*- coding: utf-8 -*-

# %% [markdown]
# # 1단계: 성동구 일반음식점 생존분석 데이터 전처리
#
# 입력  data/raw/          인허가·업태 매핑·상권 영역(shp)·점포/추정매출(2021)·상가임대료
#       data/interim/geocode_cache.csv   좌표 결측 점포의 Kakao 지오코딩 결과(재실행 시 API 호출 생략)
# 출력  data/interim/final_restaurant_level_dataset.csv  (점포 1행, 3,775행)
# 실행  python code/01_preprocess.py   → 다음 단계: code/02_attach_reviews.py
#
# 이 스크립트는 **2021년 1분기(Q1)의 입지·상권 환경을 기준 변수로 사용**하고,
# **2021-04-01부터 2026-10-13까지 음식점 폐업을 추적**하기 위한 음식점 단위 분석 데이터를 구축한다.
#
# ### 처리 순서
# 1. 일반음식점 인허가정보 로드
# 2. Baseline 시점 영업 음식점 추출
# 3. 서울시 서비스 업종 매핑
# 4. 좌표 정리 및 결측 좌표 지오코딩
# 5. 공식 상권 Spatial Join
# 6. 2021 Q1 점포·추정매출 데이터 정리
# 7. 상권×업종 점포당 추정매출 계산
# 8. 음식점에 상권 매출정보 연결
# 9. 매출환경 결측 보완
# 10. 임대료 변수 생성
# 11. 대중교통 변수 — 본 분석에서는 보류
# 12. 경쟁·집적 변수 생성
# 13. Survival 변수 생성
# 14. 음식점당 1행 최종 데이터 저장

# %% [markdown]
# ## 0. 환경 설정

# %%
from pathlib import Path
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

from pyproj import Transformer

import zipfile

# 경로: 이 파일(code/) 기준으로 프로젝트 루트를 찾는다 → 어디서 실행해도 동작
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SEONGDONG_RAW_DIR", ROOT / "data" / "raw"))   # 원자료 (csv 또는 csv.zip)
INTERIM_DIR = ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
GEOCODE_CACHE = INTERIM_DIR / "geocode_cache.csv"   # Kakao 지오코딩 결과 캐시

BASELINE_DATE = pd.Timestamp("2021-04-01")
# 관찰 종료일 = 인허가 데이터 추출일(최종수정일자 최댓값). 2026년 10월 추출본을 써서 네이버 리뷰 수집 시점(2026-10)과 맞췄다.
STUDY_END_DATE = pd.Timestamp("2026-10-13")

# 7주차 보고서(논문 초안)와 같은 전처리 설정 (관찰 종료일만 2026-09-12 → 2026-10-13으로 연장)
#  - 업태 매핑: 보고서 분석 당시 매핑표에는 아래 5개 업태가 없었다(이후 추가). 보고서의 '업종 미분류 780곳'은 이 기준.
#  - 상권 매출 보간: 보고서 데이터는 공식 상권 밖 점포에도 1km 이내 동일 업종 상권 매출을 IDW로 보간했다.
#  PAPER_VERSION = False 로 바꾸면 최신 매핑표를 쓰고, 상권 밖 점포의 매출은 보간하지 않는다(결과 수치가 달라짐).
PAPER_VERSION = True
MAPPING_ADDED_AFTER_PAPER = ["통닭(치킨)", "외국음식전문점(인도,태국등)", "식육(숯불구이)", "김밥(도시락)", "탕류(보신용)"]
IMPUTE_SALES_OUTSIDE_AREA = PAPER_VERSION

def normalized(text):
    return unicodedata.normalize("NFC", str(text))

def find_file(*keywords, suffix=None):
    """DATA_DIR에서 파일명을 NFC 정규화한 뒤 키워드가 모두 포함된 파일 1개를 찾는다.
    suffix='.csv'이면 압축 파일(.csv.zip)도 찾는다."""
    matches = []
    for p in DATA_DIR.iterdir():
        name = normalized(p.name)
        if suffix is not None and not (name.lower().endswith(suffix.lower())
                                       or name.lower().endswith(suffix.lower() + ".zip")):
            continue
        if all(keyword in name for keyword in keywords):
            matches.append(p)

    # 같은 파일의 압축본(.csv.zip)과 압축을 푼 파일(.csv)이 함께 있으면 압축을 푼 파일을 쓴다
    unzipped = {normalized(p.name) for p in matches if p.suffix.lower() != ".zip"}
    matches = [p for p in matches if not (p.suffix.lower() == ".zip" and normalized(p.name)[:-4] in unzipped)]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"검색 조건 {keywords}, suffix={suffix}에 해당하는 파일이 1개여야 합니다. "
            f"현재 검색 결과: {[p.name for p in matches]}"
        )
    return matches[0]

def read_raw_csv(path, **kwargs):
    """csv 또는 csv 1개가 든 zip을 읽는다 (zip 안의 __MACOSX 항목은 무시)."""
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX") and m.lower().endswith(".csv")]
        assert len(members) == 1, f"{path.name}: csv가 1개여야 합니다 {members}"
        with zf.open(members[0]) as f:
            return pd.read_csv(f, **kwargs)

print("DATA_DIR:", DATA_DIR)

# %% [markdown]
# ## 1. 일반음식점 인허가정보 → 기준 데이터

# %%
# '인허가 정보'로 찾아야 6단계용 갱신본(성동구_일반음식점_인허가_갱신_20261019.csv)과 겹치지 않는다
restaurant_path = find_file("성동구", "일반음식점", "인허가 정보", suffix=".csv")
mapping_path = find_file("성동구_음식점_업태_서울상권서비스업종_매핑", suffix=".csv")

restaurant = read_raw_csv(
    restaurant_path,
    encoding="cp949",
    low_memory=False
)

mapping = read_raw_csv(
    mapping_path,
    encoding="utf-8-sig"
)

restaurant["인허가일자"] = pd.to_datetime(
    restaurant["인허가일자"], errors="coerce"
)
restaurant["폐업일자"] = pd.to_datetime(
    restaurant["폐업일자"], errors="coerce"
)

restaurant["restaurant_id"] = restaurant["관리번호"].astype(str)

print("인허가 데이터:", restaurant.shape)
print("Mapping Table:", mapping.shape)
print("관리번호 중복:", restaurant["restaurant_id"].duplicated().sum())
print("인허가일 결측:", restaurant["인허가일자"].isna().sum())

# 관찰 종료일이 인허가 데이터 추출 시점과 맞는지 확인 (다른 시점의 인허가 파일로 바꾸면 STUDY_END_DATE도 바꿔야 한다)
last_update = pd.to_datetime(restaurant["최종수정일자"], errors="coerce").max().normalize()
print("인허가 데이터 최종수정일:", last_update.date(), "| 관찰 종료일:", STUDY_END_DATE.date())
assert restaurant["폐업일자"].max() <= STUDY_END_DATE, "관찰 종료일 이후의 폐업일이 있습니다. STUDY_END_DATE를 확인하세요."
if last_update != STUDY_END_DATE:
    print("  ※ 관찰 종료일과 인허가 데이터 추출일이 다릅니다. 새 인허가 파일이면 STUDY_END_DATE를 바꾸세요.")

# %% [markdown]
# ## 2. Baseline 시점 영업 음식점 추출
#
# 2021 Q1의 환경변수를 사용하므로 추적 시작일은 **2021-04-01**로 둔다.
# 이 단계에서는 baseline 시점에 영업 중이던 음식점만 선정하며,
# `event`와 `duration`은 모든 설명변수를 만든 뒤 13절에서 생성한다.

# %%
baseline_df = restaurant[
    (restaurant["인허가일자"] <= BASELINE_DATE)
    &
    (
        restaurant["폐업일자"].isna()
        | (restaurant["폐업일자"] >= BASELINE_DATE)
    )
].copy()

print("전체 인허가 데이터:", len(restaurant))
print("Baseline 음식점:", len(baseline_df))
print("restaurant_id 중복:", baseline_df["restaurant_id"].duplicated().sum())
print(
    "Baseline 이전 폐업:",
    (
        baseline_df["폐업일자"].notna()
        & (baseline_df["폐업일자"] < BASELINE_DATE)
    ).sum()
)

# %% [markdown]
# ## 3. 업태 Mapping Table 병합

# %%
if PAPER_VERSION:
    mapping = mapping[~mapping["업태구분명"].isin(MAPPING_ADDED_AFTER_PAPER)]

baseline_df = baseline_df.merge(
    mapping,
    on="업태구분명",
    how="left",
    validate="many_to_one"
)

mapped = baseline_df["서울시_서비스_업종_코드"].notna()

print("병합 후 음식점:", len(baseline_df))
print("업종 매핑 성공:", mapped.sum())
print("업종 매핑 실패:", (~mapped).sum())
print("매핑 성공률:", round(mapped.mean() * 100, 2), "%")

# %% [markdown]
# ## 4. 음식점 좌표 정리

# %%
baseline_df["좌표정보(X)"] = pd.to_numeric(
    baseline_df["좌표정보(X)"], errors="coerce"
)
baseline_df["좌표정보(Y)"] = pd.to_numeric(
    baseline_df["좌표정보(Y)"], errors="coerce"
)

def clean_address(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return np.nan if x == "" else x

baseline_df["도로명주소_clean"] = baseline_df["도로명주소"].apply(clean_address)
baseline_df["지번주소_clean"] = baseline_df["지번주소"].apply(clean_address)

missing_coord = baseline_df[
    baseline_df["좌표정보(X)"].isna()
    | baseline_df["좌표정보(Y)"].isna()
].copy()

print("Baseline 음식점:", len(baseline_df))
print("좌표 결측:", len(missing_coord))
print(
    "좌표 결측 중 주소 존재:",
    (
        missing_coord["도로명주소_clean"].notna()
        | missing_coord["지번주소_clean"].notna()
    ).sum()
)

# %% [markdown]
# ### 4-1. 결측 좌표 Kakao 지오코딩
#
# API 키는 제출 파일에 직접 저장하지 않고 환경변수 `KAKAO_REST_API_KEY`에서 읽는다.
# 이미 원본 좌표가 있는 음식점에는 API를 호출하지 않는다.

# %%
def simplify_address(address):
    if address is None or pd.isna(address):
        return None

    address = str(address).strip()
    if address == "":
        return None

    address = re.sub(r"\s*\(.*$", "", address)
    address = address.split(",")[0].strip()
    return address


def geocode_kakao(address, api_key):
    if address is None or pd.isna(address):
        return None, None

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"query": address},
            timeout=10
        )

        if response.status_code != 200:
            return None, None

        documents = response.json().get("documents", [])
        if not documents:
            return None, None

        return float(documents[0]["x"]), float(documents[0]["y"])

    except requests.RequestException:
        return None, None


def geocode_restaurant(row, api_key):
    candidates = [
        ("road_original", row["도로명주소_clean"]),
        ("road_simple", simplify_address(row["도로명주소_clean"])),
        ("jibun_original", row["지번주소_clean"]),
        ("jibun_simple", simplify_address(row["지번주소_clean"]))
    ]

    tried = set()

    for method, address in candidates:
        if address is None or pd.isna(address) or address in tried:
            continue

        tried.add(address)
        lon, lat = geocode_kakao(address, api_key)

        if lon is not None:
            return lon, lat, method, address

    return None, None, "failed", None

# %%
# API 키는 코드에 적지 않고 환경변수에서 읽는다:  export KAKAO_REST_API_KEY="..."
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")

missing_ids = baseline_df.loc[
    baseline_df["좌표정보(X)"].isna() | baseline_df["좌표정보(Y)"].isna(), "restaurant_id"
]

# 이전에 지오코딩한 결과(캐시)가 있으면 재사용하고, 캐시에 없는 점포만 API를 호출한다
if GEOCODE_CACHE.exists():
    cache = pd.read_csv(GEOCODE_CACHE, encoding="utf-8-sig", dtype={"restaurant_id": str})
else:
    cache = pd.DataFrame(columns=["restaurant_id", "longitude", "latitude",
                                  "geocode_method", "geocode_used_address"])
to_query = missing_ids[~missing_ids.isin(cache["restaurant_id"])]
print(f"좌표 결측 {len(missing_ids):,}개 중 캐시 사용 {len(missing_ids) - len(to_query):,}개, API 호출 {len(to_query):,}개")

if len(to_query) > 0 and not KAKAO_REST_API_KEY:
    print(f"  ※ API 키가 없어 {len(to_query)}개는 지오코딩하지 않고 좌표 결측으로 둡니다. "
          "(지오코딩하려면 export KAKAO_REST_API_KEY=\"...\" 후 다시 실행)")
    to_query = to_query.iloc[:0]

results = []
for count, (idx, rid) in enumerate(to_query.items(), start=1):
    lon, lat, method, used_address = geocode_restaurant(baseline_df.loc[idx], KAKAO_REST_API_KEY)
    results.append({"restaurant_id": rid, "longitude": lon, "latitude": lat,
                    "geocode_method": method, "geocode_used_address": used_address})
    if count % 100 == 0:
        print(f"{count} / {len(to_query)} 완료")
    time.sleep(0.05)

if results:
    cache = pd.concat([cache, pd.DataFrame(results)], ignore_index=True)
    cache.to_csv(GEOCODE_CACHE, index=False, encoding="utf-8-sig")

geocode_results = (
    baseline_df.loc[missing_ids.index, ["restaurant_id"]]
    .reset_index()
    .merge(cache, on="restaurant_id", how="left")
    .set_index("index")
    .drop(columns="restaurant_id")
)
for c in ["longitude", "latitude"]:
    geocode_results[c] = pd.to_numeric(geocode_results[c], errors="coerce")

# %%
baseline_df["kakao_longitude"] = np.nan
baseline_df["kakao_latitude"] = np.nan
baseline_df["geocode_method"] = pd.NA
baseline_df["geocode_used_address"] = pd.NA

if len(geocode_results) > 0:
    baseline_df.loc[
        geocode_results.index, "kakao_longitude"
    ] = geocode_results["longitude"]

    baseline_df.loc[
        geocode_results.index, "kakao_latitude"
    ] = geocode_results["latitude"]

    baseline_df.loc[
        geocode_results.index, "geocode_method"
    ] = geocode_results["geocode_method"]

    baseline_df.loc[
        geocode_results.index, "geocode_used_address"
    ] = geocode_results["geocode_used_address"]

# 원자료 좌표는 EPSG:5174 → WGS84 변환
transformer = Transformer.from_crs(
    "EPSG:5174", "EPSG:4326", always_xy=True
)

has_original_coord = (
    baseline_df["좌표정보(X)"].notna()
    & baseline_df["좌표정보(Y)"].notna()
)

lon, lat = transformer.transform(
    baseline_df.loc[has_original_coord, "좌표정보(X)"].to_numpy(),
    baseline_df.loc[has_original_coord, "좌표정보(Y)"].to_numpy()
)

baseline_df.loc[has_original_coord, "original_longitude"] = lon
baseline_df.loc[has_original_coord, "original_latitude"] = lat

baseline_df["longitude"] = baseline_df["original_longitude"].fillna(
    baseline_df["kakao_longitude"]
)
baseline_df["latitude"] = baseline_df["original_latitude"].fillna(
    baseline_df["kakao_latitude"]
)

baseline_df["coordinate_source"] = np.select(
    [
        baseline_df["original_longitude"].notna(),
        baseline_df["kakao_longitude"].notna()
    ],
    ["original", "kakao_geocoded"],
    default="missing"
)

print(baseline_df["coordinate_source"].value_counts(dropna=False))
print("\n최종 좌표 결측:",
      (baseline_df["longitude"].isna() | baseline_df["latitude"].isna()).sum())

# %% [markdown]
# ## 5. 공식 상권코드 Spatial Join

# %%
shp_path = find_file("상권분석서비스", "영역-상권", suffix=".shp")
commercial = gpd.read_file(shp_path)

seongdong_commercial = commercial[
    commercial["SIGNGU_CD"].astype(str) == "11200"
].copy()

restaurant_geo = baseline_df[
    baseline_df["longitude"].notna()
    & baseline_df["latitude"].notna()
].copy()

restaurant_geo = gpd.GeoDataFrame(
    restaurant_geo,
    geometry=gpd.points_from_xy(
        restaurant_geo["longitude"],
        restaurant_geo["latitude"]
    ),
    crs="EPSG:4326"
).to_crs(commercial.crs)

restaurant_joined = gpd.sjoin(
    restaurant_geo,
    seongdong_commercial[
        [
            "TRDAR_SE_C",
            "TRDAR_SE_1",
            "TRDAR_CD",
            "TRDAR_CD_N",
            "geometry"
        ]
    ],
    how="left",
    predicate="within"
)

print("전체 상권:", len(commercial))
print("성동구 상권:", len(seongdong_commercial))
print("좌표 보유 음식점:", len(restaurant_geo))
print("공식 상권 내부:", restaurant_joined["TRDAR_CD"].notna().sum())
print("공식 상권 밖:", restaurant_joined["TRDAR_CD"].isna().sum())
print("restaurant_id 중복:",
      restaurant_joined["restaurant_id"].duplicated().sum())

# %% [markdown]
# ### 5-1. 공식 상권 밖 음식점 거리 진단
#
# 상권 밖 음식점은 임의로 공식 상권 내부로 편입하지 않는다.
# 다만 가장 가까운 공식 상권까지의 거리를 진단용으로 계산한다.

# %%
outside = restaurant_joined[
    restaurant_joined["TRDAR_CD"].isna()
].drop(columns=["index_right"], errors="ignore").copy()

nearest = gpd.sjoin_nearest(
    outside,
    seongdong_commercial[
        ["TRDAR_CD", "TRDAR_CD_N", "TRDAR_SE_1", "geometry"]
    ],
    how="left",
    distance_col="distance_to_market"
)

nearest_unique = (
    nearest
    .sort_values(["restaurant_id", "distance_to_market"])
    .drop_duplicates("restaurant_id", keep="first")
    .copy()
)

print("공식 상권 밖:", len(outside))
print("최근접 결과(중복 제거):", len(nearest_unique))
print(nearest_unique["distance_to_market"].describe())

# %% [markdown]
# ## 6. 2021 Q1 점포-상권 + 추정매출-상권 병합

# %%
store_path = find_file("점포-상권", "2021", suffix=".csv")
sales_path = find_file("추정매출-상권", "2021", suffix=".csv")

store_df = read_raw_csv(
    store_path, encoding="cp949", low_memory=False
)
sales_df = read_raw_csv(
    sales_path, encoding="cp949", low_memory=False
)

store_q1 = store_df[
    store_df["기준_년분기_코드"] == 20211
].copy()
sales_q1 = sales_df[
    sales_df["기준_년분기_코드"] == 20211
].copy()

food_codes = [
    "CS100001", "CS100002", "CS100003", "CS100004", "CS100005",
    "CS100006", "CS100007", "CS100008", "CS100009", "CS100010"
]

seongdong_market_codes = set(
    seongdong_commercial["TRDAR_CD"].astype(str)
)

for df in [store_q1, sales_q1]:
    df["상권_코드"] = df["상권_코드"].astype(str)
    df["서비스_업종_코드"] = df["서비스_업종_코드"].astype(str)

store_sd_food = store_q1[
    store_q1["상권_코드"].isin(seongdong_market_codes)
    & store_q1["서비스_업종_코드"].isin(food_codes)
].copy()

sales_sd_food = sales_q1[
    sales_q1["상권_코드"].isin(seongdong_market_codes)
    & sales_q1["서비스_업종_코드"].isin(food_codes)
].copy()

print("2021 Q1 점포:", store_sd_food.shape)
print("2021 Q1 추정매출:", sales_sd_food.shape)
print("점포 key 중복:",
      store_sd_food.duplicated(["상권_코드", "서비스_업종_코드"]).sum())
print("매출 key 중복:",
      sales_sd_food.duplicated(["상권_코드", "서비스_업종_코드"]).sum())

# %% [markdown]
# ## 7. 상권×업종 점포당 추정매출 계산

# %%
store_columns = [
    "상권_코드", "상권_코드_명",
    "서비스_업종_코드", "서비스_업종_코드_명",
    "점포_수", "유사_업종_점포_수",
    "개업_율", "개업_점포_수",
    "폐업_률", "폐업_점포_수",
    "프랜차이즈_점포_수"
]

sales_columns = [
    "상권_코드", "서비스_업종_코드",
    "당월_매출_금액", "당월_매출_건수"
]

area_q1 = (
    store_sd_food[store_columns]
    .merge(
        sales_sd_food[sales_columns],
        on=["상권_코드", "서비스_업종_코드"],
        how="left",
        validate="one_to_one"
    )
)

area_q1["area_category_sales_per_store"] = np.where(
    area_q1["점포_수"] > 0,
    area_q1["당월_매출_금액"] / area_q1["점포_수"],
    np.nan
)

print("상권×업종 조합:", len(area_q1))
print(
    "점포당 추정매출 관측:",
    area_q1["area_category_sales_per_store"].notna().sum()
)

# %% [markdown]
# ## 8. 음식점에 상권 매출정보 연결

# %%
restaurant_joined["TRDAR_CD"] = (
    restaurant_joined["TRDAR_CD"].astype("string")
)
restaurant_joined["서울시_서비스_업종_코드"] = (
    restaurant_joined["서울시_서비스_업종_코드"].astype("string")
)

area_q1["상권_코드"] = area_q1["상권_코드"].astype("string")
area_q1["서비스_업종_코드"] = (
    area_q1["서비스_업종_코드"].astype("string")
)

area_variables = area_q1[
    [
        "상권_코드", "서비스_업종_코드",
        "점포_수", "유사_업종_점포_수",
        "개업_율", "개업_점포_수",
        "폐업_률", "폐업_점포_수",
        "프랜차이즈_점포_수",
        "당월_매출_금액", "당월_매출_건수",
        "area_category_sales_per_store"
    ]
].copy()

area_variables = area_variables.rename(columns={
    "상권_코드": "TRDAR_CD",
    "서비스_업종_코드": "서울시_서비스_업종_코드",
    "점포_수": "area_category_store_count",
    "유사_업종_점포_수": "area_similar_store_count",
    "개업_율": "area_open_rate",
    "개업_점포_수": "area_open_count",
    "폐업_률": "area_close_rate",
    "폐업_점포_수": "area_close_count",
    "프랜차이즈_점포_수": "area_franchise_count",
    "당월_매출_금액": "area_category_sales",
    "당월_매출_건수": "area_category_sales_count"
})

restaurant_merged = restaurant_joined.merge(
    area_variables,
    on=["TRDAR_CD", "서울시_서비스_업종_코드"],
    how="left",
    validate="many_to_one"
)

inside = restaurant_merged["TRDAR_CD"].notna()
mapped = restaurant_merged["서울시_서비스_업종_코드"].notna()
direct_sales = restaurant_merged["area_category_sales_per_store"].notna()

print("음식점:", len(restaurant_merged))
print("공식 상권 내부:", inside.sum())
print("업종 매핑 성공:", mapped.sum())
print("Direct 매출환경 관측:", direct_sales.sum())

# %% [markdown]
# ## 9. 매출환경 결측 보완
#
# 업종 매핑이 되었지만 해당 상권×업종의 매출이 없는 경우
# **동일 업종의 매출 관측 상권**(중심점 기준)을 최대 1km 안에서 탐색한다.
#
# - 참조 상권 3개 이상: 가까운 3개 IDW
# - 참조 상권 2개: 2개 IDW
# - 참조 상권 1개: 최근접 1개
# - 1km 이내 참조 없음: 결측 유지
#
# 업종 미매핑 음식점은 매출환경을 추정하지 않는다.
# 공식 상권 밖 음식점은 IMPUTE_SALES_OUTSIDE_AREA = True(보고서 버전)일 때만 같은 방식으로 보간한다.

# %%
# 성동구 55개 상권의 centroid
market_centroids = seongdong_commercial[
    ["TRDAR_CD", "geometry"]
].copy()

market_centroids["TRDAR_CD"] = (
    market_centroids["TRDAR_CD"].astype("string")
)
market_centroids["geometry"] = market_centroids.geometry.centroid

# 실제 점포당 매출이 관측된 상권×업종만 참조 후보로 사용
sales_reference = area_variables[
    area_variables["area_category_sales_per_store"].notna()
].merge(
    market_centroids,
    on="TRDAR_CD",
    how="left",
    validate="many_to_one"
)

sales_reference = gpd.GeoDataFrame(
    sales_reference,
    geometry="geometry",
    crs=seongdong_commercial.crs
)

restaurant_merged["area_category_sales_per_store_final"] = (
    restaurant_merged["area_category_sales_per_store"]
)
restaurant_merged["sales_source"] = np.where(
    restaurant_merged["area_category_sales_per_store"].notna(),
    "direct",
    pd.NA
)

missing_mask = (
    (restaurant_merged["TRDAR_CD"].notna() | IMPUTE_SALES_OUTSIDE_AREA)
    & restaurant_merged["서울시_서비스_업종_코드"].notna()
    & restaurant_merged["area_category_sales_per_store"].isna()
)

for idx, row in restaurant_merged.loc[missing_mask].iterrows():
    category = row["서울시_서비스_업종_코드"]

    refs = sales_reference[
        sales_reference["서울시_서비스_업종_코드"] == category
    ].copy()

    if refs.empty:
        continue

    distances = refs.geometry.distance(row.geometry)
    refs = refs.assign(distance=distances).query("distance <= 1000")
    refs = refs.sort_values("distance").head(3)

    if len(refs) == 0:
        continue

    values = refs["area_category_sales_per_store"].to_numpy(dtype=float)
    d = refs["distance"].to_numpy(dtype=float)

    # 동일 위치가 있으면 그 값을 그대로 사용
    if np.any(d == 0):
        estimate = values[np.argmin(d)]
    elif len(refs) >= 2:
        weights = 1 / d
        estimate = np.average(values, weights=weights)
    else:
        estimate = values[0]

    if len(refs) >= 3:
        method = "idw_3"
    elif len(refs) == 2:
        method = "idw_2"
    else:
        method = "nearest_1"

    restaurant_merged.at[
        idx, "area_category_sales_per_store_final"
    ] = estimate
    restaurant_merged.at[idx, "sales_source"] = method

restaurant_merged.loc[
    missing_mask
    & restaurant_merged["area_category_sales_per_store_final"].isna(),
    "sales_source"
] = "unavailable"

print(
    restaurant_merged.loc[missing_mask, "sales_source"]
    .value_counts(dropna=False)
)
print(
    "최종 매출환경 관측:",
    restaurant_merged["area_category_sales_per_store_final"]
    .notna().sum()
)

# %%
# 업종별 direct 매출 분포의 20/40/60/80 분위수를 기준으로 5단계 구간 생성
quantiles = (
    area_q1
    .dropna(subset=["area_category_sales_per_store"])
    .groupby("서비스_업종_코드")["area_category_sales_per_store"]
    .quantile([0.2, 0.4, 0.6, 0.8])
    .unstack()
)

quantiles.columns = ["q20", "q40", "q60", "q80"]

def classify_sales_level(row):
    value = row["area_category_sales_per_store_final"]
    category = row["서울시_서비스_업종_코드"]

    if pd.isna(value) or pd.isna(category) or category not in quantiles.index:
        return pd.NA

    q20, q40, q60, q80 = quantiles.loc[
        category, ["q20", "q40", "q60", "q80"]
    ]

    if value <= q20:
        return "하"
    if value <= q40:
        return "중하"
    if value <= q60:
        return "중"
    if value <= q80:
        return "중상"
    return "상"

restaurant_merged["sales_level_5"] = restaurant_merged.apply(
    classify_sales_level,
    axis=1
)

restaurant_merged["sales_missing_reason"] = "available"

restaurant_merged.loc[
    restaurant_merged["TRDAR_CD"].isna()
    & restaurant_merged["area_category_sales_per_store_final"].isna(),
    "sales_missing_reason"
] = "outside_commercial_area"

restaurant_merged.loc[
    restaurant_merged["서울시_서비스_업종_코드"].isna(),
    "sales_missing_reason"
] = "category_unmapped"

restaurant_merged.loc[
    missing_mask
    & restaurant_merged["area_category_sales_per_store_final"].isna(),
    "sales_missing_reason"
] = "no_reference_within_1km"

print(restaurant_merged["sales_missing_reason"].value_counts())
print("\n매출수준:")
print(restaurant_merged["sales_level_5"].value_counts(dropna=False))

# %% [markdown]
# ## 10. 임대료 변수
#
# 교수님 피드백에 따라 임대료는 핵심 통제변수로 사용한다.
# 2021 Q1의 성동구 임대료 자료를 **동 단위로 집계**하여 음식점 주소의 법정동과 연결한다.
# 연속형 평균 임대료와 함께 해석이 쉬운 **상·중·하 3단계 변수**도 생성한다.

# %%
rent_path = find_file("성동구", "상가임대료", suffix=".csv")

rent = read_raw_csv(
    rent_path,
    encoding="cp949",
    low_memory=False
)

rent["기준년월"] = pd.to_datetime(
    rent["기준년월"], errors="coerce"
)
rent["평당전세가"] = pd.to_numeric(
    rent["평당전세가"], errors="coerce"
)

rent_q1 = rent[
    rent["기준년월"].between(
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-03-31")
    )
].copy()

rent_dong_summary = (
    rent_q1
    .groupby("읍면동")["평당전세가"]
    .agg(
        rent_mean="mean",
        rent_median="median",
        rent_count="count"
    )
    .reset_index()
    .rename(columns={"읍면동": "rent_dong"})
)

# 동별 평균 임대료를 동일 가중으로 3분위 분류
rent_dong_summary["rent_level_3"] = pd.qcut(
    rent_dong_summary["rent_mean"].rank(method="first"),
    q=3,
    labels=["하", "중", "상"]
)

def extract_dong(address):
    if pd.isna(address):
        return pd.NA
    match = re.search(
        r"([가-힣]+(?:동\d*가|동|가))(?=\s|\d|-|$)",
        str(address)
    )
    return match.group(1) if match else pd.NA

restaurant_merged["rent_dong"] = (
    restaurant_merged["지번주소"].apply(extract_dong)
)

restaurant_merged = restaurant_merged.merge(
    rent_dong_summary,
    on="rent_dong",
    how="left",
    validate="many_to_one"
)

print("임대료 매칭 성공:", restaurant_merged["rent_mean"].notna().sum())
print("임대료 매칭 실패:", restaurant_merged["rent_mean"].isna().sum())
print(restaurant_merged["rent_level_3"].value_counts(dropna=False))

# %% [markdown]
# ## 11. 지하철/버스 반경 변수
#
# 대중교통 접근성 변수는 추가 데이터 검토 후 확장할 수 있도록 본 제출본에서는 보류한다.
# 현재 분석에서는 임대료가 입지의 종합적인 시장가치를 반영하는 핵심 통제변수라는 교수님 피드백을 우선 반영한다.

# %% [markdown]
# ## 12. 경쟁 및 집적 변수
#
# 각 음식점을 중심으로 300m와 500m 반경 안의 음식점 수를 계산한다.
#
# - `restaurants_all_*`: 주변 전체 음식점 수
# - `competitors_same_*`: 동일 서울시 서비스 업종 음식점 수
#
# 자기 자신은 경쟁 음식점 수에서 제외한다.

# %%
# restaurant_merged는 이미 EPSG:5181 GeoDataFrame
points = restaurant_merged[
    ["restaurant_id", "서울시_서비스_업종_코드", "geometry"]
].copy()

points = gpd.GeoDataFrame(
    points,
    geometry="geometry",
    crs=restaurant_merged.crs
)

left = points[
    ["restaurant_id", "서울시_서비스_업종_코드", "geometry"]
].rename(columns={
    "restaurant_id": "restaurant_id_left",
    "서울시_서비스_업종_코드": "category_left"
})

right = points[
    ["restaurant_id", "서울시_서비스_업종_코드", "geometry"]
].rename(columns={
    "restaurant_id": "restaurant_id_right",
    "서울시_서비스_업종_코드": "category_right"
})

# 최대 반경 500m 후보 쌍 생성
buffers = left.copy()
buffers["geometry"] = buffers.geometry.buffer(500)

pairs = gpd.sjoin(
    buffers,
    right,
    how="left",
    predicate="intersects"
)

# 자기 자신 제외
pairs = pairs[
    pairs["restaurant_id_left"] != pairs["restaurant_id_right"]
].copy()

# 실제 점-점 거리 계산
left_geom = points.set_index("restaurant_id")["geometry"]
right_geom = points.set_index("restaurant_id")["geometry"]

pairs["distance_m"] = [
    left_geom.loc[lid].distance(right_geom.loc[rid])
    for lid, rid in zip(
        pairs["restaurant_id_left"],
        pairs["restaurant_id_right"]
    )
]

pairs["same_category"] = (
    pairs["category_left"].notna()
    & (pairs["category_left"] == pairs["category_right"])
)

def neighbor_counts(radius):
    subset = pairs[pairs["distance_m"] <= radius]

    all_count = (
        subset.groupby("restaurant_id_left")
        .size()
        .reindex(points["restaurant_id"], fill_value=0)
    )

    same_count = (
        subset[subset["same_category"]]
        .groupby("restaurant_id_left")
        .size()
        .reindex(points["restaurant_id"], fill_value=0)
    )

    return all_count, same_count

all_300, same_300 = neighbor_counts(300)
all_500, same_500 = neighbor_counts(500)

restaurant_merged["restaurants_all_300m"] = (
    restaurant_merged["restaurant_id"].map(all_300).astype(int)
)
restaurant_merged["restaurants_all_500m"] = (
    restaurant_merged["restaurant_id"].map(all_500).astype(int)
)

restaurant_merged["competitors_same_300m"] = (
    restaurant_merged["restaurant_id"].map(same_300).astype("Int64")
)
restaurant_merged["competitors_same_500m"] = (
    restaurant_merged["restaurant_id"].map(same_500).astype("Int64")
)

# 업종 자체가 매핑되지 않은 음식점은 동일업종 경쟁 변수 정의 불가
unmapped = restaurant_merged["서울시_서비스_업종_코드"].isna()
restaurant_merged.loc[
    unmapped,
    ["competitors_same_300m", "competitors_same_500m"]
] = pd.NA

print(
    restaurant_merged[
        [
            "restaurants_all_300m",
            "competitors_same_300m",
            "restaurants_all_500m",
            "competitors_same_500m"
        ]
    ].describe()
)

print(
    "\n500m < 300m 오류:",
    (
        restaurant_merged["restaurants_all_500m"]
        < restaurant_merged["restaurants_all_300m"]
    ).sum()
)

# %% [markdown]
# ## 13. Duration, Event 생성
#
# - 추적 시작: 2021-04-01
# - 관찰 종료: 2026-10-13 (인허가 데이터 추출일)
# - 관찰기간 중 폐업: `event = 1`
# - 관찰 종료일까지 폐업 미관측: `event = 0`
# - `duration`: baseline부터 폐업일 또는 관찰종료일까지의 기간
# - `business_age_at_baseline`: baseline 이전에 이미 영업한 기간

# %%
restaurant_merged["인허가일자"] = pd.to_datetime(
    restaurant_merged["인허가일자"], errors="coerce"
)
restaurant_merged["폐업일자"] = pd.to_datetime(
    restaurant_merged["폐업일자"], errors="coerce"
)

restaurant_merged["event"] = (
    restaurant_merged["폐업일자"].notna()
    & (restaurant_merged["폐업일자"] >= BASELINE_DATE)
    & (restaurant_merged["폐업일자"] <= STUDY_END_DATE)
).astype(int)

restaurant_merged["observation_end"] = (
    restaurant_merged["폐업일자"]
    .where(
        restaurant_merged["event"] == 1,
        STUDY_END_DATE
    )
)

restaurant_merged["duration_days"] = (
    restaurant_merged["observation_end"] - BASELINE_DATE
).dt.days

restaurant_merged["duration_months"] = (
    restaurant_merged["duration_days"] / 30.4375
)

restaurant_merged["business_age_at_baseline_years"] = (
    (BASELINE_DATE - restaurant_merged["인허가일자"]).dt.days
    / 365.25
)

print("분석 대상 음식점:", len(restaurant_merged))
print("\nEvent:")
print(restaurant_merged["event"].value_counts().sort_index())
print("\nDuration:")
print(
    restaurant_merged[
        ["duration_days", "duration_months"]
    ].describe()
)
print("\n음수 duration:",
      (restaurant_merged["duration_days"] < 0).sum())
print("duration 결측:",
      restaurant_merged["duration_days"].isna().sum())

# %%
# Cox 모델에서 duration=0인 관측치가 있는지 별도 확인
zero_duration = restaurant_merged[
    restaurant_merged["duration_days"] == 0
]

print("duration=0 음식점:", len(zero_duration))

if len(zero_duration) > 0:
    print(
        zero_duration[
            [
                "restaurant_id",
                "사업장명",
                "인허가일자",
                "폐업일자",
                "event",
                "duration_days"
            ]
        ]
    )

# 기준일 당일 폐업(duration = 0)한 점포는 생존분석에서 제외한다.
restaurant_merged = restaurant_merged[restaurant_merged["duration_days"] > 0].copy()
print("duration=0 제외 후 음식점:", len(restaurant_merged))

# %% [markdown]
# ## 14. 최종 음식점당 1행 저장

# %%
final_columns = [
    # 식별 정보
    "restaurant_id",
    "사업장명",
    "인허가일자",
    "폐업일자",
    "업태구분명",

    # 위치
    "도로명주소",
    "지번주소",
    "longitude",
    "latitude",

    # 업종
    "서울시_서비스_업종_코드",
    "서울시_서비스_업종_코드_명",

    # 공식 상권
    "TRDAR_CD",
    "TRDAR_CD_N",
    "TRDAR_SE_1",

    # 매출환경: direct 원값 + 보완된 최종값을 모두 보존
    "area_category_sales_per_store",
    "area_category_sales_per_store_final",
    "sales_source",
    "sales_level_5",
    "sales_missing_reason",

    # 임대료
    "rent_dong",
    "rent_mean",
    "rent_median",
    "rent_count",
    "rent_level_3",

    # 경쟁·집적
    "restaurants_all_300m",
    "competitors_same_300m",
    "restaurants_all_500m",
    "competitors_same_500m",

    # Survival
    "business_age_at_baseline_years",
    "event",
    "observation_end",
    "duration_days",
    "duration_months"
]

final_df = restaurant_merged[final_columns].copy()

print("최종 데이터 shape:", final_df.shape)
print("고유 restaurant_id:", final_df["restaurant_id"].nunique())
print("restaurant_id 중복:",
      final_df["restaurant_id"].duplicated().sum())

# %%
check_columns = [
    "서울시_서비스_업종_코드",
    "TRDAR_CD",
    "area_category_sales_per_store_final",
    "rent_mean",
    "restaurants_all_300m",
    "competitors_same_300m",
    "duration_days"
]

missing_summary = pd.DataFrame({
    "missing_count": final_df[check_columns].isna().sum(),
    "missing_percent": (
        final_df[check_columns]
        .isna()
        .mean()
        .mul(100)
        .round(2)
    )
})

print(missing_summary)

assert final_df["restaurant_id"].duplicated().sum() == 0
assert final_df["duration_days"].isna().sum() == 0
assert (final_df["duration_days"] < 0).sum() == 0
assert final_df["rent_mean"].isna().sum() == 0
assert final_df["restaurants_all_300m"].isna().sum() == 0

print("최종 검증 완료")

# %%
output_path = INTERIM_DIR / "final_restaurant_level_dataset.csv"

final_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("저장 완료:", output_path)
