"""
2단계: 전처리 결과에 네이버 리뷰 요약 변수를 점포 ID로 붙인다.

입력  data/interim/final_restaurant_level_dataset.csv   (code/01_preprocess.py 결과)
      data/interim/review_features.csv                  (리뷰 요약 변수, 리뷰가 매칭된 1,395개 점포)
출력  data/interim/restaurants_with_reviews_raw.csv     (code/03_build_features.py 입력)

실행  python code/02_attach_reviews.py   → 다음 단계: code/03_build_features.py

review_features.csv 는 2026년 10월 수집한 네이버 리뷰를 요약한 결과이다.
  (code/review_collection/ 의 naver_crawling_cohort.py → combine_parts.py → review_merge.py 를 거친 최종 값)
리뷰는 수집 시점마다 달라지므로, 다시 크롤링하지 않고 이 파일을 고정된 입력으로 사용한다.

  place_id, match_status, shared_place   네이버 매장 ID, 매칭 상태, 한 매장에 인허가 여러 건 여부
  review_volume                          log(1 + 방문자 리뷰 총수)
  review_segment, review_segment_n       성동구 평균 대비 특히 많은 손님층, 손님층 태그가 있는 리뷰 수
  review_revisit_share                   재방문 리뷰 비율
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
BASE_PATH = INTERIM / "final_restaurant_level_dataset.csv"
REVIEW_PATH = INTERIM / "review_features.csv"
OUT_PATH = INTERIM / "restaurants_with_reviews_raw.csv"


def main() -> None:
    base = pd.read_csv(BASE_PATH, encoding="utf-8-sig", dtype={"restaurant_id": str})
    reviews = pd.read_csv(REVIEW_PATH, encoding="utf-8-sig", dtype={"restaurant_id": str, "place_id": str})
    assert reviews["restaurant_id"].is_unique

    out = base.merge(reviews, on="restaurant_id", how="left", validate="one_to_one")
    out["has_review"] = out["place_id"].notna().astype(int)

    unmatched = set(reviews["restaurant_id"]) - set(base["restaurant_id"])
    print(f"점포 {len(out):,}개 | 리뷰 매칭 {out['has_review'].sum():,}개 ({out['has_review'].mean():.1%})")
    if unmatched:
        print(f"  ※ 리뷰는 있으나 전처리 결과에 없는 점포 {len(unmatched)}개 (좌표 결측 등으로 제외됨)")
    print(pd.crosstab(out["event"], out["has_review"], margins=True).to_string())

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_PATH.relative_to(ROOT)}  ({len(out):,}행 × {out.shape[1]}열)")


if __name__ == "__main__":
    main()
