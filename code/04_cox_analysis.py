"""
4단계 생존분석: Kaplan-Meier → 반경 경쟁 Cox 모형(H1–H3) → 경쟁 네트워크 Cox 모형(H4) → 강건성 검토
논문 표 1(cox_radius_models.csv), 표 2(cox_network_models.csv), 3.1절 본문 수치(KM, 시간가변 계수, 강건성)

입력  data/final_dataset_with_reviews.csv      (code/03_build_features.py 결과)
출력  outputs/tables/*.csv, outputs/figures/*.png, outputs/results_summary.md

실행  python code/04_cox_analysis.py

모형 (모든 모형 공통: 통제 변수 Z + 상권 단위 군집 강건 표준오차, Efron 동점 처리)
  모형 A  h(t) = h0(t)·exp(β1·C + β2·BIZ + γ'Z)              동종 이웃 수 C와 전체 음식점 수 BIZ
  모형 B  h(t) = h0(t)·exp(β1·R + β2·BIZ + γ'Z)              C 대신 동종 비율 R (R = C/BIZ라 따로 추정)
  모형 C  h(t) = h0(t)·exp(β'K + δ'N + γ'Z)                   모형 A + 네트워크 지표 N (H4: 증분 설명력)
  통제 변수 Z: 업력, 임대료 수준(low/mid/high, 기준 low), log 상권 점포당 매출, 업태 더미(기준 한식)
  표본: Kaplan-Meier는 전체 점포, Cox 모형은 상권 매출이 관측된 점포만 사용한다.
        매출 결측(대부분 업종 미분류 점포)은 대입하거나 더미로 처리하지 않고 결측 그대로 두어 분석에서 제외한다.
  단위: C·BIZ 10곳, R 10%p, 네트워크 지표 표준편차 1 (매개중심성은 log 변환 후)
  보완: 비례위험 가정 위반 신호가 있는 Competitor_R은 관찰 기간을 1년 단위로 나눈 시간가변 계수 모형으로 다시 추정한다.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "final_dataset_with_reviews.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
SUMMARY_PATH = ROOT / "outputs" / "results_summary.md"

RADII = [100, 200, 300, 500]
BASELINE = pd.Timestamp("2021-04-01")
STUDY_END = pd.Timestamp("2026-10-13")             # 관찰 종료일 = 인허가 데이터 추출일 (01_preprocess.py와 동일)
REOPEN_DATE = pd.Timestamp("2022-04-18")       # 사회적 거리두기 해제일
BULK_CLOSURE_MIN = 20                          # 하루 20곳 이상 폐업 처리된 날 = 일괄 정리 의심
UNIT = {"competitor_c": 10, "adjacent_biz": 10, "competitor_r": 0.1}
LABEL = {"competitor_c": "Competitor_C", "adjacent_biz": "Adjacent_BIZ", "competitor_r": "Competitor_R",
         "net_strength": "연결강도", "net_betweenness": "매개중심성", "net_clustering": "군집계수"}


# ---------------------------------------------------------------- 데이터 준비
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["closure_date"])
    df["category_name"] = df["category_name"].fillna("미분류")
    df["log_sales"] = np.log(df["area_sales_per_store"])      # 결측은 결측 그대로 둔다
    df["log_rent"] = np.log(df["rent_median"])
    # 군집: 공식 상권 코드, 상권 밖 점포는 법정동 단위로 묶음
    df["cluster"] = np.where(df["trdar_code"].notna(),
                             df["trdar_code"].astype("Int64").astype(str), "out_" + df["dong"])
    return df


def cox_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Cox 분석 표본: 상권 매출(log_sales)이 결측인 점포를 제외하고 회귀용 변수를 만든다."""
    return add_model_vars(df[df["log_sales"].notna()].copy())


def add_model_vars(df: pd.DataFrame) -> pd.DataFrame:
    """회귀용 스케일 변환 (표준화는 주어진 표본 기준)"""
    for r in RADII:
        for v, u in UNIT.items():
            df[f"x_{v}_{r}m"] = df[f"{v}_{r}m"] / u
        df[f"x_net_betweenness_{r}m"] = zscore(np.log1p(df[f"net_betweenness_{r}m"] * 1e4))
        for v in ["net_strength", "net_clustering"]:
            df[f"x_{v}_{r}m"] = zscore(df[f"{v}_{r}m"])
    return df


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def controls(df: pd.DataFrame, rent: str = "level", sales: bool = True) -> pd.DataFrame:
    """통제 변수 행렬. rent: 'level'(범주, 본 모형) | 'log'(연속) | 'dong'(동 고정효과), sales: log 매출 포함 여부"""
    parts = [df[["business_age_years", "log_sales"] if sales else ["business_age_years"]]]
    if rent == "level":
        parts.append(pd.get_dummies(df["rent_level"], prefix="rent").drop(columns="rent_low"))
    elif rent == "log":
        parts.append(df[["log_rent"]])
    elif rent == "dong":
        parts.append(pd.get_dummies(df["dong"], prefix="dong", drop_first=True))
    parts.append(pd.get_dummies(df["category_name"], prefix="cat").drop(columns="cat_한식음식점"))
    X = pd.concat(parts, axis=1).astype(float)
    return X.loc[:, X.std() > 0]


# ---------------------------------------------------------------- Cox 추정
def fit_cox(df: pd.DataFrame, X: pd.DataFrame, robust: bool = True) -> dict:
    groups = pd.factorize(df["cluster"])[0] if robust else None
    res = PHReg(df["duration_days"].to_numpy(), X.to_numpy(), status=df["event"].to_numpy(),
                ties="efron").fit(groups=groups)
    b, se = res.params, res.bse
    coef = pd.DataFrame({"hr": np.exp(b), "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se),
                         "p": res.pvalues}, index=X.columns)
    lp = X.to_numpy() @ b
    return {"coef": coef, "res": res, "llf": res.llf, "k": X.shape[1],
            "c_index": harrell_c(df["duration_days"].to_numpy(), df["event"].to_numpy(), lp),
            "ph_p": ph_test(res, df, X.columns)}


def harrell_c(t: np.ndarray, e: np.ndarray, lp: np.ndarray) -> float:
    num = den = 0.0
    for i in np.flatnonzero(e == 1):
        later = t > t[i]
        den += later.sum()
        num += (lp[i] > lp[later]).sum() + 0.5 * (lp[i] == lp[later]).sum()
    return num / den


def ph_test(res, df: pd.DataFrame, cols) -> pd.Series:
    """비례위험 가정 점검: Schoenfeld 잔차와 log(시간)의 상관 p값"""
    ev = df["event"].to_numpy() == 1
    log_t = np.log(df["duration_days"].to_numpy()[ev])
    resid = res.schoenfeld_residuals[ev]
    out = {}
    for j, c in enumerate(cols):
        ok = ~np.isnan(resid[:, j])
        out[c] = stats.pearsonr(log_t[ok], resid[ok, j])[1]
    return pd.Series(out)


def tidy(fit: dict, keep: list, **meta) -> list:
    rows = []
    for c in keep:
        row = fit["coef"].loc[c].to_dict()
        rows.append({**meta, "variable": c.removeprefix("x_"), **row, "ph_p": fit["ph_p"][c],
                     "c_index": fit["c_index"]})
    return rows


# ---------------------------------------------------------------- 분석 단계
def km_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df["biz200_group"] = pd.qcut(df["adjacent_biz_200m"], 3, labels=["하위", "중위", "상위"])
    rows, curves = [], {}
    for g, d in df.groupby("biz200_group", observed=True):
        sf = SurvfuncRight(d["duration_days"], d["event"])
        curves[g] = sf
        rows.append({"group": g, "n": len(d), "biz200_min": d["adjacent_biz_200m"].min(),
                     "biz200_max": d["adjacent_biz_200m"].max(), "closures": int(d["event"].sum()),
                     "survival_at_end": sf.surv_prob[-1]})
    chi2, p = survdiff(df["duration_days"], df["event"], df["biz200_group"].astype(str))
    table = pd.DataFrame(rows).assign(logrank_chi2=chi2, logrank_p=p)
    plot_km(curves, table)
    return table


def radius_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, fits = [], {}
    for r in RADII:
        specs = {"A": [f"x_competitor_c_{r}m", f"x_adjacent_biz_{r}m"],
                 "B": [f"x_competitor_r_{r}m", f"x_adjacent_biz_{r}m"]}
        for model, cols in specs.items():
            fit = fit_cox(df, pd.concat([df[cols], controls(df)], axis=1))
            fits[(model, r)] = fit
            rows += tidy(fit, cols, model=model, radius_m=r)
    out = pd.DataFrame(rows)
    # 반경 간 비교용: 표준편차 1 증가당 HR
    sd = out.apply(lambda x: df[x["variable"]].std() / UNIT[x["variable"].rsplit("_", 1)[0]], axis=1)
    for c in ["hr", "ci_low", "ci_high"]:
        out[f"{c}_per_sd"] = np.exp(np.log(out[c]) * sd)
    return out, fits


def network_models(df: pd.DataFrame, base_fits: dict) -> pd.DataFrame:
    """H4: 모형 A에 네트워크 지표를 더했을 때의 증분 설명력(우도비 검정, AIC, C-index)"""
    rows = []
    for r in RADII:
        k_cols = [f"x_competitor_c_{r}m", f"x_adjacent_biz_{r}m"]
        n_cols = [f"x_net_strength_{r}m", f"x_net_betweenness_{r}m", f"x_net_clustering_{r}m"]
        base = base_fits[("A", r)]
        fit = fit_cox(df, pd.concat([df[k_cols + n_cols], controls(df)], axis=1))
        lr = 2 * (fit["llf"] - base["llf"])
        lr_p = stats.chi2.sf(lr, df=len(n_cols))
        vif = variance_inflation(df[k_cols + n_cols])
        for row in tidy(fit, k_cols + n_cols, model="C", radius_m=r):
            row.update(vif=vif[row["variable"]], lr_chi2=lr, lr_df=len(n_cols), lr_p=lr_p,
                       aic_base=-2 * base["llf"] + 2 * base["k"], aic=-2 * fit["llf"] + 2 * fit["k"],
                       c_index_base=base["c_index"])
            rows.append(row)
    return pd.DataFrame(rows)


def variance_inflation(X: pd.DataFrame) -> pd.Series:
    inv = np.linalg.inv(np.corrcoef(X.to_numpy(), rowvar=False))
    return pd.Series(np.diag(inv), index=[c.removeprefix("x_") for c in X.columns])


def robustness(df: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """300m 모형 B를 설정을 바꿔 다시 추정. df = Cox 표본, df_all = 매출 결측 점포를 포함한 전체 표본"""
    cols = ["x_competitor_r_300m", "x_adjacent_biz_300m"]
    vc = df_all["closure_date"].value_counts()             # 일괄 폐업일은 전체 점포 기준으로 찾는다
    bulk_dates = vc[vc >= BULK_CLOSURE_MIN].index
    cases = [
        ("본 모형 (임대료 범주형)", df, "level", True, True),
        ("임대료 연속형 log(rent_median)", df, "log", True, True),
        ("동 고정효과 (임대료 대신)", df, "dong", True, True),
        ("매출 변수 제외 (매출 결측 점포 포함 전체)", df_all, "level", True, False),
        ("300m 음식점 수 상위 5% 점포 제외 (초밀집 건물·상가)", df[df["adjacent_biz_300m"] <= df["adjacent_biz_300m"].quantile(0.95)], "level", True, True),
        (f"일괄 폐업일(하루 {BULK_CLOSURE_MIN}곳 이상) 점포 제외", df[~df["closure_date"].isin(bulk_dates)], "level", True, True),
        ("일반 표준오차 (군집 미보정)", df, "level", False, True),
    ]
    rows = []
    for name, d, rent, robust, sales in cases:
        fit = fit_cox(d, pd.concat([d[cols], controls(d, rent, sales)], axis=1), robust=robust)
        for c in cols:
            rows.append({"spec": name, "n": len(d), "events": int(d["event"].sum()),
                         "variable": c.removeprefix("x_"), **fit["coef"].loc[c].to_dict()})
    out = pd.DataFrame(rows)
    out.attrs["bulk_dates"] = [str(x.date()) for x in bulk_dates]
    out.attrs["bulk_n"] = int(df_all["closure_date"].isin(bulk_dates).sum())
    return out


def split_episodes(df: pd.DataFrame, cuts_days: list) -> pd.DataFrame:
    """관찰 기간을 구간으로 나눈 계수과정(counting process) 자료: 점포 × 구간마다 1행, entry/exit/ev"""
    edges = [0, *cuts_days, np.inf]
    parts = []
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        e = df[df["duration_days"] > a].copy()
        e["entry"], e["exit"] = a, np.minimum(e["duration_days"], b)
        e["ev"] = ((e["duration_days"] <= b) & (e["event"] == 1)).astype(int)
        e["period"] = k
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


def time_varying_r(df: pd.DataFrame) -> pd.DataFrame:
    """Competitor_R의 비례위험 가정 보완: 모형 B에서 R의 계수를 관찰 1년 구간마다 따로 추정 (R × 구간)"""
    cuts = [365, 730, 1095, 1460]
    labels = ["1년차", "2년차", "3년차", "4년차", "5년차 이후"]
    ep = split_episodes(df, cuts)
    groups = pd.factorize(ep["cluster"])[0]
    rows = []
    for r in RADII:
        r_col, b_col = f"x_competitor_r_{r}m", f"x_adjacent_biz_{r}m"
        inter = pd.concat([(ep[r_col] * (ep["period"] == k)).rename(f"r_{k}") for k in range(len(labels))], axis=1)
        X = pd.concat([ep[[b_col]], inter, controls(ep)], axis=1).astype(float)
        res = PHReg(ep["exit"].to_numpy(), X.to_numpy(), status=ep["ev"].to_numpy(), entry=ep["entry"].to_numpy(),
                    ties="efron").fit(groups=groups)
        b, se, pv = (pd.Series(v, index=X.columns) for v in (res.params, res.bse, res.pvalues))
        # 계수가 구간마다 같은지 Wald 검정 (H0: 모든 구간 계수 동일)
        k = len(labels)
        idx = [X.columns.get_loc(f"r_{j}") for j in range(k)]
        C = np.zeros((k - 1, X.shape[1]))
        for j in range(k - 1):
            C[j, idx[j]], C[j, idx[j + 1]] = 1, -1
        diff = C @ res.params
        wald = float(diff @ np.linalg.inv(C @ res.cov_params() @ C.T) @ diff)
        wald_p = stats.chi2.sf(wald, k - 1)
        for j, lab in enumerate(labels):
            c = f"r_{j}"
            rows.append({"radius_m": r, "period": lab, "hr": np.exp(b[c]), "ci_low": np.exp(b[c] - 1.96 * se[c]),
                         "ci_high": np.exp(b[c] + 1.96 * se[c]), "p": pv[c], "biz_hr": np.exp(b[b_col]),
                         "equal_wald_chi2": wald, "equal_wald_df": k - 1, "equal_wald_p": wald_p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
def set_font() -> None:
    for path in font_manager.findSystemFonts():
        if any(k in path for k in ["AppleSDGothicNeo", "NotoSansCJK", "NanumGothic", "malgun"]):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})


def plot_km(curves: dict, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    for (g, sf), alpha in zip(curves.items(), [0.35, 0.65, 1.0]):
        t = np.r_[0, sf.surv_times] / 365.25
        s = np.r_[1, sf.surv_prob]
        row = table.set_index("group").loc[g]
        ax.step(t, s, where="post", color="#2a5d9f", alpha=alpha, lw=1.5)
        ax.text(t[-1] + 0.05, s[-1], f"{g} {row.biz200_min:.0f}–{row.biz200_max:.0f}곳  {s[-1]:.1%}", va="center", fontsize=8)
    reopen = (REOPEN_DATE - BASELINE).days / 365.25
    ax.axvline(reopen, color="#999", lw=0.7, ls="--")
    ax.text(reopen + 0.05, 0.47, "거리두기 해제", fontsize=7, color="#777")
    ax.set(xlim=(0, 6.8), ylim=(0.45, 1.0), xlabel="기준일(2021-04-01) 이후 경과 연수", ylabel="생존 확률",
           title="반경 200m 음식점 수 3분위별 Kaplan-Meier 생존곡선")
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    fig.savefig(FIG_DIR / "fig_km_by_adjacent_biz_200m.png", bbox_inches="tight")
    plt.close(fig)


def plot_forest(radius: pd.DataFrame) -> None:
    panels = [("adjacent_biz", "B"), ("competitor_c", "A"), ("competitor_r", "B")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    y, ticks, labels = 0, [], []
    for var, model in panels:
        ax.text(0.66, y, f"{LABEL[var]} (모형 {model})", fontweight="bold", va="center", fontsize=8.5)
        y -= 1
        for r in RADII:
            row = radius[(radius.model == model) & (radius.variable == f"{var}_{r}m")].iloc[0]
            color = "#2a5d9f" if row.p < 0.05 else "#8c8c8c"
            ax.plot([row.ci_low_per_sd, row.ci_high_per_sd], [y, y], color=color, lw=1.3)
            ax.plot(row.hr_per_sd, y, "s", color=color, ms=4.5)
            ax.text(1.36, y, f"{row.hr_per_sd:.2f} ({row.ci_low_per_sd:.2f}–{row.ci_high_per_sd:.2f})"
                    + ("*" if row.p < 0.05 else ""), va="center", fontsize=8)
            ticks.append(y); labels.append(f"{r}m"); y -= 1
        y -= 0.4
    ax.axvline(1, color="#666", lw=0.7, ls="--")
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"])
    ax.set(xlim=(0.68, 1.33), ylim=(y + 0.6, 0.7), yticks=ticks, yticklabels=labels,
           xlabel="표준편차 1 증가당 위험비(HR)와 95% 신뢰구간  (← 폐업 위험 감소 | 증가 →, * p < .05)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(FIG_DIR / "fig_forest_competition_by_radius.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 요약 보고서
def fmt(row) -> str:
    star = "***" if row.p < .001 else "**" if row.p < .01 else "*" if row.p < .05 else "†" if row.p < .1 else ""
    return f"{row.hr:.3f} ({row.ci_low:.3f}–{row.ci_high:.3f}){star}"


def write_summary(df_all, df, km, radius, network, robust, full, tv) -> None:
    def pick(model, var, r):
        return radius[(radius.model == model) & (radius.variable == f"{var}_{r}m")].iloc[0]
    lines = [
        "# 분석 결과 요약", "",
        f"- 전체 표본(Kaplan-Meier): {len(df_all):,}개 점포, 폐업 {int(df_all.event.sum()):,}건 ({df_all.event.mean():.1%}), "
        f"관찰 {BASELINE.date()} ~ {STUDY_END.date()}",
        f"- Cox 표본: 상권 매출 결측 {len(df_all) - len(df):,}곳 제외 → {len(df):,}개 점포, 폐업 {int(df.event.sum()):,}건 ({df.event.mean():.1%})",
        "- 모든 Cox 모형: 업력, 임대료 수준(low/mid/high), log 상권 매출, 업태 더미 통제, "
        f"상권 단위 군집 강건 표준오차({df.cluster.nunique()}개 군집)",
        "- 표기: HR (95% CI), *** p<.001 ** p<.01 * p<.05 † p<.10", "",
        "## 1. Kaplan-Meier: 반경 200m 음식점 수 3분위", "",
        "| 집단 | 200m 음식점 수 | n | 관찰 종료 시점 생존율 |", "|---|---|---|---|",
        *[f"| {r.group} | {r.biz200_min}–{r.biz200_max} | {r.n:,} | {r.survival_at_end:.1%} |" for r in km.itertuples()],
        "", f"log-rank χ² = {km.logrank_chi2.iloc[0]:.1f} (df = 2), p = {km.logrank_p.iloc[0]:.1e}", "",
        "## 2. 반경 경쟁 변수 (H1–H3)", "",
        "단위: C·BIZ 10곳, R 10%p 증가당", "",
        "| 반경 | 모형 A: C | 모형 A: BIZ | 모형 B: R | 모형 B: BIZ | C-index (A/B) | BIZ HR per SD (B) |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r}m | {fmt(pick('A','competitor_c',r))} | {fmt(pick('A','adjacent_biz',r))} | {fmt(pick('B','competitor_r',r))} | "
          f"{fmt(pick('B','adjacent_biz',r))} | {pick('A','competitor_c',r).c_index:.3f} / {pick('B','competitor_r',r).c_index:.3f} | "
          f"{pick('B','adjacent_biz',r).hr_per_sd:.2f} |" for r in RADII],
        "", "비례위험 가정(Schoenfeld 잔차 vs log t) 최소 p: "
        + ", ".join(f"{LABEL[v]} {radius[radius.variable.str.startswith(v)].ph_p.min():.3f}"
                    for v in ["competitor_c", "adjacent_biz", "competitor_r"]), "",
        "## 3. 경쟁 네트워크 변수 (H4, 모형 C = 모형 A + 네트워크 지표)", "",
        "단위: 네트워크 지표 표준편차 1 증가당 (매개중심성은 log 변환 후)", "",
        "| 반경 | 연결강도 | 매개중심성 | 군집계수 | LR χ²(3), p | AIC (A → C) | C-index (A → C) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in RADII:
        g = network[network.radius_m == r].set_index("variable")
        lines.append(f"| {r}m | {fmt(g.loc[f'net_strength_{r}m'])} | {fmt(g.loc[f'net_betweenness_{r}m'])} | "
                     f"{fmt(g.loc[f'net_clustering_{r}m'])} | {g.lr_chi2.iloc[0]:.1f}, p = {g.lr_p.iloc[0]:.3f} | "
                     f"{g.aic_base.iloc[0]:.1f} → {g.aic.iloc[0]:.1f} | {g.c_index_base.iloc[0]:.3f} → {g.c_index.iloc[0]:.3f} |")
    lines += ["", f"연결강도와 Competitor_C의 VIF: {network[network.variable.str.startswith(('net_strength','competitor_c'))].vif.max():.1f} (최대)", "",
              "## 4. 통제 변수 (300m, 모형 B)", "", "| 변수 | HR (95% CI) |", "|---|---|",
              *[f"| {i} | {fmt(r)} |" for i, r in full.iterrows() if not i.startswith("cat_")], "",
              "## 5. 강건성 (300m, 모형 B)", "", "| 설정 | n | BIZ | R |", "|---|---|---|---|"]
    for spec, g in robust.groupby("spec", sort=False):
        g = g.set_index("variable")
        lines.append(f"| {spec} | {g.n.iloc[0]:,} | {fmt(g.loc['adjacent_biz_300m'])} | {fmt(g.loc['competitor_r_300m'])} |")
    lines += ["", f"일괄 폐업일: {', '.join(robust.attrs['bulk_dates'])} (전체 표본 기준 총 {robust.attrs['bulk_n']}곳)", "",
              "## 6. 비례위험 가정 보완: Competitor_R 시간가변 계수 (모형 B, R × 관찰 1년 구간)", "",
              "단위: R 10%p 증가당. 구간 계수가 모두 같은지 Wald 검정", "",
              "| 반경 | " + " | ".join(tv.period.unique()) + " | 구간 동일성 Wald χ², p | BIZ HR |",
              "|---" * (tv.period.nunique() + 3) + "|"]
    for r, g in tv.groupby("radius_m"):
        lines.append(f"| {r}m | " + " | ".join(fmt(x) for x in g.itertuples()) +
                     f" | {g.equal_wald_chi2.iloc[0]:.1f} (df = {g.equal_wald_df.iloc[0]}), p = {g.equal_wald_p.iloc[0]:.3f} | {g.biz_hr.iloc[0]:.3f} |")
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- 실행
def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_font()
    df_all = load_data()
    df = cox_sample(df_all)
    print(f"전체 {len(df_all):,}개 (폐업 {int(df_all.event.sum()):,}건) | "
          f"Cox 표본 {len(df):,}개 (매출 결측 {len(df_all) - len(df):,}곳 제외, 폐업 {int(df.event.sum()):,}건, 군집 {df.cluster.nunique()}개)")

    print("1) Kaplan-Meier (전체 표본)")
    km = km_analysis(df_all)
    km.to_csv(TABLE_DIR / "km_by_adjacent_biz_200m.csv", index=False, encoding="utf-8-sig")

    print("2) 반경 경쟁 Cox 모형 (모형 A·B × 4개 반경)")
    radius, fits = radius_models(df)
    radius.to_csv(TABLE_DIR / "cox_radius_models.csv", index=False, encoding="utf-8-sig")
    plot_forest(radius)

    full = fit_cox(df, pd.concat([df[["x_competitor_r_300m", "x_adjacent_biz_300m"]], controls(df)], axis=1))["coef"]
    full.index = full.index.str.removeprefix("x_")
    full.to_csv(TABLE_DIR / "cox_full_300m_model_b.csv", encoding="utf-8-sig")

    print("3) 경쟁 네트워크 Cox 모형 (모형 C)")
    network = network_models(df, fits)
    network.to_csv(TABLE_DIR / "cox_network_models.csv", index=False, encoding="utf-8-sig")

    print("4) 강건성 검토")
    robust = robustness(df, add_model_vars(df_all.copy()))
    robust.to_csv(TABLE_DIR / "cox_robustness_300m.csv", index=False, encoding="utf-8-sig")

    print("5) 비례위험 가정 보완 (Competitor_R 시간가변 계수)")
    tv = time_varying_r(df)
    tv.to_csv(TABLE_DIR / "cox_time_varying_r.csv", index=False, encoding="utf-8-sig")

    write_summary(df_all, df, km, radius, network, robust, full, tv)
    print(f"완료: {TABLE_DIR.relative_to(ROOT)}, {FIG_DIR.relative_to(ROOT)}, {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
