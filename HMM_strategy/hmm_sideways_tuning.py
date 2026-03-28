"""
HMM Sideways 倉位調整實驗
─────────────────────────────────────────────────────────
在 hmm_regime.py 的基礎上，掃描不同的 SIDEWAYS_POS 組合
找出 Sharpe 和 MDD 最佳平衡點

固定不變：
  HMM 訓練邏輯、特徵、狀態辨識
  bear 反轉訊號（mom > 3% + 站上MA60 → 0.60）

調整的是：
  SIDEWAYS_POS["both"]    → MA60上方 且 動能正
  SIDEWAYS_POS["either"]  → 其中一個條件
  SIDEWAYS_POS["neither"] → 兩個條件都不符
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"]       = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution

# ── 設定（與 hmm_regime.py 保持一致）────────────────────────────────────────
SYMBOL      = "^TWII"
PERIOD      = "10y"
N_STATES    = 3
N_ITER      = 300
TRAIN_RATIO = 0.60
CASH        = 100_000
FEATURE_COLS = ["ret_1d", "vol_5d", "mom_10d", "vol_ratio"]

POSITION_MAP = {
    "bull": 1.00,
    "bear": 0.00,
}

# ── 掃描的倉位範圍 ────────────────────────────────────────────────────────────
BOTH_RANGE    = [0.85, 0.90, 0.95, 1.00]   # MA60上方 + 動能正
EITHER_RANGE  = [0.55, 0.65, 0.70, 0.75]   # 其中一個條件
NEITHER_RANGE = [0.25, 0.35, 0.40, 0.50]   # 兩個條件都不符


def build_features(df):
    close = df["Close"]
    log_r = np.log(close / close.shift(1))
    df["ret_1d"]    = log_r
    df["vol_5d"]    = log_r.rolling(5).std() * np.sqrt(252)
    df["mom_10d"]   = np.log(close / close.shift(10))
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["MA60"]      = close.rolling(60).mean()
    return df.dropna()


def train_hmm(X_train):
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=42,
    )
    model.fit(X_train)
    return model


def identify_states(model, scaler):
    means_orig = scaler.inverse_transform(model.means_)
    means_df   = pd.DataFrame(means_orig, columns=FEATURE_COLS)
    ret_col    = means_df["ret_1d"]
    bull_id    = int(ret_col.idxmax())
    bear_id    = int(ret_col.idxmin())
    side_id    = [i for i in range(N_STATES) if i not in (bull_id, bear_id)][0]
    return {bull_id: "bull", bear_id: "bear", side_id: "sideways"}


def backtest_with_pos(df_test, states_test, label_map, sideways_pos):
    close    = df_test["Close"].values
    ma60     = df_test["MA60"].values
    mom      = df_test["mom_10d"].values
    n        = len(close)
    cash     = float(CASH)
    shares   = 0.0
    port_val = []

    for i in range(n):
        port_val.append(cash + shares * close[i])
        if i == n - 1:
            break

        label = label_map[states_test[i]]

        if label == "sideways":
            above_ma = close[i] > ma60[i]
            pos_mom  = mom[i] > 0
            if above_ma and pos_mom:
                target_pct = sideways_pos["both"]
            elif above_ma or pos_mom:
                target_pct = sideways_pos["either"]
            else:
                target_pct = sideways_pos["neither"]
        elif label == "bear":
            strong_up  = (mom[i] > 0.03) and (close[i] > ma60[i])
            target_pct = 0.60 if strong_up else POSITION_MAP["bear"]
        else:
            target_pct = POSITION_MAP[label]

        total      = cash + shares * close[i]
        target_val = total * target_pct
        curr_val   = shares * close[i]
        adj        = (target_val - curr_val) / close[i]
        shares    += adj
        cash      -= adj * close[i]

    port = pd.Series(port_val, index=df_test.index)
    dr   = port.pct_change().dropna()
    sharpe = ((dr.mean() - 0.02/252) / dr.std() * 252**0.5) if dr.std() > 0 else 0
    cum    = (1 + dr).cumprod()
    mdd    = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    ret    = (port.iloc[-1] / port.iloc[0] - 1) * 100
    return {"sharpe": sharpe, "mdd": mdd, "ret": ret}


def main():
    print(f"下載 {SYMBOL} 資料（{PERIOD}）...")
    raw = yf.Ticker(SYMBOL).history(period=PERIOD, interval="1d")
    df  = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    df  = build_features(df)

    train_end = int(len(df) * TRAIN_RATIO)
    df_train  = df.iloc[:train_end]
    df_test   = df.iloc[train_end:].copy()

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test  = scaler.transform(df_test[FEATURE_COLS].values)

    print("訓練 HMM...")
    model     = train_hmm(X_train)
    label_map = identify_states(model, scaler)
    states_test = model.predict(X_test)

    # Buy & Hold 基準
    bh_dr  = df_test["Close"].pct_change().dropna()
    bh_sh  = (bh_dr.mean() - 0.02/252) / bh_dr.std() * 252**0.5
    bh_ret = (df_test["Close"].iloc[-1] / df_test["Close"].iloc[0] - 1) * 100
    bh_cum = (1 + bh_dr).cumprod()
    bh_mdd = float(((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min() * 100)
    print(f"\nBuy & Hold  Sharpe={bh_sh:.3f}  MDD={bh_mdd:.2f}%  報酬={bh_ret:.2f}%")

    # 掃描所有組合
    print(f"\n掃描 {len(BOTH_RANGE)*len(EITHER_RANGE)*len(NEITHER_RANGE)} 個倉位組合...\n")
    results = []

    for both, either, neither in itertools.product(BOTH_RANGE, EITHER_RANGE, NEITHER_RANGE):
        if not (both >= either >= neither):   # 確保邏輯一致：雙確認 ≥ 單確認 ≥ 無確認
            continue
        pos = {"both": both, "either": either, "neither": neither}
        m   = backtest_with_pos(df_test, states_test, label_map, pos)
        results.append({**pos, **m})

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # 印出前 10 名
    print(f"{'both':>6} {'either':>7} {'neither':>8}  {'Sharpe':>8}  {'MDD':>8}  {'報酬':>8}")
    print("─" * 60)
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['both']:.2f}   {row['either']:.2f}     {row['neither']:.2f}"
              f"    {row['sharpe']:>7.3f}   {row['mdd']:>7.2f}%  {row['ret']:>7.2f}%")

    # 最佳組合
    best = results_df.iloc[0]
    print(f"\n最佳組合：")
    print(f"  SIDEWAYS_POS = {{")
    print(f"      'both':    {best['both']:.2f},")
    print(f"      'either':  {best['either']:.2f},")
    print(f"      'neither': {best['neither']:.2f},")
    print(f"  }}")
    print(f"  Sharpe={best['sharpe']:.3f}  MDD={best['mdd']:.2f}%  報酬={best['ret']:.2f}%")

    # 視覺化：Sharpe vs MDD 散點圖
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results_df["mdd"], results_df["sharpe"],
                    c=results_df["ret"], cmap="RdYlGn", alpha=0.7, s=60)
    ax.axvline(bh_mdd, color="gray",  linestyle="--", label=f"B&H MDD {bh_mdd:.1f}%")
    ax.axhline(bh_sh,  color="black", linestyle="--", label=f"B&H Sharpe {bh_sh:.3f}")
    plt.colorbar(sc, label="總報酬 (%)")
    ax.set_xlabel("最大回撤 (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Grid Search：Sideways 倉位組合 Sharpe vs 回撤")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── Differential Evolution 優化 ──────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  Differential Evolution 優化（連續搜尋空間）")
    print(f"{'─'*58}")

    de_trials = []   # 記錄每次 DE 評估的結果

    def de_objective(params):
        both, either, neither = params
        # 違反邏輯約束（both >= either >= neither）時給予懲罰
        if not (both >= either >= neither):
            return 10.0
        pos = {"both": both, "either": either, "neither": neither}
        m   = backtest_with_pos(df_test, states_test, label_map, pos)
        de_trials.append({**pos, **m})
        return -m["sharpe"]   # minimize → 加負號

    bounds = [
        (0.50, 1.00),   # both    範圍
        (0.20, 0.90),   # either  範圍
        (0.00, 0.60),   # neither 範圍
    ]

    de_result = differential_evolution(
        de_objective,
        bounds,
        seed=42,
        maxiter=200,
        popsize=12,
        tol=0.0001,
        polish=True,    # 最後用 L-BFGS-B 精煉
        disp=False,
    )

    b_opt, e_opt, n_opt = de_result.x
    best_pos = {"both": b_opt, "either": e_opt, "neither": n_opt}
    best_m   = backtest_with_pos(df_test, states_test, label_map, best_pos)

    print(f"\n  DE 最佳組合：")
    print(f"    SIDEWAYS_POS = {{")
    print(f"        'both':    {b_opt:.4f},")
    print(f"        'either':  {e_opt:.4f},")
    print(f"        'neither': {n_opt:.4f},")
    print(f"    }}")
    print(f"    Sharpe={best_m['sharpe']:.3f}  "
          f"MDD={best_m['mdd']:.2f}%  "
          f"報酬={best_m['ret']:.2f}%")

    # Grid Search 最佳 vs DE 最佳 對比
    gs_best = results_df.iloc[0]
    print(f"\n  {'':20s}  {'Sharpe':>8}  {'MDD':>9}  {'報酬':>8}")
    print(f"  {'Grid Search 最佳':20s}  "
          f"{gs_best['sharpe']:>8.3f}  "
          f"{gs_best['mdd']:>8.2f}%  "
          f"{gs_best['ret']:>7.2f}%")
    print(f"  {'DE 最佳':20s}  "
          f"{best_m['sharpe']:>8.3f}  "
          f"{best_m['mdd']:>8.2f}%  "
          f"{best_m['ret']:>7.2f}%")
    print(f"  {'Buy & Hold':20s}  "
          f"{bh_sh:>8.3f}  "
          f"{bh_mdd:>8.2f}%  "
          f"{bh_ret:>7.2f}%")


if __name__ == "__main__":
    main()