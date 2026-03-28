"""
HMM 三段切分驗證（Train / Validation / Test）
─────────────────────────────────────────────────────────
正確的參數優化流程：

  訓練集（60%）→ HMM 學習狀態分類
  驗證集（20%）→ DE 優化 SIDEWAYS_POS（選參數用）
  測試集（20%）→ 最終評估（只跑一次，不用來選參數）

這樣才能確認 SIDEWAYS_POS 是真的有效，
而不是對測試集過擬合的結果。
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"]       = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution

# ── 設定 ─────────────────────────────────────────────────────────────────────
SYMBOL      = "^TWII"
PERIOD      = "10y"
N_STATES    = 3
N_ITER      = 300
CASH        = 100_000
FEATURE_COLS = ["ret_1d", "vol_5d", "mom_10d", "vol_ratio"]

# 三段切分比例
TRAIN_RATIO = 0.60   # 訓練集
VAL_RATIO   = 0.20   # 驗證集（選參數）
# 測試集 = 剩餘 20%   → 最終評估

POSITION_MAP = {"bull": 1.00, "bear": 0.00}


# ── 特徵工程 ──────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_r = np.log(close / close.shift(1))
    df["ret_1d"]    = log_r
    df["vol_5d"]    = log_r.rolling(5).std() * np.sqrt(252)
    df["mom_10d"]   = np.log(close / close.shift(10))
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["MA60"]      = close.rolling(60).mean()
    return df.dropna()


# ── HMM 訓練 ─────────────────────────────────────────────────────────────────
def train_hmm(X_train: np.ndarray) -> GaussianHMM:
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=42,
    )
    model.fit(X_train)
    return model


def identify_states(model, scaler) -> dict:
    means_orig = scaler.inverse_transform(model.means_)
    means_df   = pd.DataFrame(means_orig, columns=FEATURE_COLS)
    ret_col    = means_df["ret_1d"]
    bull_id    = int(ret_col.idxmax())
    bear_id    = int(ret_col.idxmin())
    side_id    = [i for i in range(N_STATES) if i not in (bull_id, bear_id)][0]
    label_map  = {bull_id: "bull", bear_id: "bear", side_id: "sideways"}
    print(f"\n  HMM 狀態辨識結果：")
    for sid, lbl in label_map.items():
        r   = means_df.loc[sid, "ret_1d"]
        vol = means_df.loc[sid, "vol_5d"]
        print(f"    State {sid} ({lbl:8s})  平均日報酬={r:+.4f}  年化波動率={vol:.3f}")
    return label_map


# ── 回測（帶 SIDEWAYS_POS 參數）─────────────────────────────────────────────
def backtest(df_seg, states, label_map, sideways_pos) -> dict:
    close    = df_seg["Close"].values
    ma60     = df_seg["MA60"].values
    mom      = df_seg["mom_10d"].values
    n        = len(close)
    cash     = float(CASH)
    shares   = 0.0
    port_val = []

    for i in range(n):
        port_val.append(cash + shares * close[i])
        if i == n - 1:
            break

        label = label_map[states[i]]

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
        adj        = (total * target_pct - shares * close[i]) / close[i]
        shares    += adj
        cash      -= adj * close[i]

    port = pd.Series(port_val, index=df_seg.index)
    dr   = port.pct_change().dropna()
    sharpe = ((dr.mean() - 0.02/252) / dr.std() * 252**0.5) if dr.std() > 0 else 0
    cum    = (1 + dr).cumprod()
    mdd    = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    ret    = (port.iloc[-1] / port.iloc[0] - 1) * 100
    return {"sharpe": sharpe, "mdd": mdd, "ret": ret, "port": port}


def bh_metrics(price_series: pd.Series) -> dict:
    dr  = price_series.pct_change().dropna()
    sh  = ((dr.mean() - 0.02/252) / dr.std() * 252**0.5) if dr.std() > 0 else 0
    cum = (1 + dr).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    ret = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
    return {"sharpe": sh, "mdd": mdd, "ret": ret}


# ── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    print(f"下載 {SYMBOL} 資料（{PERIOD}）...")
    raw = yf.Ticker(SYMBOL).history(period=PERIOD, interval="1d")
    df  = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    df  = build_features(df)
    print(f"  共 {len(df)} 根 K 線（{df.index[0].date()} → {df.index[-1].date()}）")

    # 三段切分
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    print(f"\n  訓練集：{df.index[0].date()} → {df.index[train_end-1].date()} "
          f"（{len(df_train)} 天）")
    print(f"  驗證集：{df.index[train_end].date()} → {df.index[val_end-1].date()} "
          f"（{len(df_val)} 天）")
    print(f"  測試集：{df.index[val_end].date()} → {df.index[-1].date()} "
          f"（{len(df_test)} 天）")

    # 標準化
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_val   = scaler.transform(df_val[FEATURE_COLS].values)
    X_test  = scaler.transform(df_test[FEATURE_COLS].values)

    # 訓練 HMM
    print(f"\n訓練 HMM（{N_STATES} 個隱藏狀態，{N_ITER} 次迭代）...")
    model       = train_hmm(X_train)
    label_map   = identify_states(model, scaler)
    states_val  = model.predict(X_val)
    states_test = model.predict(X_test)

    # ── STEP 1：用驗證集做 DE 優化 ──────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  STEP 1：Differential Evolution（驗證集）")
    print(f"{'═'*58}")

    def de_objective(params):
        both, either, neither = params
        if not (both >= either >= neither):
            return 10.0
        pos = {"both": both, "either": either, "neither": neither}
        m   = backtest(df_val, states_val, label_map, pos)
        return -m["sharpe"]

    de_result = differential_evolution(
        de_objective,
        bounds=[(0.50, 1.00), (0.20, 0.90), (0.00, 0.60)],
        seed=42, maxiter=200, popsize=12, tol=0.0001, polish=True, disp=False,
    )

    b_opt, e_opt, n_opt = de_result.x
    best_pos = {"both": b_opt, "either": e_opt, "neither": n_opt}
    val_m    = backtest(df_val, states_val, label_map, best_pos)
    val_bh   = bh_metrics(df_val["Close"])

    print(f"\n  DE 在驗證集找到的最佳參數：")
    print(f"    both={b_opt:.4f}  either={e_opt:.4f}  neither={n_opt:.4f}")
    print(f"\n  驗證集績效：")
    print(f"  {'':20s}  {'策略':>8}  {'B&H':>8}")
    print(f"  {'Sharpe':20s}  {val_m['sharpe']:>8.3f}  {val_bh['sharpe']:>8.3f}")
    print(f"  {'MDD':20s}  {val_m['mdd']:>7.2f}%  {val_bh['mdd']:>7.2f}%")
    print(f"  {'總報酬':20s}  {val_m['ret']:>7.2f}%  {val_bh['ret']:>7.2f}%")

    # ── STEP 2：用測試集做最終評估（只跑一次）──────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  STEP 2：最終評估（測試集，從未用於選參數）")
    print(f"{'═'*58}")

    test_m  = backtest(df_test, states_test, label_map, best_pos)
    test_bh = bh_metrics(df_test["Close"])

    print(f"\n  {'':20s}  {'策略':>8}  {'B&H':>8}")
    print(f"  {'Sharpe':20s}  {test_m['sharpe']:>8.3f}  {test_bh['sharpe']:>8.3f}")
    print(f"  {'MDD':20s}  {test_m['mdd']:>7.2f}%  {test_bh['mdd']:>7.2f}%")
    print(f"  {'總報酬':20s}  {test_m['ret']:>7.2f}%  {test_bh['ret']:>7.2f}%")

    # 過擬合判斷
    print(f"\n  過擬合檢驗：")
    gap = val_m['sharpe'] - test_m['sharpe']
    print(f"    驗證集 Sharpe - 測試集 Sharpe = {gap:+.3f}")
    if abs(gap) < 0.10:
        print(f"    → 差距小，參數穩健，過擬合風險低 ✅")
    elif abs(gap) < 0.25:
        print(f"    → 差距中等，有輕微過擬合，可用但需注意 ⚠️")
    else:
        print(f"    → 差距大，過擬合嚴重，參數不可信 ❌")

    # ── 視覺化 ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    # 驗證集策略線
    val_curve  = test_m["port"] / test_m["port"].iloc[0]
    bh_curve   = df_test["Close"] / df_test["Close"].iloc[0]

    ax.plot(df_test.index, val_curve, color="#0066cc", linewidth=1.5, label="HMM 策略")
    ax.plot(df_test.index, bh_curve,  color="#333333", linewidth=1,   linestyle="--", label="Buy & Hold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"最終測試集績效（未曾用於參數選擇）— {SYMBOL}")
    ax.set_ylabel("累積報酬（倍）")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── 最終建議 ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  建議更新 hmm_regime.py 的 SIDEWAYS_POS：")
    print(f"{'═'*58}")
    print(f"  SIDEWAYS_POS = {{")
    print(f"      'both':    {b_opt:.4f},")
    print(f"      'either':  {e_opt:.4f},")
    print(f"      'neither': {n_opt:.4f},")
    print(f"  }}")


if __name__ == "__main__":
    main()
