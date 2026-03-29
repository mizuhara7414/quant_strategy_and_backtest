"""
參數敏感性測試 — HMM 輪動策略
─────────────────────────────────────────────────────────
逐一改變關鍵參數，其餘固定，觀察績效變化。
若微調參數後結果大幅崩潰 → 過擬合
若結果穩定 → 策略具有真實有效性
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.rcParams["font.family"]       = ["Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import hmm_rotation_realtime as base
from hmm_rotation_realtime import (
    build_features, train_hmm,
    forward_decode, backtest,
    EQUITY_SYMBOL, INVERSE_SYMBOL, PERIOD,
    TRAIN_RATIO, FEATURE_COLS,
)

# ── 基準參數 ──────────────────────────────────────────────────────────────────
BASE_PARAMS = {
    "MAX_INVERSE_DAYS":   40,
    "BEAR_INVERSE_PCT":   0.30,
    "CONFIRM_DAYS":       2,
    "N_STATES":           3,
}

# ── 測試範圍 ──────────────────────────────────────────────────────────────────
PARAM_RANGES = {
    "MAX_INVERSE_DAYS":  [10, 20, 30, 40, 50, 60],
    "BEAR_INVERSE_PCT":  [0.10, 0.20, 0.30, 0.40, 0.50],
    "CONFIRM_DAYS":      [1, 2, 3, 4],
    "N_STATES":          [2, 3, 4],
}


def prepare_base_data():
    """下載資料，回傳訓練/測試集（只下載一次）"""
    print("下載資料...")
    raw_eq   = yf.Ticker(EQUITY_SYMBOL).history(period=PERIOD, interval="1d")
    raw_bond = yf.Ticker(INVERSE_SYMBOL).history(period=PERIOD, interval="1d")

    df_eq   = raw_eq[["Open","High","Low","Close","Volume"]].copy().dropna()
    df_bond = raw_bond[["Close"]].copy().dropna()
    df_eq   = build_features(df_eq)

    train_end  = int(len(df_eq) * TRAIN_RATIO)
    df_train   = df_eq.iloc[:train_end]
    df_test_eq = df_eq.iloc[train_end:].copy()

    print(f"  訓練：{df_train.index[0].date()} → {df_train.index[-1].date()}")
    print(f"  測試：{df_test_eq.index[0].date()} → {df_test_eq.index[-1].date()}\n")

    return df_train, df_test_eq, df_bond


def identify_states_flexible(model, scaler, n_states):
    """
    彈性版狀態辨識：支援任意 N_STATES。
    依平均日報酬排序：最高 → bull，最低 → bear，其餘 → sideways。
    """
    means_orig = scaler.inverse_transform(model.means_)
    means_df   = pd.DataFrame(means_orig, columns=FEATURE_COLS)
    ret_col    = means_df["ret_1d"]
    bull_id    = int(ret_col.idxmax())
    bear_id    = int(ret_col.idxmin())
    label_map  = {}
    for i in range(n_states):
        if i == bull_id:
            label_map[i] = "bull"
        elif i == bear_id:
            label_map[i] = "bear"
        else:
            label_map[i] = "sideways"
    return label_map


def run_one(df_train, df_test_eq, df_bond, params):
    """用指定參數跑一次回測，回傳績效指標"""
    # 覆蓋模組參數
    base.MAX_INVERSE_DAYS = params["MAX_INVERSE_DAYS"]
    base.BEAR_INVERSE_PCT = params["BEAR_INVERSE_PCT"]
    base.CONFIRM_DAYS     = params["CONFIRM_DAYS"]

    n_states = params["N_STATES"]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test  = scaler.transform(df_test_eq[FEATURE_COLS].values)

    try:
        model = base.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=base.N_ITER,
            random_state=42,
        )
        model.fit(X_train)
        label_map   = identify_states_flexible(model, scaler, n_states)
        states_test = forward_decode(model, X_test)
    except Exception:
        return None

    result, _, _ = backtest(df_test_eq, df_bond, states_test, label_map,
                            use_protection=True)
    port = result["port_value"]
    dr   = port.pct_change().dropna().values
    cum  = np.cumprod(1 + dr)
    ret  = (cum[-1] - 1) * 100
    sh   = (dr.mean() - 0.02/252) / (dr.std() + 1e-10) * 252**0.5
    mdd  = float(((cum / np.maximum.accumulate(cum)) - 1).min() * 100)
    return {"ret": ret, "sharpe": sh, "mdd": mdd}


def run_sensitivity(df_train, df_test_eq, df_bond):
    all_results = {}

    for param_name, values in PARAM_RANGES.items():
        print(f"  測試 {param_name}：{values}")
        results = []
        for v in values:
            params = BASE_PARAMS.copy()
            params[param_name] = v
            m = run_one(df_train, df_test_eq, df_bond, params)
            if m:
                results.append({"value": v, **m})
            else:
                results.append({"value": v, "ret": np.nan,
                                "sharpe": np.nan, "mdd": np.nan})
        all_results[param_name] = results

    # 還原基準參數
    base.MAX_INVERSE_DAYS = BASE_PARAMS["MAX_INVERSE_DAYS"]
    base.BEAR_INVERSE_PCT = BASE_PARAMS["BEAR_INVERSE_PCT"]
    base.CONFIRM_DAYS     = BASE_PARAMS["CONFIRM_DAYS"]

    return all_results


def print_results(all_results):
    print(f"\n{'═'*65}")
    print(f"  參數敏感性測試結果")
    print(f"{'═'*65}")

    for param_name, results in all_results.items():
        base_val = BASE_PARAMS[param_name]
        print(f"\n  {param_name}（基準值 = {base_val}）")
        print(f"  {'值':>8}  {'累積報酬':>10}  {'Sharpe':>8}  {'MDD':>8}  {'vs基準'}")
        print(f"  {'─'*55}")

        base_ret = next((r["ret"] for r in results if r["value"] == base_val), None)

        for r in results:
            marker = " ←基準" if r["value"] == base_val else ""
            diff   = f"{r['ret'] - base_ret:+.1f}%" if base_ret and not np.isnan(r["ret"]) and r["value"] != base_val else ""
            print(f"  {str(r['value']):>8}  {r['ret']:>+9.1f}%  "
                  f"{r['sharpe']:>8.3f}  {r['mdd']:>7.1f}%  {diff}{marker}")


def plot_sensitivity(all_results):
    n_params = len(all_results)
    fig, axes = plt.subplots(3, n_params, figsize=(4 * n_params, 10))

    metrics = [("ret", "累積報酬 (%)"), ("sharpe", "Sharpe Ratio"), ("mdd", "MDD (%)")]

    for col, (param_name, results) in enumerate(all_results.items()):
        base_val = BASE_PARAMS[param_name]
        xs  = [r["value"] for r in results]
        colors = ["#e74c3c" if x == base_val else "#4a90d9" for x in xs]

        for row, (metric, ylabel) in enumerate(metrics):
            ax  = axes[row, col]
            ys  = [r[metric] for r in results]
            ax.bar(range(len(xs)), ys, color=colors, alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")

            # 標示基準值的橫線
            base_y = next((r[metric] for r in results if r["value"] == base_val), None)
            if base_y:
                ax.axhline(base_y, color="#e74c3c", linewidth=1,
                           linestyle=":", alpha=0.6)

            if row == 0:
                ax.set_title(param_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    plt.suptitle("參數敏感性測試（紅色 = 基準值）", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_nstates_curves(df_train, df_test_eq, df_bond):
    """N_STATES = 2 / 3 / 4 的淨值曲線對比"""
    colors = {2: "#e74c3c", 3: "#2ecc71", 4: "#4a90d9"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bh_close = df_test_eq["Close"].values
    bh_curve = bh_close / bh_close[0] * 100

    for n in [2, 3, 4]:
        params = BASE_PARAMS.copy()
        params["N_STATES"] = n

        base.MAX_INVERSE_DAYS = params["MAX_INVERSE_DAYS"]
        base.BEAR_INVERSE_PCT = params["BEAR_INVERSE_PCT"]
        base.CONFIRM_DAYS     = params["CONFIRM_DAYS"]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
        X_test  = scaler.transform(df_test_eq[FEATURE_COLS].values)

        model = base.GaussianHMM(
            n_components=n, covariance_type="full",
            n_iter=base.N_ITER, random_state=42,
        )
        model.fit(X_train)
        label_map   = identify_states_flexible(model, scaler, n)
        states_test = forward_decode(model, X_test)
        result, _, _ = backtest(df_test_eq, df_bond, states_test, label_map,
                                use_protection=True)

        port  = result["port_value"].values
        curve = port / port[0] * 100
        dr    = np.diff(port) / port[:-1]
        cum   = np.cumprod(1 + dr)
        ret   = (cum[-1] - 1) * 100
        sh    = (dr.mean() - 0.02/252) / (dr.std() + 1e-10) * 252**0.5
        mdd   = float(((cum / np.maximum.accumulate(cum)) - 1).min() * 100)

        x = np.arange(len(curve))
        label = f"N={n}  報酬{ret:+.0f}%  Sharpe {sh:.2f}  MDD {mdd:.1f}%"

        # 圖1：線性尺度
        axes[0].plot(x, curve, color=colors[n], linewidth=1.8,
                     label=label, linestyle="--" if n != 3 else "-")
        # 圖2：對數尺度（讓差距更清楚）
        axes[1].semilogy(x, curve, color=colors[n], linewidth=1.8,
                         label=label, linestyle="--" if n != 3 else "-")

    for ax, title in zip(axes, ["淨值曲線（線性）", "淨值曲線（對數尺度）"]):
        ax.plot(np.arange(len(bh_curve)), bh_curve, color="gray",
                linewidth=1.2, linestyle=":", label="買持 0050")
        ax.axhline(100, color="gray", linewidth=0.6, linestyle="--")
        ax.set_xlabel("交易天數")
        ax.set_ylabel("淨值（起始=100）")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # 還原基準參數
    base.MAX_INVERSE_DAYS = BASE_PARAMS["MAX_INVERSE_DAYS"]
    base.BEAR_INVERSE_PCT = BASE_PARAMS["BEAR_INVERSE_PCT"]
    base.CONFIRM_DAYS     = BASE_PARAMS["CONFIRM_DAYS"]

    plt.suptitle("N_STATES 比較：2 vs 3 vs 4", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    df_train, df_test_eq, df_bond = prepare_base_data()

    print("開始敏感性測試...")
    all_results = run_sensitivity(df_train, df_test_eq, df_bond)

    print_results(all_results)
    plot_sensitivity(all_results)

    print("\n繪製 N_STATES 淨值曲線對比...")
    plot_nstates_curves(df_train, df_test_eq, df_bond)


if __name__ == "__main__":
    main()
