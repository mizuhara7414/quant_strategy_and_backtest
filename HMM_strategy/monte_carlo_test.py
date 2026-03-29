"""
蒙地卡羅測試 — HMM 輪動策略統計顯著性驗證
─────────────────────────────────────────────────────────
方法：訊號隨機化（Signal Randomization）
  1. 用原始訓練好的模型跑出真實 HMM 訊號序列
  2. 隨機打亂訊號順序 N 次（保留相同的牛/熊/盤整天數比例）
  3. 每次用打亂的訊號跑完整回測
  4. 比較真實策略 vs 隨機訊號的績效分布

這測試的核心問題：
  HMM 的進出場時機比隨機猜測好多少？
  若真實策略落在前 5%，代表 HMM 的判斷有統計顯著性。
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

from hmm_rotation_realtime import (
    build_features, train_hmm, identify_states,
    forward_decode, backtest,
    EQUITY_SYMBOL, INVERSE_SYMBOL, PERIOD,
    TRAIN_RATIO, FEATURE_COLS
)

# ── 參數 ──────────────────────────────────────────────────────────────────────
N_SIMULATIONS = 1000
RANDOM_SEED   = 42


def prepare_data():
    """下載資料、訓練模型、取得測試集與真實 HMM 訊號"""
    print("下載資料並訓練模型...")
    raw_eq   = yf.Ticker(EQUITY_SYMBOL).history(period=PERIOD, interval="1d")
    raw_bond = yf.Ticker(INVERSE_SYMBOL).history(period=PERIOD, interval="1d")

    df_eq   = raw_eq[["Open","High","Low","Close","Volume"]].copy().dropna()
    df_bond = raw_bond[["Close"]].copy().dropna()
    df_eq   = build_features(df_eq)

    train_end  = int(len(df_eq) * TRAIN_RATIO)
    df_train   = df_eq.iloc[:train_end]
    df_test_eq = df_eq.iloc[train_end:].copy()

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test  = scaler.transform(df_test_eq[FEATURE_COLS].values)

    model       = train_hmm(X_train)
    label_map   = identify_states(model, scaler)
    states_real = forward_decode(model, X_test)

    return df_test_eq, df_bond, states_real, label_map


def run_backtest_metrics(df_test_eq, df_bond, states, label_map):
    """跑回測並回傳 (port_value series, 累積報酬%, Sharpe, MDD%)"""
    result, _, _ = backtest(df_test_eq, df_bond, states, label_map,
                            use_protection=True)
    port = result["port_value"]
    dr   = port.pct_change().dropna().values
    cum  = np.cumprod(1 + dr)
    ret  = (cum[-1] - 1) * 100
    sh   = (dr.mean() - 0.02/252) / (dr.std() + 1e-10) * 252**0.5
    mdd  = float(((cum / np.maximum.accumulate(cum)) - 1).min() * 100)
    return port.values, ret, sh, mdd


def simulate_random_signals(df_test_eq, df_bond, states_real, label_map,
                             n_sim, seed):
    """
    隨機打亂 HMM 訊號順序，跑 n_sim 次回測。
    保留相同的狀態分布（牛/熊/盤整天數比例不變），只改變時序。
    """
    rng = np.random.default_rng(seed)
    sim_rets    = []
    sim_sharpes = []
    sim_mdds    = []
    sim_curves  = []

    for i in range(n_sim):
        shuffled = rng.permutation(states_real)
        _, ret, sh, mdd = run_backtest_metrics(
            df_test_eq, df_bond, shuffled, label_map)
        sim_rets.append(ret)
        sim_sharpes.append(sh)
        sim_mdds.append(mdd)

        if i < 200:  # 只儲存前 200 條曲線供畫圖用
            result, _, _ = backtest(df_test_eq, df_bond, shuffled, label_map,
                                    use_protection=True)
            port = result["port_value"].values
            sim_curves.append(port / port[0])

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_sim} 完成...")

    return (np.array(sim_rets), np.array(sim_sharpes),
            np.array(sim_mdds), sim_curves)


def percentile_rank(value, distribution):
    return float((distribution < value).mean() * 100)


def plot_results(sim_curves, actual_port, sim_rets, sim_sharpes, sim_mdds,
                 actual_ret, actual_sh, actual_mdd,
                 pct_ret, pct_sh, pct_mdd_good):

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 圖1：模擬淨值曲線 vs 真實策略
    ax = axes[0, 0]
    norm_actual = actual_port / actual_port[0]
    x = np.arange(len(norm_actual))
    for curve in sim_curves:
        ax.plot(x, curve * 100, color="#aec6cf", linewidth=0.4, alpha=0.3)
    ax.plot(x, norm_actual * 100, color="#e74c3c", linewidth=2,
            label=f"真實策略 {actual_ret:+.1f}%", zorder=5)
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("交易天數")
    ax.set_ylabel("淨值（起始=100）")
    ax.set_title("隨機訊號模擬曲線 vs 真實策略")
    ax.legend(fontsize=9)

    # 圖2：最終報酬分布
    ax = axes[0, 1]
    ax.hist(sim_rets, bins=60, color="#4a90d9",
            edgecolor="white", linewidth=0.3, alpha=0.8, density=True)
    ax.axvline(actual_ret, color="#e74c3c", linewidth=2,
               label=f"真實 {actual_ret:.1f}%（前 {100-pct_ret:.1f}%）")
    ax.axvline(np.percentile(sim_rets, 95), color="#f39c12",
               linewidth=1.5, linestyle="--", label="95th percentile")
    ax.set_xlabel("累積報酬 (%)")
    ax.set_ylabel("密度")
    ax.set_title("隨機訊號最終報酬分布")
    ax.legend(fontsize=9)

    # 圖3：Sharpe 分布
    ax = axes[1, 0]
    ax.hist(sim_sharpes, bins=60, color="#2ecc71",
            edgecolor="white", linewidth=0.3, alpha=0.8, density=True)
    ax.axvline(actual_sh, color="#e74c3c", linewidth=2,
               label=f"真實 {actual_sh:.3f}（前 {100-pct_sh:.1f}%）")
    ax.axvline(np.percentile(sim_sharpes, 95), color="#f39c12",
               linewidth=1.5, linestyle="--", label="95th percentile")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("密度")
    ax.set_title("隨機訊號 Sharpe 分布")
    ax.legend(fontsize=9)

    # 圖4：MDD 分布
    ax = axes[1, 1]
    ax.hist(sim_mdds, bins=60, color="#9b59b6",
            edgecolor="white", linewidth=0.3, alpha=0.8, density=True)
    ax.axvline(actual_mdd, color="#e74c3c", linewidth=2,
               label=f"真實 {actual_mdd:.1f}%（優於 {pct_mdd_good:.1f}% 模擬）")
    ax.axvline(np.percentile(sim_mdds, 5), color="#f39c12",
               linewidth=1.5, linestyle="--", label="5th percentile")
    ax.set_xlabel("最大回撤 (%)")
    ax.set_ylabel("密度")
    ax.set_title("隨機訊號 MDD 分布（越靠右越差）")
    ax.legend(fontsize=9)

    plt.suptitle("蒙地卡羅測試 — HMM 訊號 vs 隨機訊號", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    df_test_eq, df_bond, states_real, label_map = prepare_data()

    # 真實策略
    actual_port, actual_ret, actual_sh, actual_mdd = run_backtest_metrics(
        df_test_eq, df_bond, states_real, label_map)

    print(f"\n  真實策略：累積報酬 {actual_ret:+.1f}%  "
          f"Sharpe {actual_sh:.3f}  MDD {actual_mdd:.1f}%")

    # 狀態分布
    unique, counts = np.unique(states_real, return_counts=True)
    print(f"  HMM 訊號分布：", end="")
    for s, c in zip(unique, counts):
        print(f"{label_map[s]} {c}天({c/len(states_real)*100:.0f}%)  ", end="")
    print()

    print(f"\n  執行 {N_SIMULATIONS} 次隨機訊號模擬（打亂訊號順序）...")
    sim_rets, sim_sharpes, sim_mdds, sim_curves = simulate_random_signals(
        df_test_eq, df_bond, states_real, label_map, N_SIMULATIONS, RANDOM_SEED)

    # 百分位排名
    pct_ret      = percentile_rank(actual_ret, sim_rets)
    pct_sh       = percentile_rank(actual_sh,  sim_sharpes)
    pct_mdd      = percentile_rank(actual_mdd, sim_mdds)
    pct_mdd_good = 100 - pct_mdd

    # 報告
    print(f"\n{'═'*62}")
    print(f"  蒙地卡羅測試結果（{N_SIMULATIONS} 次隨機訊號模擬）")
    print(f"{'═'*62}")
    print(f"  指標        真實值     隨機中位數   超越比例    顯著？")
    print(f"  {'─'*58}")
    sig_ret = "✓ 顯著" if pct_ret    > 95 else "✗ 不顯著"
    sig_sh  = "✓ 顯著" if pct_sh     > 95 else "✗ 不顯著"
    sig_mdd = "✓ 顯著" if pct_mdd_good > 95 else "✗ 不顯著"
    print(f"  累積報酬  {actual_ret:>+8.1f}%  "
          f"{np.median(sim_rets):>+8.1f}%    前{100-pct_ret:5.1f}%    {sig_ret}")
    print(f"  Sharpe    {actual_sh:>+8.3f}  "
          f"{np.median(sim_sharpes):>+8.3f}    前{100-pct_sh:5.1f}%    {sig_sh}")
    print(f"  MDD       {actual_mdd:>+8.1f}%  "
          f"{np.median(sim_mdds):>+8.1f}%  優於{pct_mdd_good:5.1f}%    {sig_mdd}")
    print(f"{'═'*62}")

    print(f"\n  隨機訊號分布分位數：")
    for p in [5, 25, 50, 75, 95]:
        print(f"    {p:3d}th：報酬 {np.percentile(sim_rets, p):+6.1f}%  "
              f"Sharpe {np.percentile(sim_sharpes, p):+.3f}  "
              f"MDD {np.percentile(sim_mdds, p):.1f}%")

    plot_results(sim_curves, actual_port, sim_rets, sim_sharpes, sim_mdds,
                 actual_ret, actual_sh, actual_mdd,
                 pct_ret, pct_sh, pct_mdd_good)


if __name__ == "__main__":
    main()
