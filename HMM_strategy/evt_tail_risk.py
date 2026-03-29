"""
EVT 尾部風險分析（Peak-Over-Threshold 方法）
─────────────────────────────────────────────
從 HMM 輪動策略取得歷史報酬，透過廣義帕累托分布（GPD）
估計極端損失的尾部行為，計算 VaR 與 ES，
並給出 OTM Put 執行價的建議範圍。
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.set_loglevel("error")
matplotlib.rcParams["font.family"]       = ["Microsoft JhengHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from scipy.optimize import minimize_scalar

# ── 從 hmm_rotation 取得策略報酬 ─────────────────────────────────────────────
import sys, os
sys.path.append(os.path.dirname(__file__))

from hmm_rotation import (
    build_features, train_hmm, identify_states, backtest,
    EQUITY_SYMBOL, INVERSE_SYMBOL, PERIOD, TRAIN_RATIO,
    N_STATES, N_ITER, FEATURE_COLS
)
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def get_strategy_returns():
    """執行 HMM 回測，回傳策略每日報酬序列（損失為正值）"""
    print("下載資料並執行回測...")
    raw_eq   = yf.Ticker(EQUITY_SYMBOL).history(period=PERIOD, interval="1d")
    raw_bond = yf.Ticker(INVERSE_SYMBOL).history(period=PERIOD, interval="1d")

    df_eq   = raw_eq[["Open","High","Low","Close","Volume"]].copy().dropna()
    df_bond = raw_bond[["Close"]].copy().dropna()
    df_eq   = build_features(df_eq)

    train_end   = int(len(df_eq) * TRAIN_RATIO)
    df_train    = df_eq.iloc[:train_end]
    df_test_eq  = df_eq.iloc[train_end:].copy()

    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test      = scaler.transform(df_test_eq[FEATURE_COLS].values)

    model       = train_hmm(X_train)
    label_map   = identify_states(model, scaler)
    states_test = model.predict(X_test)

    result, _, _ = backtest(df_test_eq, df_bond, states_test, label_map, use_protection=True)

    port = result["port_value"]
    daily_returns = port.pct_change().dropna()

    # 損失 = 負報酬轉為正值（跌 3% → 損失 0.03）
    losses = -daily_returns
    return daily_returns, losses, df_test_eq["Close"]


# ── EVT 核心函數 ──────────────────────────────────────────────────────────────
def select_threshold(losses, percentile=90):
    """以指定百分位數作為門檻"""
    return np.percentile(losses, percentile)


def fit_gpd(losses, threshold):
    """對超過門檻的損失擬合廣義帕累托分布（GPD）"""
    exceedances = losses[losses > threshold] - threshold
    if len(exceedances) < 10:
        raise ValueError(f"超標樣本不足（{len(exceedances)} 筆），請調低門檻")

    # scipy genpareto：shape=ξ, scale=τ, loc=0
    xi, _, tau = genpareto.fit(exceedances, floc=0)
    return xi, tau, exceedances


def calc_var(xi, tau, threshold, n_total, n_exceed, alpha):
    """
    POT 方法的 VaR 計算
    alpha = 超過機率（例如 0.01 代表 99% VaR）
    """
    prob_exceed = n_exceed / n_total   # 超過門檻的比率
    if abs(xi) < 1e-6:
        var = threshold + tau * np.log(prob_exceed / alpha)
    else:
        var = threshold + tau / xi * ((prob_exceed / alpha) ** xi - 1)
    return var


def calc_es(xi, tau, var, threshold):
    """
    POT 方法的 ES（預期損失）計算
    ES = VaR + 超過 VaR 的平均超額損失
    """
    if abs(xi) < 1e-6:
        es = var + tau
    else:
        es = (var + tau - xi * threshold) / (1 - xi)
    return es


# ── 視覺化 ────────────────────────────────────────────────────────────────────
def plot_evt(losses, threshold, xi, tau, exceedances, var_99, es_99, var_995, es_995):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 圖1：損失分布 + 門檻標示
    ax = axes[0]
    ax.hist(losses * 100, bins=60, color="#4a90d9", edgecolor="white",
            linewidth=0.3, density=True, alpha=0.8)
    ax.axvline(threshold * 100, color="#e74c3c", linewidth=1.5,
               linestyle="--", label=f"門檻 {threshold*100:.2f}%")
    ax.axvline(var_99 * 100, color="#f39c12", linewidth=1.5,
               linestyle="--", label=f"VaR(99%) {var_99*100:.2f}%")
    ax.axvline(es_99 * 100, color="#8e44ad", linewidth=1.5,
               linestyle="--", label=f"ES(99%) {es_99*100:.2f}%")
    ax.set_xlabel("單日損失 (%)")
    ax.set_ylabel("密度")
    ax.set_title("損失分布與風險指標")
    ax.legend(fontsize=8)

    # 圖2：GPD 擬合品質（超標損失的 QQ 圖）
    ax = axes[1]
    n = len(exceedances)
    empirical = np.sort(exceedances)
    theoretical = genpareto.ppf(np.arange(1, n + 1) / (n + 1), c=xi, scale=tau)
    ax.scatter(theoretical * 100, empirical * 100, s=15, alpha=0.6, color="#2ecc71")
    lim_max = max(theoretical.max(), empirical.max()) * 100 * 1.05
    ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1)
    ax.set_xlabel("GPD 理論分位數 (%)")
    ax.set_ylabel("實際超標損失 (%)")
    ax.set_title(f"GPD 擬合 QQ 圖（ξ={xi:.3f}, τ={tau*100:.3f}%）")

    # 圖3：尾部超過機率曲線
    ax = axes[2]
    x_range = np.linspace(threshold, losses.max() * 1.3, 200)
    n_total  = len(losses)
    n_exceed = (losses > threshold).sum()
    prob_exceed = n_exceed / n_total
    if abs(xi) < 1e-6:
        survival = prob_exceed * np.exp(-(x_range - threshold) / tau)
    else:
        survival = prob_exceed * (1 + xi * (x_range - threshold) / tau) ** (-1 / xi)
    ax.semilogy(x_range * 100, survival, color="#e74c3c", linewidth=2)
    ax.axvline(var_99 * 100, color="#f39c12", linestyle="--",
               linewidth=1.2, label=f"VaR(99%) {var_99*100:.2f}%")
    ax.axvline(var_995 * 100, color="#8e44ad", linestyle="--",
               linewidth=1.2, label=f"VaR(99.5%) {var_995*100:.2f}%")
    ax.set_xlabel("損失 (%)")
    ax.set_ylabel("超過機率（對數尺度）")
    ax.set_title("尾部超過機率曲線")
    ax.legend(fontsize=8)

    plt.suptitle("HMM 輪動策略 EVT 尾部風險分析", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    daily_returns, losses, bh_close = get_strategy_returns()
    losses_arr = losses.values

    # ── 門檻選擇（第 90 百分位）
    threshold = select_threshold(losses_arr, percentile=90)
    n_total   = len(losses_arr)
    n_exceed  = (losses_arr > threshold).sum()

    print(f"\n  總交易日：{n_total} 天")
    print(f"  門檻（90th percentile）：{threshold*100:.3f}%")
    print(f"  超過門檻的樣本數：{n_exceed} 筆（{n_exceed/n_total*100:.1f}%）")

    # ── GPD 擬合
    xi, tau, exceedances = fit_gpd(losses_arr, threshold)
    print(f"\n  GPD 擬合參數：")
    print(f"    形狀參數 ξ（xi）= {xi:.4f}  {'（厚尾，符合金融資產特性）' if xi > 0 else '（薄尾）'}")
    print(f"    尺度參數 τ（tau）= {tau*100:.4f}%")

    # ── 風險指標計算
    alphas = {"99%": 0.01, "99.5%": 0.005, "99.9%": 0.001}
    results = {}
    for label, alpha in alphas.items():
        var = calc_var(xi, tau, threshold, n_total, n_exceed, alpha)
        es  = calc_es(xi, tau, var, threshold)
        results[label] = {"VaR": var, "ES": es}

    # ── 報告
    print(f"\n{'═'*52}")
    print(f"  尾部風險指標")
    print(f"{'═'*52}")
    print(f"  {'':10s}  {'VaR（最大損失門檻）':>16s}  {'ES（超過後平均損失）':>18s}")
    for label, r in results.items():
        print(f"  {label:10s}  {r['VaR']*100:>15.2f}%  {r['ES']*100:>17.2f}%")

    # ── OTM Put 執行價建議
    var_99  = results["99%"]["VaR"]
    es_99   = results["99%"]["ES"]
    var_995 = results["99.5%"]["VaR"]
    es_995  = results["99.5%"]["ES"]

    print(f"\n{'─'*52}")
    print(f"  OTM Put 執行價建議")
    print(f"{'─'*52}")
    print(f"  基本保護（抓住 99% 的極端事件）：")
    print(f"    執行價 = 現價 × (1 - {var_99*100:.1f}%)  →  跌 {var_99*100:.1f}% 以上才獲利")
    print(f"    此時 Put 已進價內，平均損失（ES）為 {es_99*100:.1f}%")
    print(f"\n  強化保護（抓住 99.5% 的極端事件）：")
    print(f"    執行價 = 現價 × (1 - {var_995*100:.1f}%)  →  跌 {var_995*100:.1f}% 以上才獲利")
    print(f"    此時 Put 已進價內，平均損失（ES）為 {es_995*100:.1f}%")
    print(f"\n  解讀：若買 -{var_99*100:.0f}% OTM Put，")
    print(f"    在觸發的情況下，平均可以對沖 {es_99*100:.1f}% 的損失")
    print(f"    但每年需要付出 Put 的時間價值成本")

    # ── 視覺化
    plot_evt(losses_arr, threshold, xi, tau, exceedances,
             var_99, es_99, var_995, es_995)


if __name__ == "__main__":
    main()
