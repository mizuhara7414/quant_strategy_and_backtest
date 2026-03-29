"""
Walk-Forward Test — HMM 輪動策略滾動驗證
─────────────────────────────────────────────────────────
每個窗口：
  訓練集 → 訓練 HMM + Scaler
  測試集 → Forward Algorithm 推論 + 回測

結果彙整：每個窗口的報酬、Sharpe、MDD，
確認策略在不同市場環境下是否穩定獲利。
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
    forward_decode, backtest, calc_metrics,
    EQUITY_SYMBOL, INVERSE_SYMBOL, PERIOD, FEATURE_COLS
)

# ── 參數 ──────────────────────────────────────────────────────────────────────
MIN_TRAIN_DAYS = 1500   # 最小訓練天數（約 6 年）
STEP_DAYS      = 250    # 每次滾動步長（約 1 年）
MIN_TEST_DAYS  = 120    # 測試窗口最小天數（避免最後一段太短）


def run_walk_forward(df_eq: pd.DataFrame, df_bond: pd.DataFrame):
    n = len(df_eq)
    results = []
    fold = 0

    train_end = MIN_TRAIN_DAYS

    while train_end + MIN_TEST_DAYS <= n:
        test_end = min(train_end + STEP_DAYS, n)

        df_train   = df_eq.iloc[:train_end]
        df_test_eq = df_eq.iloc[train_end:test_end].copy()
        df_test_bd = df_bond.copy()

        period_start = df_test_eq.index[0].date()
        period_end   = df_test_eq.index[-1].date()
        fold += 1

        print(f"\n  Fold {fold}：訓練 {df_train.index[0].date()} → {df_train.index[-1].date()}"
              f"  ｜  測試 {period_start} → {period_end}"
              f"  （{len(df_test_eq)} 天）")

        # 訓練
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
        X_test  = scaler.transform(df_test_eq[FEATURE_COLS].values)

        try:
            model     = train_hmm(X_train)
            label_map = identify_states(model, scaler)
        except Exception as e:
            print(f"    ✗ 訓練失敗：{e}，跳過此窗口")
            train_end += STEP_DAYS
            continue

        # 推論 + 回測
        states_test = forward_decode(model, X_test)
        result, _, _ = backtest(df_test_eq, df_test_bd, states_test, label_map,
                                use_protection=True)
        m = calc_metrics(result["port_value"], df_test_eq["Close"])

        # 年化報酬（測試期天數換算）
        test_years = len(df_test_eq) / 252
        annualized = ((1 + m["ret"] / 100) ** (1 / test_years) - 1) * 100
        bh_annualized = ((1 + m["bh_ret"] / 100) ** (1 / test_years) - 1) * 100

        print(f"    策略：累積{m['ret']:+.1f}%  年化{annualized:+.1f}%  "
              f"Sharpe {m['sharpe']:.3f}  MDD {m['mdd']:.1f}%")
        print(f"    買持：累積{m['bh_ret']:+.1f}%  年化{bh_annualized:+.1f}%  "
              f"Sharpe {m['bh_sh']:.3f}  MDD {m['bh_dd']:.1f}%")

        results.append({
            "fold":         fold,
            "start":        str(period_start),
            "end":          str(period_end),
            "days":         len(df_test_eq),
            "ret":          m["ret"],
            "ann_ret":      annualized,
            "sharpe":       m["sharpe"],
            "mdd":          m["mdd"],
            "bh_ret":       m["bh_ret"],
            "bh_ann_ret":   bh_annualized,
            "bh_sharpe":    m["bh_sh"],
            "bh_mdd":       m["bh_dd"],
            "port_value":   result["port_value"].values,
            "bh_close":     df_test_eq["Close"].values,
        })

        train_end += STEP_DAYS

    return results


def print_summary(results):
    if not results:
        print("沒有任何有效窗口")
        return

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("port_value", "bh_close")}
                       for r in results])

    print(f"\n{'═'*80}")
    print(f"  Walk-Forward 彙整結果（共 {len(df)} 個窗口）")
    print(f"{'═'*80}")
    print(f"  {'Fold':>4}  {'測試區間':>23}  {'年化報酬':>8}  {'Sharpe':>7}  "
          f"{'MDD':>7}  {'BH年化':>8}  {'勝負'}")
    print(f"  {'─'*75}")

    wins = 0
    for _, row in df.iterrows():
        beat = row["ann_ret"] > row["bh_ann_ret"]
        wins += int(beat)
        flag = "✓" if beat else "✗"
        print(f"  {int(row['fold']):>4}  {row['start']} → {row['end']}  "
              f"{row['ann_ret']:>+7.1f}%  {row['sharpe']:>7.3f}  "
              f"{row['mdd']:>6.1f}%  {row['bh_ann_ret']:>+7.1f}%  {flag}")

    print(f"  {'─'*75}")
    print(f"  {'平均':>4}  {'':>23}  "
          f"{df['ann_ret'].mean():>+7.1f}%  {df['sharpe'].mean():>7.3f}  "
          f"{df['mdd'].mean():>6.1f}%  {df['bh_ann_ret'].mean():>+7.1f}%")
    print(f"\n  勝率（年化報酬超過買持）：{wins}/{len(df)} = {wins/len(df)*100:.0f}%")

    positive = (df["ann_ret"] > 0).sum()
    print(f"  正報酬窗口：{positive}/{len(df)} = {positive/len(df)*100:.0f}%")
    print(f"{'═'*80}")


def plot_results(results):
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 圖1：各窗口年化報酬對比
    ax = axes[0, 0]
    folds      = [r["fold"] for r in results]
    ann_rets   = [r["ann_ret"] for r in results]
    bh_ann     = [r["bh_ann_ret"] for r in results]
    x = np.arange(n)
    w = 0.35
    bars1 = ax.bar(x - w/2, ann_rets, w, label="策略", color="#4a90d9", alpha=0.85)
    bars2 = ax.bar(x + w/2, bh_ann,   w, label="買持0050", color="#e67e22", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylabel("年化報酬 (%)")
    ax.set_title("各窗口年化報酬")
    ax.legend(fontsize=9)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    # 圖2：各窗口 Sharpe 對比
    ax = axes[0, 1]
    sharpes = [r["sharpe"] for r in results]
    bh_sh   = [r["bh_sharpe"] for r in results]
    ax.bar(x - w/2, sharpes, w, label="策略", color="#2ecc71", alpha=0.85)
    ax.bar(x + w/2, bh_sh,   w, label="買持0050", color="#e74c3c", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#27ae60", linewidth=1, linestyle=":", label="Sharpe=1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("各窗口 Sharpe Ratio")
    ax.legend(fontsize=9)

    # 圖3：各窗口 MDD 對比
    ax = axes[1, 0]
    mdds   = [r["mdd"] for r in results]
    bh_mdd = [r["bh_mdd"] for r in results]
    ax.bar(x - w/2, mdds,   w, label="策略", color="#9b59b6", alpha=0.85)
    ax.bar(x + w/2, bh_mdd, w, label="買持0050", color="#95a5a6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylabel("MDD (%)")
    ax.set_title("各窗口最大回撤（越小越好）")
    ax.legend(fontsize=9)

    # 圖4：拼接各窗口的淨值曲線
    ax = axes[1, 1]
    colors_strat = plt.cm.Blues(np.linspace(0.4, 0.9, n))
    colors_bh    = plt.cm.Oranges(np.linspace(0.4, 0.9, n))
    for i, r in enumerate(results):
        pv  = r["port_value"] / r["port_value"][0] * 100
        bh  = r["bh_close"]  / r["bh_close"][0]  * 100
        idx = np.arange(len(pv))
        ax.plot(idx, pv, color=colors_strat[i], linewidth=1.2,
                label=f"策略 F{r['fold']}" if i == 0 else "")
        ax.plot(idx, bh, color=colors_bh[i],    linewidth=1.0,
                linestyle="--",
                label=f"買持 F{r['fold']}" if i == 0 else "")
    ax.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel("測試天數")
    ax.set_ylabel("相對淨值（起始=100）")
    ax.set_title("各窗口淨值曲線（藍=策略，橘虛=買持）")

    plt.suptitle("Walk-Forward Test 結果彙整", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    print("下載資料...")
    raw_eq   = yf.Ticker(EQUITY_SYMBOL).history(period=PERIOD, interval="1d")
    raw_bond = yf.Ticker(INVERSE_SYMBOL).history(period=PERIOD, interval="1d")

    df_eq   = raw_eq[["Open","High","Low","Close","Volume"]].copy().dropna()
    df_bond = raw_bond[["Close"]].copy().dropna()
    df_eq   = build_features(df_eq)

    print(f"  {EQUITY_SYMBOL}：{len(df_eq)} 根K線  "
          f"（{df_eq.index[0].date()} → {df_eq.index[-1].date()}）")
    print(f"\n設定：最小訓練 {MIN_TRAIN_DAYS} 天，步長 {STEP_DAYS} 天\n")

    results = run_walk_forward(df_eq, df_bond)
    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()
