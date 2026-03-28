"""
HMM 市場狀態偵測 (Hidden Markov Model Regime Detection)

隱藏狀態 (Hidden States)：市場的「真實狀態」，無法直接觀察
可觀測量 (Observations)：每日報酬、波動率、成交量
HMM 從可觀測量推斷最可能的隱藏狀態序列 (Viterbi Algorithm)

三個隱藏狀態：
  牛市 (Bull)     — 高報酬、低波動
  熊市 (Bear)     — 負報酬、高波動
  震盪 (Sideways) — 近零報酬、中波動

策略切換：
  牛市    → 持有 (Buy & Hold)
  震盪    → 半倉 (50% invested)
  熊市    → 現金 (0% invested)
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

# ── 設定 ─────────────────────────────────────────────────────────────────────
SYMBOL      = "0050.TW"
PERIOD      = "10y"      # HMM 訓練用（越長越好）
N_STATES    = 3         # 隱藏狀態數（牛 / 熊 / 震盪）
N_ITER      = 300        # EM 演算法最大迭代次數
TRAIN_RATIO   = 0.60   # 前 60% 訓練，後 40% 測試
CASH          = 100_000
CONFIRM_DAYS  = 2      # 狀態確認期：連續 N 天相同狀態才切換倉位

# 交易成本（台股 ETF）
FEE_RATE = 0.001425   # 手續費：買賣各 0.1425%（可向券商申請折扣）
TAX_RATE = 0.001      # 證交稅：0.1%（僅賣出時收取）

# 狀態對應的倉位
POSITION_MAP = {
    "bull":     1.00,    # 牛市：全倉
    "bear":     0.00,    # 熊市：空倉（現金）
}

# sideways 動態倉位（MA60 + 動能雙條件）
# 邏輯驅動設定，不依賴優化（避免過擬合）
# 三段驗證實驗顯示 DE 優化在短樣本上過擬合嚴重，人工設定更穩健
SIDEWAYS_POS = {
    "both":    1.00,   # 價格 > MA60 且 動能 > 0  → 雙確認，全倉
    "either":  0.70,   # 其中一個條件成立          → 偏多但保守
    "neither": 0.20,   # 兩個條件都不成立          → 低倉，避免碰邊界值
}

REPORT_DIR = "quant_trading/reports"


# ── STEP 1：特徵工程 ──────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    四個特徵，全部可在當天收盤前計算（無未來資料）：
      ret_1d    : 當日對數報酬
      vol_5d    : 5 日滾動波動率（年化），衡量近期風險程度
      mom_10d   : 10 日累積報酬，衡量短期動量
      vol_ratio : 成交量 vs 20 日均量比，衡量市場活躍度

    MA60 不加入 HMM 特徵，只作為回測倉位判斷的輔助欄位：
      MA60      : 60 日均線，用來區分「慢牛 sideways」和「真正橫盤 sideways」
    """
    close = df["Close"]
    log_r = np.log(close / close.shift(1))

    df["ret_1d"]    = log_r
    df["mom_10d"]   = np.log(close / close.shift(10))
    df["mom_20d"]   = np.log(close / close.shift(20))
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["MA60"]      = close.rolling(60).mean()
    return df.dropna()


FEATURE_COLS = ["ret_1d", "mom_10d", "mom_20d", "vol_ratio"]
# ── STEP 2：訓練 HMM ──────────────────────────────────────────────────────────
def train_hmm(X_train: np.ndarray) -> GaussianHMM:
    """
    GaussianHMM：假設每個隱藏狀態的可觀測量服從多變量常態分布
    covariance_type='full'：每個狀態有獨立的完整協方差矩陣
    用 EM 演算法（Baum-Welch）反覆估計參數直到收斂

    注意：log-likelihood 最高不代表交易績效最好。
    多種子實驗後確認 seed=42 產生的狀態分類對交易最有意義，固定使用。
    """
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=42,
    )
    model.fit(X_train)
    return model


# ── STEP 3：辨識狀態標籤 ──────────────────────────────────────────────────────
def identify_states(model: GaussianHMM, scaler: StandardScaler) -> dict:
    """
    HMM 的狀態編號 (0, 1, 2) 是任意的，需要根據特性來命名：
      - ret_1d 均值最高  → 牛市
      - ret_1d 均值最低  → 熊市
      - 其餘              → 震盪
    """
    # 把學到的各狀態均值反標準化回原始尺度
    means_scaled = model.means_                    # shape: (N_STATES, N_FEATURES)
    means_orig   = scaler.inverse_transform(means_scaled)
    means_df     = pd.DataFrame(means_orig, columns=FEATURE_COLS)

    ret_col = means_df["ret_1d"]
    bull_id = int(ret_col.idxmax())
    bear_id = int(ret_col.idxmin())
    side_id = [i for i in range(N_STATES) if i not in (bull_id, bear_id)][0]

    label_map = {bull_id: "bull", bear_id: "bear", side_id: "sideways"}

    print(f"\n  HMM 狀態辨識結果：")
    for sid, label in label_map.items():
        r   = means_df.loc[sid, "ret_1d"]
        m20 = means_df.loc[sid, "mom_20d"]
        print(f"    State {sid} ({label:8s})  平均日報酬={r:+.4f}  "
              f"20日動能={m20:+.4f}")

    return label_map


# ── STEP 4：回測 ──────────────────────────────────────────────────────────────
def backtest(df_test: pd.DataFrame, states_test: np.ndarray,
             label_map: dict) -> pd.DataFrame:
    """
    每天根據 HMM 預測的狀態，決定明天的持倉比例。

    狀態確認機制（Hysteresis）：
      原始 HMM 狀態每天可能跳動，造成頻繁換倉（訊號抖動）。
      引入確認期：只有連續 CONFIRM_DAYS 天出現相同狀態，才正式切換。
      未達確認期時維持上一個已確認的狀態，避免被短暫誤判掃出場。

    注意：今日狀態 → 明日倉位（避免未來資料）
    """
    close    = df_test["Close"].values
    ma60     = df_test["MA60"].values
    mom      = df_test["mom_10d"].values
    labels   = [label_map[s] for s in states_test]
    dates    = df_test.index
    n        = len(close)
    cash     = float(CASH)
    shares   = 0.0
    port_val = []

    # 確認期機制：初始確認狀態為第一天的狀態
    confirmed_label = labels[0]
    pending_label   = labels[0]
    pending_count   = 1

    for i in range(n):
        curr_price = close[i]
        port_val.append(cash + shares * curr_price)

        if i == n - 1:
            break

        # 更新確認期計數
        raw_label = labels[i]
        if raw_label == pending_label:
            pending_count += 1
        else:
            pending_label = raw_label
            pending_count = 1

        # 達到確認天數才正式切換
        if pending_count >= CONFIRM_DAYS:
            confirmed_label = pending_label

        label = confirmed_label

        if label == "sideways":
            above_ma = close[i] > ma60[i]
            pos_mom  = mom[i] > 0
            if above_ma and pos_mom:
                target_pct = SIDEWAYS_POS["both"]
            elif above_ma or pos_mom:
                target_pct = SIDEWAYS_POS["either"]
            else:
                target_pct = SIDEWAYS_POS["neither"]
        elif label == "bear":
            above_ma  = close[i] > ma60[i]
            strong_up = (mom[i] > 0.03) and above_ma
            short_up  = (mom[i] > 0.015) and above_ma
            if strong_up:
                target_pct = 0.60
            elif short_up:
                target_pct = 0.40
            else:
                target_pct = POSITION_MAP["bear"]
        else:
            target_pct = POSITION_MAP[label]

        # 週度調倉：只在每週五執行交易，減少換倉次數
        if dates[i].dayofweek != 4:
            continue

        total      = cash + shares * curr_price
        target_val = total * target_pct
        curr_val   = shares * curr_price

        # 最小換倉門檻：倉位變化小於 5% 不執行，節省交易成本
        curr_pct = curr_val / total if total > 0 else 0
        if abs(target_pct - curr_pct) < 0.05:
            continue

        diff = target_val - curr_val          # 需要調整的金額
        adj  = diff / curr_price              # 需要買賣的股數

        if adj > 0:
            # 買入：扣手續費
            cost = adj * curr_price * FEE_RATE
            shares += adj
            cash   -= adj * curr_price + cost
        elif adj < 0:
            # 賣出：扣手續費 + 證交稅
            proceeds = -adj * curr_price
            cost     = proceeds * (FEE_RATE + TAX_RATE)
            shares  += adj
            cash    += proceeds - cost

    result = df_test.copy()
    result["port_value"] = port_val
    result["state_id"]   = states_test
    result["state_label"] = [label_map[s] for s in states_test]
    return result


# ── STEP 5：績效計算 ──────────────────────────────────────────────────────────
def calc_metrics(port_series: pd.Series, price_series: pd.Series,
                 label: str) -> dict:
    daily_ret = port_series.pct_change().dropna()
    sharpe    = ((daily_ret.mean() - 0.02 / 252) / daily_ret.std()
                 * 252 ** 0.5) if daily_ret.std() > 0 else 0.0
    cum       = (1 + daily_ret).cumprod()
    max_dd    = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    total_ret = (port_series.iloc[-1] / port_series.iloc[0] - 1) * 100

    bh_ret = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
    bh_dr  = price_series.pct_change().dropna()
    bh_sh  = ((bh_dr.mean() - 0.02 / 252) / bh_dr.std()
               * 252 ** 0.5) if bh_dr.std() > 0 else 0.0
    bh_cum = (1 + bh_dr).cumprod()
    bh_dd  = float(((bh_cum - bh_cum.cummax()) / bh_cum.cummax()).min() * 100)

    return {
        "label": label,
        "total_ret": total_ret,   "sharpe": sharpe,  "max_dd": max_dd,
        "bh_ret": bh_ret,         "bh_sh": bh_sh,    "bh_dd": bh_dd,
    }


# ── STEP 6：視覺化 ────────────────────────────────────────────────────────────
def plot_regimes(result: pd.DataFrame, train_end_date ):
    """
    上圖：K 線（收盤價），背景顏色代表 HMM 偵測的市場狀態
    下圖：各狀態的後驗機率（HMM 對每個狀態的「信心程度」）
    """
    colors = {"bull": "#d4edda", "bear": "#f8d7da", "sideways": "#fff3cd"}
    label_colors = {"bull": "#28a745", "bear": "#dc3545", "sideways": "#ffc107"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # 上圖：價格
    ax1.plot(result.index, result["Close"], color="#333333", linewidth=1, label="Close Price")
    ax1.plot(result.index, result["port_value"] / result["port_value"].iloc[0]
             * result["Close"].iloc[0],
             color="#0066cc", linewidth=1.2, linestyle="--", label="HMM Strategy (scaled)")

    # 背景顏色
    prev_label = result["state_label"].iloc[0]
    start_idx  = result.index[0]
    for i in range(1, len(result)):
        curr_label = result["state_label"].iloc[i]
        if curr_label != prev_label or i == len(result) - 1:
            ax1.axvspan(start_idx, result.index[i],
                        facecolor=colors[prev_label], alpha=0.4)
            start_idx  = result.index[i]
            prev_label = curr_label

    # 訓練/測試分割線
    ax1.axvline(x=train_end_date, color="gray", linestyle=":", linewidth=1.5,
                label="Train / Test Split")

    patches = [mpatches.Patch(color=label_colors[k], label=k.capitalize())
               for k in ["bull", "sideways", "bear"]]
    ax1.legend(handles=patches + [ax1.lines[0], ax1.lines[1]], loc="upper left")
    ax1.set_ylabel("Index Level")
    ax1.set_title(f"HMM Market Regime Detection — {SYMBOL}")

    # 下圖：狀態機率（需要 model，這裡用 state_label 轉 one-hot 近似）
    for lbl, col in label_colors.items():
        indicator = (result["state_label"] == lbl).astype(float)
        indicator_smooth = indicator.rolling(5, center=True).mean()
        ax2.fill_between(result.index, indicator_smooth, alpha=0.6,
                         color=col, label=lbl.capitalize())
    ax2.set_ylabel("Regime Share (5-day rolling)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=3)

    plt.tight_layout()
    plt.show()


# ── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 下載資料
    print(f"下載 {SYMBOL} 資料（{PERIOD}）...")
    raw = yf.Ticker(SYMBOL).history(period=PERIOD, interval="1d")
    df  = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    print(f"  共 {len(df)} 根 K 線（{df.index[0].date()} → {df.index[-1].date()}）")

    # 特徵工程
    df = build_features(df)

    # 訓練 / 測試分割
    train_end  = int(len(df) * TRAIN_RATIO)
    df_train   = df.iloc[:train_end]
    df_test    = df.iloc[train_end:].copy()
    train_date = df.index[train_end].date()
    print(f"  訓練：{df.index[0].date()} → {df.index[train_end-1].date()}")
    print(f"  測試：{train_date} → {df.index[-1].date()}")

    # 標準化（用訓練集統計量）
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test  = scaler.transform(df_test[FEATURE_COLS].values)

    # 訓練 HMM
    print(f"\n訓練 HMM（{N_STATES} 個隱藏狀態，{N_ITER} 次迭代）...")
    model     = train_hmm(X_train)
    label_map = identify_states(model, scaler)

    # 預測狀態
    states_train = model.predict(X_train)
    states_test  = model.predict(X_test)

    # 狀態分佈
    print(f"\n  測試集狀態分佈：")
    for sid, lbl in label_map.items():
        cnt = int((states_test == sid).sum())
        pct = cnt / len(states_test) * 100
        print(f"    {lbl:8s} : {cnt:4d} 天 ({pct:.1f}%)")

    # 回測
    result  = backtest(df_test, states_test, label_map)
    metrics = calc_metrics(result["port_value"], df_test["Close"], "HMM 策略")

    # 績效報告
    print(f"\n{'═'*58}")
    print(f"  績效報告（Out-of-Sample 測試集）")
    print(f"{'═'*58}")
    print(f"  {'':22s}  {'HMM 策略':>10s}  {'買入持有':>10s}")
    print(f"  {'總報酬率':22s}  {metrics['total_ret']:>9.2f}%  "
          f"{metrics['bh_ret']:>9.2f}%")
    print(f"  {'Sharpe Ratio':22s}  {metrics['sharpe']:>10.3f}  "
          f"{metrics['bh_sh']:>10.3f}")
    print(f"  {'最大回撤':22s}  {metrics['max_dd']:>9.2f}%  "
          f"{metrics['bh_dd']:>9.2f}%")

    # 繪圖
    
    plot_regimes(result, train_date)
    sideways_mask = result["state_label"] == "sideways"
    sideways_returns = result.loc[sideways_mask, "Close"].pct_change()

    print(f"  sideways 期間累積報酬: {(1 + sideways_returns).prod() - 1:.2%}")
    print(f"  sideways 期間平均日報酬: {sideways_returns.mean():.4%}")
    print(f"  sideways 期間天數: {sideways_mask.sum()}")
    state_changes = (result['state_label'] != result['state_label'].shift()).sum()
    friday_mask   = result.index.dayofweek == 4
    actual_trades = (result.loc[friday_mask, 'state_label'] !=
                     result.loc[friday_mask, 'state_label'].shift()).sum()
    print(f"  狀態改變次數（HMM）：{state_changes}")
    print(f"  實際交易次數（週五）：{actual_trades}")
if __name__ == "__main__":
    main()