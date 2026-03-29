"""
HMM 多標的輪動策略（反向 ETF 版）
─────────────────────────────────────────────────────────
HMM 用台股特徵判斷市場狀態：
  bull / sideways → 持有 0050.TW（台灣 50 ETF）
  bear            → 輪動至 00632R.TW（元大台灣50反1）

核心邏輯：
  台股熊市時轉進反向 ETF，直接對沖下行風險
  同樣台幣計價，無匯率風險
  持有期間反向 ETF 可獲利，降低整體回撤

注意：反向 ETF 有波動衰變，不適合長期持有
      CONFIRM_DAYS + 週頻調倉可降低頻繁切換風險

交易成本：買賣手續費 0.1425% + 賣出證交稅 0.1%
調倉頻率：每週五檢查，倉位變化 < 5% 不動

保護機制：
  1. 反向 ETF 最長持有 MAX_INVERSE_DAYS 天，到期強制轉回股票
  2. 波動率高時動態縮減反向 ETF 倉位比例
  3. 單日跌幅超過 EMERGENCY_DROP 緊急減倉至 50%（不等週五）
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
EQUITY_SYMBOL   = "0050.TW"    # 台股 ETF（bull/sideways 持有）
INVERSE_SYMBOL  = "00632R.TW"  # 台灣50反1（bear 持有）
PERIOD        = "10y"
N_STATES      = 3
N_ITER        = 300
TRAIN_RATIO   = 0.60
CASH          = 100_000
CONFIRM_DAYS  = 2

FEE_RATE = 0.001425
TAX_RATE = 0.001

SIDEWAYS_POS = {
    "both":    1.00,
    "either":  0.70,
    "neither": 0.20,
}

BEAR_INVERSE_PCT = 0.30   # 熊市只押 30% 反向 ETF，降低誤判代價

# ── 保護機制參數 ──────────────────────────────────────────────────────────────
MAX_INVERSE_DAYS   = 40     # 反向 ETF 最長持有天數（超過強制出場）
EMERGENCY_DROP     = -0.04  # 單日跌幅門檻（跌超過 4% 緊急減倉，不等週五）
EMERGENCY_EQUITY_PCT = 0.50 # 緊急減倉後保留的股票倉位比例

FEATURE_COLS = ["ret_1d", "mom_10d", "mom_20d", "vol_ratio"]


def dynamic_inverse_pct(vol_20d: float) -> float:
    """根據近 20 日波動率動態調整反向 ETF 倉位比例，波動越高衰變越嚴重"""
    if vol_20d > 0.025:    # 日波動 > 2.5%（高波動）
        return 0.15
    elif vol_20d > 0.015:  # 日波動 1.5~2.5%（中波動）
        return 0.22
    else:                  # 日波動 < 1.5%（低波動）
        return BEAR_INVERSE_PCT


# ── 特徵工程（用台股訓練 HMM）────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_r = np.log(close / close.shift(1))
    df["ret_1d"]    = log_r
    df["mom_10d"]   = np.log(close / close.shift(10))
    df["mom_20d"]   = np.log(close / close.shift(20))
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["MA60"]      = close.rolling(60).mean()
    return df.dropna()


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
        m20 = means_df.loc[sid, "mom_20d"]
        print(f"    State {sid} ({lbl:8s})  平均日報酬={r:+.4f}  20日動能={m20:+.4f}")
    return label_map


# ── 回測（雙標的輪動）────────────────────────────────────────────────────────
def backtest(df_equity: pd.DataFrame, df_bond: pd.DataFrame,
             states_test: np.ndarray, label_map: dict,
             use_protection: bool = True) -> pd.DataFrame:
    """
    bull/sideways → 持有股票 ETF（0050）
    bear（需雙重確認）→ 輪動至反向 ETF（00632R）

    雙重確認：HMM 判熊 + 跌破 MA60 + 動能為負，兩者皆符合才切換
    未達技術確認的 bear → 降級為 sideways（持有股票，動態倉位）

    輪動時的交易成本：
      賣出原標的（手續費 + 證交稅）
      買入新標的（手續費）
    """
    equity_close = df_equity["Close"].values
    bond_close   = df_bond["Close"].reindex(df_equity.index).ffill().values
    ma60         = df_equity["MA60"].values
    mom          = df_equity["mom_10d"].values
    vol_20d      = df_equity["ret_1d"].rolling(20).std().values
    dates        = df_equity.index
    labels       = [label_map[s] for s in states_test]
    n            = len(equity_close)

    cash          = float(CASH)
    equity_shares = 0.0
    bond_shares   = 0.0
    port_val      = []
    holding       = "equity"   # 目前持有的標的

    confirmed_label  = labels[0]
    pending_label    = labels[0]
    pending_count    = 1

    inverse_hold_days  = 0      # 反向 ETF 已持有天數（機制1）

    trade_count      = 0
    trigger_m1       = 0   # 機制1：持有天數上限觸發次數
    trigger_m3_dates = []  # 機制3：動態倉位調整記錄（日期 + 比例）
    trigger_m4       = 0   # 機制4：緊急減倉觸發次數

    for i in range(n):
        eq_price   = equity_close[i]
        bond_price = bond_close[i]
        total      = cash + equity_shares * eq_price + bond_shares * bond_price
        port_val.append(total)

        if i == n - 1:
            break

        # 確認期機制
        raw_label = labels[i]
        if raw_label == pending_label:
            pending_count += 1
        else:
            pending_label = raw_label
            pending_count = 1
        if pending_count >= CONFIRM_DAYS:
            confirmed_label = pending_label

        label = confirmed_label

        # 雙重確認：HMM 判熊且技術指標也確認才算真熊市
        # 否則降級為 sideways（持有股票 ETF，動態倉位）
        if label == "bear":
            is_technical_bear = (eq_price < ma60[i]) and (mom[i] < 0)
            if not is_technical_bear:
                label = "sideways"

        # 決定目標標的
        if label == "bear":
            target_holding = "bond"
        else:
            target_holding = "equity"

        if use_protection:
            # ── 機制1：反向 ETF 持有天數上限，等下一個週五出場 ─────────────
            if holding == "bond":
                inverse_hold_days += 1
                if inverse_hold_days >= MAX_INVERSE_DAYS:
                    target_holding = "equity"
                    if inverse_hold_days == MAX_INVERSE_DAYS:  # 只在第一次觸發時計數
                        trigger_m1 += 1

            # ── 機制4：緊急減倉（單日跌幅超標，不等週五）──────────────────
            if i > 0 and holding == "equity" and equity_shares > 0:
                daily_ret = (eq_price - equity_close[i - 1]) / equity_close[i - 1]
                if daily_ret <= EMERGENCY_DROP:
                    total_now  = cash + equity_shares * eq_price
                    target_val = total_now * EMERGENCY_EQUITY_PCT
                    curr_val   = equity_shares * eq_price
                    diff       = target_val - curr_val
                    adj        = diff / eq_price
                    if adj < 0:
                        proceeds       = -adj * eq_price
                        cost           = proceeds * (FEE_RATE + TAX_RATE)
                        equity_shares += adj
                        cash          += proceeds - cost
                        trade_count   += 1
                        trigger_m4    += 1

        # 週度調倉
        if dates[i].dayofweek != 4:
            continue

        # 需要輪動
        if target_holding != holding:
            if holding == "equity":
                # 賣出股票 → 買入部分反向 ETF，機制3：波動率高時縮減倉位
                vol      = vol_20d[i] if not np.isnan(vol_20d[i]) else 0.02
                bear_pct = dynamic_inverse_pct(vol) if use_protection else BEAR_INVERSE_PCT
                if use_protection:
                    trigger_m3_dates.append((dates[i].date(), f"{bear_pct:.0%}"))

                proceeds      = equity_shares * eq_price
                sell_cost     = proceeds * (FEE_RATE + TAX_RATE)
                cash         += proceeds - sell_cost
                equity_shares = 0.0

                buy_amount        = cash * bear_pct
                buy_cost          = buy_amount * FEE_RATE
                bond_shares       = (buy_amount - buy_cost) / bond_price
                cash             -= buy_amount
                inverse_hold_days = 0
            else:
                # 賣出反向 ETF → 全倉買入股票
                proceeds      = bond_shares * bond_price
                sell_cost     = proceeds * (FEE_RATE + TAX_RATE)
                cash         += proceeds - sell_cost
                bond_shares   = 0.0

                buy_amount    = cash
                buy_cost      = buy_amount * FEE_RATE
                equity_shares = (buy_amount - buy_cost) / eq_price
                cash          = 0.0

                inverse_hold_days = 0

            holding = target_holding
            trade_count += 1

        else:
            # 同標的內調整 sideways 倉位（只對股票 ETF）
            if holding == "equity":
                above_ma = eq_price > ma60[i]
                pos_mom  = mom[i] > 0
                if above_ma and pos_mom:
                    target_pct = SIDEWAYS_POS["both"]
                elif above_ma or pos_mom:
                    target_pct = SIDEWAYS_POS["either"]
                else:
                    target_pct = SIDEWAYS_POS["neither"]

                total_now  = cash + equity_shares * eq_price
                target_val = total_now * target_pct
                curr_val   = equity_shares * eq_price
                curr_pct   = curr_val / total_now if total_now > 0 else 0

                if abs(target_pct - curr_pct) >= 0.05:
                    diff = target_val - curr_val
                    adj  = diff / eq_price
                    if adj > 0:
                        cost           = adj * eq_price * FEE_RATE
                        equity_shares += adj
                        cash          -= adj * eq_price + cost
                    elif adj < 0:
                        proceeds       = -adj * eq_price
                        cost           = proceeds * (FEE_RATE + TAX_RATE)
                        equity_shares += adj
                        cash          += proceeds - cost
                    trade_count += 1

    result = df_equity.copy()
    result["port_value"]  = port_val
    result["state_label"] = [label_map[s] for s in states_test]
    result["holding"]     = None
    result["trade_count"] = trade_count
    triggers = {
        "m1_max_days":  trigger_m1,
        "m3_entries":   trigger_m3_dates,
        "m4_emergency": trigger_m4,
    }
    return result, trade_count, triggers


# ── 績效計算 ─────────────────────────────────────────────────────────────────
def calc_metrics(port_series: pd.Series, price_series: pd.Series) -> dict:
    dr    = port_series.pct_change().dropna()
    sh    = ((dr.mean() - 0.02/252) / dr.std() * 252**0.5) if dr.std() > 0 else 0
    cum   = (1 + dr).cumprod()
    mdd   = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
    ret   = (port_series.iloc[-1] / port_series.iloc[0] - 1) * 100
    bh_dr = price_series.pct_change().dropna()
    bh_sh = ((bh_dr.mean() - 0.02/252) / bh_dr.std() * 252**0.5) if bh_dr.std() > 0 else 0
    bh_c  = (1 + bh_dr).cumprod()
    bh_dd = float(((bh_c - bh_c.cummax()) / bh_c.cummax()).min() * 100)
    bh_r  = (price_series.iloc[-1] / price_series.iloc[0] - 1) * 100
    return {"sharpe": sh, "mdd": mdd, "ret": ret,
            "bh_sh": bh_sh, "bh_dd": bh_dd, "bh_ret": bh_r}


# ── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    print(f"下載資料...")
    raw_eq   = yf.Ticker(EQUITY_SYMBOL).history(period=PERIOD, interval="1d")
    raw_bond = yf.Ticker(INVERSE_SYMBOL).history(period=PERIOD, interval="1d")

    df_eq   = raw_eq[["Open","High","Low","Close","Volume"]].copy().dropna()
    df_bond = raw_bond[["Close"]].copy().dropna()
    df_eq   = build_features(df_eq)

    print(f"  {EQUITY_SYMBOL}：{len(df_eq)} 根K線  "
          f"（{df_eq.index[0].date()} → {df_eq.index[-1].date()}）")
    print(f"  {INVERSE_SYMBOL}：{len(df_bond)} 根K線")

    # 訓練 / 測試切分
    train_end  = int(len(df_eq) * TRAIN_RATIO)
    df_train   = df_eq.iloc[:train_end]
    df_test_eq = df_eq.iloc[train_end:].copy()
    df_test_bd = df_bond.copy()
    train_date = df_eq.index[train_end].date()

    print(f"\n  訓練：{df_eq.index[0].date()} → {df_eq.index[train_end-1].date()}")
    print(f"  測試：{train_date} → {df_eq.index[-1].date()}")

    # 標準化 + 訓練 HMM
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(df_train[FEATURE_COLS].values)
    X_test      = scaler.transform(df_test_eq[FEATURE_COLS].values)
    print(f"\n訓練 HMM（{N_STATES} 個隱藏狀態，{N_ITER} 次迭代）...")
    model       = train_hmm(X_train)
    label_map   = identify_states(model, scaler)
    states_test = model.predict(X_test)

    # 狀態分佈
    print(f"\n  測試集狀態分佈：")
    for sid, lbl in label_map.items():
        cnt = int((states_test == sid).sum())
        print(f"    {lbl:8s} : {cnt:4d} 天 ({cnt/len(states_test)*100:.1f}%)")

    # 回測（有保護 vs 無保護）
    result_p,  tc_p,  triggers = backtest(df_test_eq, df_test_bd, states_test, label_map, use_protection=True)
    result_np, tc_np, _        = backtest(df_test_eq, df_test_bd, states_test, label_map, use_protection=False)
    m_p  = calc_metrics(result_p["port_value"],  df_test_eq["Close"])
    m_np = calc_metrics(result_np["port_value"], df_test_eq["Close"])

    # 績效報告（三欄並排）
    print(f"\n{'═'*70}")
    print(f"  績效報告（Out-of-Sample 測試集）")
    print(f"{'═'*70}")
    print(f"  {'':22s}  {'有保護機制':>10s}  {'無保護機制':>10s}  {'0050 B&H':>10s}")
    print(f"  {'總報酬率':22s}  {m_p['ret']:>9.2f}%  {m_np['ret']:>9.2f}%  {m_p['bh_ret']:>9.2f}%")
    print(f"  {'Sharpe Ratio':22s}  {m_p['sharpe']:>10.3f}  {m_np['sharpe']:>10.3f}  {m_p['bh_sh']:>10.3f}")
    print(f"  {'最大回撤':22s}  {m_p['mdd']:>9.2f}%  {m_np['mdd']:>9.2f}%  {m_p['bh_dd']:>9.2f}%")
    print(f"  {'實際交易次數':22s}  {tc_p:>10d}  {tc_np:>10d}")

    # 保護機制觸發報告
    print(f"\n{'─'*70}")
    print(f"  保護機制觸發記錄")
    print(f"{'─'*70}")
    print(f"  機制1 持有天數上限（>{MAX_INVERSE_DAYS}天強制出場）：{triggers['m1_max_days']} 次")
    print(f"  機制4 緊急減倉（單日跌>{abs(EMERGENCY_DROP):.0%}）    ：{triggers['m4_emergency']} 次")
    m3 = triggers["m3_entries"]
    print(f"  機制3 動態倉位（共進場反向ETF {len(m3)} 次）：")
    for dt, pct in m3:
        print(f"    {dt}  押注比例 {pct}")

    result = result_p  # 視覺化使用有保護機制的結果

    # 視覺化
    colors       = {"bull": "#d4edda", "bear": "#f8d7da", "sideways": "#fff3cd"}
    label_colors = {"bull": "#28a745", "bear": "#dc3545", "sideways": "#ffc107"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    port_scaled = (result["port_value"] / result["port_value"].iloc[0]
                   * df_test_eq["Close"].iloc[0])
    ax1.plot(result.index, df_test_eq["Close"], color="#333333",
             linewidth=1, label="0050 Close")
    ax1.plot(result.index, port_scaled, color="#0066cc",
             linewidth=1.2, linestyle="--", label="輪動策略 (scaled)")

    prev_lbl  = result["state_label"].iloc[0]
    start_idx = result.index[0]
    for i in range(1, len(result)):
        curr_lbl = result["state_label"].iloc[i]
        if curr_lbl != prev_lbl or i == len(result) - 1:
            ax1.axvspan(start_idx, result.index[i],
                        facecolor=colors[prev_lbl], alpha=0.4)
            start_idx = result.index[i]
            prev_lbl  = curr_lbl

    ax1.axvline(x=train_date, color="gray", linestyle=":", linewidth=1.5)
    patches = [mpatches.Patch(color=label_colors[k], label=k.capitalize())
               for k in ["bull", "sideways", "bear"]]
    ax1.legend(handles=patches + [ax1.lines[0], ax1.lines[1]], loc="upper left")
    ax1.set_ylabel("Price")
    ax1.set_title(f"HMM 輪動策略：{EQUITY_SYMBOL} ↔ {INVERSE_SYMBOL}")

    for lbl, col in label_colors.items():
        ind = (result["state_label"] == lbl).astype(float).rolling(5, center=True).mean()
        ax2.fill_between(result.index, ind, alpha=0.6, color=col, label=lbl.capitalize())
    ax2.set_ylabel("Regime Share (5-day)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", ncol=3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()