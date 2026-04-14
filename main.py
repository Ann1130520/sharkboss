"""
Project: AI 龍蝦量化戰情室 - 核心執行引擎 (v9.1 退回穩定版 + 終極防禦)
Author: Gemini Collaborator & User
Date: 2026-04-13
"""

import os
import yfinance as yf
import pandas as pd
import sqlite3
import json
import feedparser
import requests
from datetime import datetime

# 🌟 換回最穩定的舊版 SDK
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 系統與使用者設定區
# ==========================================
DB_PATH = "/content/drive/MyDrive/lobster_grandmaster.db"
if not os.path.exists(os.path.dirname(DB_PATH)):
    DB_PATH = "lobster_grandmaster.db" 

TRUTH_SOCIAL_USER = "realDonaldTrump"

def get_secret(key_name):
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except (ImportError, ModuleNotFoundError):
        return os.getenv(key_name)

GEMINI_API_KEY = get_secret('GEMINI_API_KEY')
NEWS_API_KEY = get_secret('MY_NEWS_API_KEY')
MAKE_WEBHOOK_URL = get_secret('MAKE_WEBHOOK_URL')
HF_TOKEN = get_secret('HF_TOKEN')

if HF_TOKEN: 
    os.environ["HF_TOKEN"] = HF_TOKEN

# ==========================================
# 2. 歷史記憶庫 (SQLite)
# ==========================================
class DatabaseLogger:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS grandmaster_logs (
                date TEXT PRIMARY KEY, vix REAL, us10y REAL, twii_change REAL, gs_ratio REAL,
                news_sent REAL, ts_sent REAL, is_bear INTEGER, atr_percent REAL,
                lunar_phase REAL, bazi_element TEXT, ai_report TEXT, ai_signal INTEGER, ai_conf REAL,
                risk_level INTEGER, focus TEXT, w_spy REAL, w_tlt REAL, w_gold REAL, w_silver REAL, w_oil REAL, w_vix REAL
            )
        ''')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS user_settings (key TEXT PRIMARY KEY, value TEXT, update_time TEXT)')
        try: self.cursor.execute('ALTER TABLE grandmaster_logs ADD COLUMN w_qqq REAL DEFAULT 0.0')
        except: pass
        self.conn.commit()

    def get_user_settings(self):
        try:
            self.cursor.execute("SELECT value FROM user_settings WHERE key='risk_level'")
            res_risk = self.cursor.fetchone()
            risk = int(res_risk[0]) if res_risk else 3
            self.cursor.execute("SELECT value FROM user_settings WHERE key='investment_focus'")
            res_focus = self.cursor.fetchone()
            focus = str(res_focus[0]) if res_focus else 'FUNDS'
            return risk, focus
        except Exception as e:
            return 3, 'FUNDS'

    def log_today(self, data):
        self.cursor.execute('INSERT OR REPLACE INTO grandmaster_logs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)
        self.conn.commit()

    # ==========================================
    # 🌟 新增：讀取昨日/最新 AI 報告作為上下文
    # ==========================================
    def get_yesterday_context(self):
        try:
            # 從你的 grandmaster_logs 表格撈取日期最新的一筆 ai_report
            self.cursor.execute("SELECT ai_report FROM grandmaster_logs ORDER BY date DESC LIMIT 1")
            result = self.cursor.fetchone()
            if result and result[0]:
                return f"昨日 AI 觀點回顧：{result[0]}"
            else:
                return "無昨日紀錄。"
        except Exception as e:
            print(f"[警告] 讀取昨日紀錄失敗: {e}")
            return "無法讀取昨日紀錄。"

# ==========================================
# 3. 數據中心
# ==========================================
class DataCenter:
    def __init__(self, news_key):
        self.news_key = news_key
        print("[系統] 正在載入 FinBERT 模型...")
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def get_macro_and_armor_data(self):
        print("[系統] 擷取跨市場金融數據...")
        try:
            tickers = ["^VIX", "CL=F", "GC=F", "SI=F", "SPY", "TLT", "^TWII", "^TNX", "JPY=X", "QQQ"]
            data = yf.download(tickers, period="250d", progress=False)
            close = data['Close']
            high, low = data['High']['SPY'], data['Low']['SPY']

            vix = float(close['^VIX'].iloc[-1])
            us10y = float(close['^TNX'].iloc[-1])
            twii_chg = float(close['^TWII'].pct_change().iloc[-1])
            gs_ratio = float(close['GC=F'].iloc[-1] / close['SI=F'].iloc[-1])
            spy_current = float(close['SPY'].iloc[-1])
            jpy_current = float(close['JPY=X'].iloc[-1])

            idx_mo = -20 if len(close) >= 20 else 0
            yield_chg = us10y - float(close['^TNX'].iloc[idx_mo])
            jpy_chg = (jpy_current / float(close['JPY=X'].iloc[idx_mo]) - 1) * 100
            oil_chg = (float(close['CL=F'].iloc[-1]) / float(close['CL=F'].iloc[idx_mo]) - 1) * 100
            gold_chg = (float(close['GC=F'].iloc[-1]) / float(close['GC=F'].iloc[idx_mo]) - 1) * 100

            is_bear = int(spy_current < close['SPY'].rolling(200).mean().iloc[-1])
            spy_trend = int(spy_current > close['SPY'].rolling(60).mean().iloc[-1])
            tlt_trend = int(close['TLT'].iloc[-1] > close['TLT'].rolling(60).mean().iloc[-1])
            gld_trend = int(close['GC=F'].iloc[-1] > close['GC=F'].rolling(60).mean().iloc[-1])
            qqq_trend = int(close['QQQ'].iloc[-1] > close['QQQ'].rolling(60).mean().iloc[-1])

            spy_mom = close['SPY'].pct_change(60).iloc[-1]
            qqq_mom = close['QQQ'].pct_change(60).iloc[-1]
            qqq_beats = int(qqq_mom > spy_mom)

            tr = pd.concat([high-low, abs(high-close['SPY'].shift(1)), abs(low-close['SPY'].shift(1))], axis=1).max(axis=1)
            atr_pct = float(tr.rolling(14).mean().iloc[-1]) / spy_current
            spy_20 = int(spy_current > close['SPY'].rolling(20).mean().iloc[-1])

            delta = close['SPY'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = float(100 - (100 / (1 + gain/loss)).iloc[-1])

            return vix, us10y, twii_chg, gs_ratio, is_bear, atr_pct, spy_trend, tlt_trend, gld_trend, jpy_current, yield_chg, jpy_chg, oil_chg, gold_chg, qqq_beats, qqq_trend, spy_20, rsi
        except Exception as e:
            print(f"[錯誤] 數據失敗: {e}"); return 20.0, 4.0, 0.0, 80.0, 0, 0.01, 1, 1, 1, 150.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 1, 50.0

    def get_news_sentiment(self):
        if not self.news_key: return 0.0
        try:
            url = f"https://newsapi.org/v2/everything?q=Federal Reserve OR S&P 500&language=en&sortBy=relevancy&apiKey={self.news_key}"
            arts = requests.get(url).json().get("articles", [])[:3]
            return sum([self.finbert(a['title'][:512])[0]['score'] * (1 if self.finbert(a['title'][:512])[0]['label']=='positive' else -1) for a in arts]) / len(arts) if arts else 0.0
        except: return 0.0

    def get_truth_social(self, user):
        for url in [f"https://rsshub.app/truthsocial/user/{user}", f"https://rsshub.moeyu.xyz/truthsocial/user/{user}"]:
            try:
                feed = feedparser.parse(url)
                if feed.entries:
                    scores = [self.finbert(p.title[:512])[0]['score'] * (1 if self.finbert(p.title[:512])[0]['label']=='positive' else -1) for p in feed.entries[:3]]
                    return sum(scores) / len(scores)
            except: continue
        return 0.0

    @staticmethod
    def get_alternative_factors(date):
        days = (date - datetime(2000, 1, 6, 18, 14)).total_seconds() / 86400
        return (days / 29.53058868) % 1, ["金", "木", "水", "火", "土"][date.toordinal() % 5]

# ==========================================
# 4. AI 決策引擎 (終極輪詢防彈版)
# ==========================================
class AIEngine:
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # 🌟 殺手鐧：模型自動備援清單 (根據你的 API 後台真實配額量身打造！)
        self.candidate_models = [
            'gemini-3.1-flash-lite',  # 🏆 首選：一天 500 次超大容量！
            'gemini-3-flash',         # 備用：一天 20 次
            'gemini-2.5-flash-lite',  # 備用：一天 20 次
            'gemini-3.1-flash'        # 盲測備用寫法
        ]

    def analyze(self, vix, us10y, twii, gs, atr, lunar, bazi, news_s, ts_s, past, jpy_current, yield_change, jpy_change, oil_change, gold_change):
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        print("[系統] 啟動 AI 決策引擎 (進入自動尋標模式)...")
        safety = {cat: HarmBlockThreshold.BLOCK_NONE for cat in [HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]}
        
        prompt = f"""
        你是一位頂尖量化經理。請基於以下數據提供分析：
        【總經指標】VIX: {vix:.2f} | 10Y美債: {us10y:.2f}% (變動: {yield_change:+.2f}%) | 金銀比: {gs:.2f}
        【匯率與地緣】日圓匯率: {jpy_current:.2f} (月變動: {jpy_change:+.2f}%) | 原油月變動: {oil_change:+.2f}% | 黃金月變動: {gold_change:+.2f}%
        【另類與情緒】台灣加權變化: {twii:.2%} | FinBERT新聞情緒: {news_s:.2f} | 五行: {bazi}

        🚨 核心判斷準則：
        1. VIX > 30 或情緒 < -0.2 為恐慌，建議防禦。
        2. 殖利率月增 > 0.2% 則科技股承壓。
        3. 日圓 > 150 且貶值過快需注意套利平倉風險。

        🎯 報告撰寫要求：
        你的 "report" 必須是一篇約 200 字的流暢文章，並且「絕對要直接引用」上面給你的具體數字！
        例如：「當前 FinBERT 新聞情緒指標為 {news_s:.2f}...」。解釋我們為何採取目前的策略。

        回傳純 JSON 格式（不要有任何 markdown 符號）：
        {{
            "report": "在這裡寫下包含具體數字的分析與配置建議...",
            "signal": 1,
            "confidence": 0.85
        }}
        """
        
        last_error = ""
        
        # 🌟 開始自動測試所有模型
        for model_name in self.candidate_models:
            print(f"   -> 嘗試呼叫模型: {model_name} ...")
            try:
                model = genai.GenerativeModel(model_name)
                res_obj = model.generate_content(prompt, safety_settings=safety)
                res_text = res_obj.text.strip()
                
                # 暴力挖出 JSON
                import re
                match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if match:
                    res_text = match.group(0)
                
                res = json.loads(res_text)
                print(f"   ✅ 模型 {model_name} 成功產出報告！")
                return res.get('report', ''), res.get('signal', 0), res.get('confidence', 0.5)
                
            except Exception as e: 
                err_msg = str(e)
                print(f"   ❌ {model_name} 失敗: {err_msg[:50]}...")
                last_error = err_msg
                continue # 遇到錯誤不當機，直接換下一個模型！

        # 如果全部陣亡才回傳 Telegram
        return f"⚠️ 【系統異常】所有模型皆無權限或無額度！\n最後死因：{last_error[:150]}", 0, 0.0
# ==========================================
# 5. 多資產配置矩陣 (🌟 V8.3：純動能、極端避險與動態曝險)
# ==========================================
class MultiAssetStrategy:
    def __init__(self, risk, focus):
        self.risk = risk
        self.focus = focus
        self.base_alloc = {
            1: [0.00, 0.95, 0.05], 2: [0.20, 0.70, 0.10],
            3: [0.50, 0.35, 0.15], 4: [0.75, 0.10, 0.15], 5: [0.85, 0.00, 0.15]
        }

    # 🌟 優化：完整啟用閒置參數 (is_bear, atr, conf, gs_ratio)
    def calculate_weights(self, is_bear, atr, signal, conf, gs_ratio, spy_trend_up, tlt_trend_up, gld_trend_up, qqq_beats_spy, qqq_trend_up, spy_20_up, rsi, vix):
        stock_p, bond_p, other_p = self.base_alloc[self.risk]
        w = {'SPY': 0.0, 'QQQ': 0.0, 'TLT': 0.0, 'GOLD': 0.0, 'SILVER': 0.0, 'OIL': 0.0, 'VIX': 0.0}

        # 👑【最高優先級】VIX 極端恐慌 (修復美債陷阱)
        if vix >= 35:
            w['SPY'] = stock_p * 0.1
            w['TLT'] = bond_p if tlt_trend_up else 0.0
            w['GOLD'] = other_p * 2 if gld_trend_up else other_p
            w['VIX'] = other_p # 加入VIX避險

        # 🥈【第二優先級】均線破裂 vs 乖離過大 (V轉保護機制)
        elif not spy_trend_up and rsi < 30:
            w['SPY'] = stock_p * 0.5            # 留一半底倉等反彈
            w['TLT'] = bond_p if tlt_trend_up else 0.0 # 避開美債陷阱
            w['GOLD'] = other_p

        # 🥉【第三優先級】大盤強牛市 vs 科技動能 (科技輪動引擎)
        elif spy_trend_up and spy_20_up:
            if qqq_beats_spy and qqq_trend_up:
                w['QQQ'] = stock_p * 0.7
                w['SPY'] = stock_p * 0.3
            else:
                w['SPY'] = stock_p
            w['TLT'] = bond_p
            w['GOLD'] = other_p

        # 🛡️【第四優先級】大盤弱牛市 (高檔震盪防禦雙巴)
        elif spy_trend_up and not spy_20_up:
            w['SPY'] = stock_p * 0.5
            w['TLT'] = bond_p
            w['GOLD'] = other_p

        # 🐻【第五優先級】傳統熊市 (股跌、債漲)
        elif not spy_trend_up and tlt_trend_up:
            w['SPY'] = stock_p * 0.2
            w['TLT'] = bond_p + stock_p * 0.8
            w['GOLD'] = other_p * 0.6
            w['VIX']  = other_p * 0.4

        # ☠️【第六優先級】股債雙殺 (通膨末日情境，完美避開 2022 陷阱)
        else:
            if gld_trend_up:
                w['GOLD'], w['VIX'] = 0.8, 0.2
            else:
                # 全部跌：極端現金為王 (僅保留測試底倉)
                w['SPY'], w['TLT'], w['GOLD'] = 0.05, 0.05, 0.1

        # ==========================================
        # 🛑 【新增】長線大局觀 (is_bear) 濾網
        # ==========================================
        # 若確認為長線大熊市，即便上方觸發了短多訊號(如反彈)，強制將股票總權重砍半
        if is_bear:
            w['SPY'] = min(w['SPY'], stock_p * 0.5)
            w['QQQ'] = min(w['QQQ'], stock_p * 0.5)

        # ==========================================
        # 附屬資產微調 (Focus 邏輯)
        # ==========================================
        # 僅在非股債雙殺的健康狀態下支援 FUNDS 偏好加成
        if w.get('GOLD', 0) <= other_p and sum(w.values()) > 0:
            if self.focus == 'METALS': 
                # 🥈【新增】利用 gs_ratio (金銀比) 動態決定主角
                # 若金銀比大於 80 (白銀超跌/便宜)，拉高白銀比例至 50%
                silver_share = 0.5 if gs_ratio > 80 else 0.3 
                w['GOLD'], w['SILVER'] = other_p * (1 - silver_share), other_p * silver_share
            
            elif self.focus == 'FUTURES': w['OIL'], w['VIX'] = other_p*0.6, other_p*0.4
            elif self.focus == 'FUNDS':
                w['SPY'] += (other_p * 0.5)
                w['QQQ'] += (other_p * 0.5)
            elif self.focus == 'BONDS': w['TLT'] += other_p


        # ==========================================
        # 📉 【新增】動態總曝險控管 (ATR 與 信心度)
        # ==========================================
        # 處理信心度 (確保在 0~1 之間)
        conf_scalar = conf if conf <= 1.0 else (conf / 100.0 if conf > 1.0 else 1.0)
        
        # 處理異常波動：若 atr 大於自定義危險值(此處以 5.0 為範例，請依你的指標尺度修改)，且未持有 VIX 避險，則整體降倉 20%
        # 注意：請將 5.0 替換成適合你系統的 ATR 閾值，例如 atr > 1.5 * atr_20
        vol_scalar = 0.8 if (atr > 5.0 and w.get('VIX', 0) == 0) else 1.0 
        
        # 最終乘數：若信心不足或波動過大，總和將小於 1，剩下的空間自然轉為「現金 (Cash)」
        total_multiplier = conf_scalar * vol_scalar

        # 自動歸一化並應用曝險乘數
        total = sum(w.values())
        if total > 0:
            return {k: round((v / total) * total_multiplier, 3) for k, v in w.items()}
        else:
            return w
# ==========================================
# 6. 系統主循環與發射器
# ==========================================
import math

def is_missing(val):
    """檢查變數是否為 None 或 nan"""
    return val is None or (isinstance(val, float) and math.isnan(val))

def run_system():
    print("\n========== 🦞 AI 龍蝦大師 v8.2 (純粹動能與極端避險版) ==========")
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')

    db = DatabaseLogger()
    data = DataCenter(NEWS_API_KEY)
    ai = AIEngine(GEMINI_API_KEY)

    # 🌟 優化：從資料庫動態讀取風險等級與投資偏好
    current_risk, current_focus = db.get_user_settings()
    print(f"[系統] 成功載入戰情室設定 -> 風險等級: {current_risk} | 投資偏好: {current_focus}")

    strat = MultiAssetStrategy(current_risk, current_focus)

    # 1. 抓取所有數據
    vix, us10y, twii, gs, is_bear, atr, spy_trend, tlt_trend, gld_trend, jpy_current, yield_change, jpy_change, oil_change, gold_change, qqq_beats_spy, qqq_trend_up, spy_20_up, rsi = data.get_macro_and_armor_data()
    news_s = data.get_news_sentiment()

    ts_raw = data.get_truth_social(TRUTH_SOCIAL_USER)
    ts_s = ts_raw if ts_raw is not None else 0.0

    lunar, bazi = data.get_alternative_factors(now)
    past = db.get_yesterday_context()

    # 2. AI 思考與策略計算
    report, signal, conf = ai.analyze(vix, us10y, twii, gs, atr, lunar, bazi, news_s, ts_s, past, jpy_current, yield_change, jpy_change, oil_change, gold_change)
    weights = strat.calculate_weights(is_bear, atr, signal, conf, gs, spy_trend, tlt_trend, gld_trend, qqq_beats_spy, qqq_trend_up, spy_20_up, rsi, vix)

    # 3. 儲存至資料庫
    db.log_today((today, vix, us10y, twii, gs, news_s, ts_s, is_bear, atr, lunar, bazi,
                  report, signal, conf, current_risk, current_focus,
                  weights.get('SPY',0), weights.get('TLT',0), weights.get('GOLD',0),
                  weights.get('SILVER',0), weights.get('OIL',0), weights.get('VIX',0),
                  weights.get('QQQ',0)))

    # 4. Telegram 訊息推播 - 🌟 版面大升級 (仿照截圖格式)
    
    # 狀態判定
    status_text = '🚨 VIX極端恐慌' if not is_missing(vix) and vix >= 35 else \
                  '🐂 強牛市攻擊' if spy_trend and spy_20_up else \
                  '⚠️ 弱牛盤整防禦' if spy_trend else \
                  '🔥 超賣搶反彈' if not is_missing(rsi) and rsi < 30 else \
                  '🐻 傳統熊市避險' if tlt_trend else '☠️ 股債雙殺 (避開美債)'
                  
    # --- 建立市場風險指標文字 ---
    # FinBERT
    val_finbert = f"{news_s:.2f}" if not is_missing(news_s) else "nan"
    icon_finbert = "⚠️ 悲觀" if (not is_missing(news_s) and news_s < -0.2) else "✅ 正常" if not is_missing(news_s) else "⚪ 未知"
    
    # 日圓匯率
    val_jpy = f"{jpy_current:.2f}" if not is_missing(jpy_current) else "nan"
    val_jpy_chg = f"{jpy_change:+.2f}%" if not is_missing(jpy_change) else "nan"
    icon_jpy = "⚠️ 留意套利平倉風險" if (not is_missing(jpy_change) and jpy_change > 0) else "✅ 穩定" if not is_missing(jpy_change) else "⚪ 未知"
    
    # VIX 恐慌指數
    val_vix = f"{vix:.2f}" if not is_missing(vix) else "nan"
    icon_vix = "⚠️ 高波動" if (not is_missing(vix) and vix >= 35) else "✅ 正常" if not is_missing(vix) else "⚪ 未知"

    # --- 檢查缺失資料 (nan) ---
    check_dict = {
        'vix': vix, 'us10y_yield': us10y, 'gold_silver_ratio': gs,
        'jpy_current': jpy_current, 'finbert_sentiment': news_s
    }
    missing_keys = [k for k, v in check_dict.items() if is_missing(v)]
    missing_str = ""
    if missing_keys:
        missing_str = f"\n⚠️ 以下資料缺失 (nan)，請補齊後再評估：\n  - {' / '.join(missing_keys)}\n"

    # --- 總結建議 ---
    risk_advice = "🔴 高風險：建議大幅降低倉位，轉向現金或債券" if (not is_missing(vix) and vix >= 35) or (not spy_trend) else "🟢 正常：依循動能與策略配置"

    # --- 組裝最終 Telegram 訊息 ---
    telegram_message = f"""
=========================================
= 📅 【龍蝦總經理晨間匯報】 {today}
=========================================
=
📊 市場風險指標摘要
=========================================
• FinBERT 情緒指標：{val_finbert} (警示線 -0.2) -> {icon_finbert}
• 日圓匯率：{val_jpy}，月變動 {val_jpy_chg} -> {icon_jpy}
• VIX 恐慌指數：{val_vix} (警示線 35) -> {icon_vix}
{missing_str}
📌 建議：{risk_advice}

=========================================
= 🎯 今日動態資產配置 (級別 {current_risk} | {current_focus})
=========================================
🛡️ 系統狀態：{status_text} (RSI: {rsi:.1f} / 信心度: {conf*100:.0f}%)
🚀 科技輪動：{'啟動 (QQQ 強於大盤)' if qqq_beats_spy and qqq_trend_up and spy_trend and spy_20_up else '未觸發'}

📈 股市 (SPY): {weights.get('SPY', 0):.1%}
🦅 科技 (QQQ): {weights.get('QQQ', 0):.1%}
🏦 美債 (TLT): {weights.get('TLT', 0):.1%}
🥇 黃金 (GOLD): {weights.get('GOLD', 0):.1%}
🥈 白銀 (SILVER): {weights.get('SILVER', 0):.1%}
🛢️ 原油 (OIL): {weights.get('OIL', 0):.1%}
🚨 恐慌 (VIX): {weights.get('VIX', 0):.1%}

=========================================
= 🤖 AI 深度分析報告
=========================================
{report}
    """

    print("\n[預覽報告內容]：" + telegram_message)

    # 🌟 推播引擎
    if MAKE_WEBHOOK_URL:
        try:
            req = requests.post(MAKE_WEBHOOK_URL, json={"report": telegram_message})
            if req.status_code == 200:
                print("\n✅ 偵測完成，晨間報告已成功推播至 Telegram！")
            else:
                print(f"\n⚠️ 發送失敗，Make 狀態碼：{req.status_code}")
        except Exception as e:
            print(f"\n❌ 傳送錯誤：{e}")
    else:
        print("\n⚠️ 未設定 MAKE_WEBHOOK_URL，僅在畫面上顯示。")

if __name__ == "__main__":
    run_system()
