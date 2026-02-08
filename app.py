from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import os
import time
import telegram
import asyncio
from threading import Thread
import pytz
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==============================================
# CONFIGURACIÓN SISTEMA PAXG/BTC SPOT
# ==============================================

# Configuración Telegram para PAXG/BTC
TELEGRAM_BOT_TOKEN = "8551183895:AAEDv30UfXSWtaKcsymSUBINjU-jE_MFqek"
TELEGRAM_CHAT_ID = "-5079948404"

# Solo PAXG/BTC para estrategias SPOT
TOP_CRYPTO_SYMBOLS = ["PAXG-BTC"]

# Temporalidades para análisis SPOT (4h, 12h, 1D, 1W)
TELEGRAM_ALERT_TIMEFRAMES = ['4h', '12h', '1D', '1W']

# Mapeo de temporalidades para análisis multi-timeframe
TIMEFRAME_HIERARCHY = {
    '4h': {'mayor': '8h', 'media': '4h', 'menor': '2h'},
    '12h': {'mayor': '1D', 'media': '12h', 'menor': '4h'},
    '1D': {'mayor': '1W', 'media': '1D', 'menor': '12h'},
    '1W': {'mayor': '1M', 'media': '1W', 'menor': '3D'}
}

# Configuración de estrategias por temporalidad (SPOT TRADING)
STRATEGY_TIMEFRAMES = {
    'Ichimoku Cloud Breakout': ['4h', '12h', '1D'],
    'Fibonacci Supertrend': ['4h', '12h', '1D', '1W'],
    'Cryptodivisa': ['4h', '12h', '1D', '1W'],
    'Reversion Soportes Resistencias': ['4h', '12h', '1D'],
    'Stochastic Fibonacci': ['4h', '12h', '1D'],
    'VWAP Reversal': ['4h', '12h', '1D'],
    'Momentum Divergence': ['4h', '12h', '1D', '1W'],
    'ADX Power Trend': ['4h', '12h', '1D'],
    'MA Convergence Divergence': ['4h', '12h', '1D'],
    'Volume Spike Momentum': ['4h', '12h', '1D'],
    'Stochastic Supertrend': ['4h', '12h', '1D'],
    'Support Resistance Bounce': ['4h', '12h', '1D'],
    'Whale DMI Combo': ['12h', '1D'],
    'Parabolic SAR Momentum': ['4h', '12h', '1D'],
    'Multi-Timeframe Confirmation': ['4h', '12h', '1D'],
    # NUEVAS ESTRATEGIAS
    'RSI Maverick DMI Divergence': ['4h', '12h', '1D'],
    'RSI Maverick Stochastic Crossover': ['4h', '12h', '1D'],
    'RSI Maverick MACD Divergence': ['4h', '12h', '1D'],
    'Whale RSI Maverick Combo': ['12h', '1D'],
    'RSI Maverick Trend Reversal': ['4h', '12h', '1D']
}

class TradingIndicator:
    def __init__(self):
        self.cache = {}
        self.alert_cache = {}
        self.active_operations = {}
        self.winrate_data = {}
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.sent_exit_signals = set()
        self.volume_ema_signals = {}
        self.strategy_signals = {}
        self.support_resistance_alerts = {}
        self.rsi_maverick_divergences = {}  # Cache para divergencias RSI Maverick
        
    def get_bolivia_time(self):
        """Obtener hora actual de Bolivia"""
        return datetime.now(self.bolivia_tz)
    
    def is_trading_time(self):
        """Verificar si es horario de trading - 24/7 para crypto"""
        return True

    def is_operational_timeframe(self, interval):
        """Verificar si la temporalidad está en las permitidas para alertas"""
        return interval in TELEGRAM_ALERT_TIMEFRAMES

    def calculate_remaining_time(self, interval, current_time):
        """Calcular tiempo restante para el cierre de la vela"""
        interval_seconds = {
            '4h': 14400, '12h': 43200, '1D': 86400, '1W': 604800
        }
        
        seconds = interval_seconds.get(interval, 3600)
        
        # PORCENTAJES DE FORMACIÓN DE VELA
        if interval == '4h':
            percent = 85
        elif interval == '12h':
            percent = 90
        elif interval == '1D':
            percent = 95
        elif interval == '1W':
            percent = 99
        else:
            percent = 50
        
        seconds_passed = current_time.timestamp() % seconds
        return seconds_passed >= (seconds * percent / 100)

    def get_kucoin_data(self, symbol, interval, limit=100):
        """Obtener datos de KuCoin para PAXG/BTC"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < 60:
                    return cached_data
            
            interval_map = {
                '15m': '15min', '30m': '30min', '1h': '1hour', '2h': '2hour',
                '4h': '4hour', '8h': '8hour', '12h': '12hour',
                '1D': '1day', '1W': '1week', '1M': '1month'
            }
            
            kucoin_interval = interval_map.get(interval, '1hour')
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type={kucoin_interval}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '200000' and data.get('data'):
                    candles = data['data']
                    if not candles:
                        return self.generate_sample_data(limit, interval, symbol)
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna()
                    result = df.tail(limit)
                    self.cache[cache_key] = (result, datetime.now())
                    return result
                
        except Exception as e:
            print(f"Error obteniendo datos para {symbol} {interval}: {e}")
        
        return self.generate_sample_data(limit, interval, symbol)

    def generate_sample_data(self, limit, interval, symbol):
        """Generar datos de ejemplo realistas para PAXG/BTC"""
        np.random.seed(42)
        base_price = 0.05  # PAXG/BTC alrededor de 0.05 BTC
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
        
        returns = np.random.normal(0.0005, 0.008, limit)  # Menor volatilidad
        prices = base_price * (1 + np.cumsum(returns))
        
        data = {
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.003, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0.005, 0.005, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0.005, 0.005, limit))),
            'close': prices,
            'volume': np.random.lognormal(8, 0.8, limit)  # Menor volumen
        }
        
        df = pd.DataFrame(data)
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df

    # ==============================================
    # INDICADORES TÉCNICOS PARA SPOT TRADING
    # ==============================================

    def calculate_sma(self, prices, period):
        """Calcular SMA"""
        if len(prices) < period:
            return np.zeros_like(prices)
        
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            if i >= period - 1:
                sma[i] = np.mean(prices[i-period+1:i+1])
            else:
                sma[i] = prices[i] if i < len(prices) else 0
        
        return sma

    def calculate_ema(self, prices, period):
        """Calcular EMA"""
        if len(prices) == 0 or period <= 0:
            return np.zeros_like(prices)
            
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema

    def calculate_atr(self, high, low, close, period=14):
        """Calcular Average True Range"""
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        atr = self.calculate_ema(tr, period)
        return atr

    def calculate_rsi(self, prices, period=14):
        """Calcular RSI tradicional"""
        if len(prices) < period + 1:
            return np.zeros_like(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(prices))
        avg_losses = np.zeros(len(prices))
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.zeros(len(prices))
        for i in range(len(prices)):
            if avg_losses[i] > 0:
                rs[i] = avg_gains[i] / avg_losses[i]
            else:
                rs[i] = 100 if avg_gains[i] > 0 else 50
        
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcular MACD"""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices, period=20, multiplier=2):
        """Calcular Bandas de Bollinger"""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
        
        sma = self.calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(len(prices)):
            if i >= period - 1:
                window = prices[i-period+1:i+1]
                std[i] = np.std(window)
        
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        
        return upper, sma, lower

    def calculate_stochastic_rsi(self, close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """Calcular RSI Estocástico"""
        try:
            n = len(close)
            
            rsi = self.calculate_rsi(close, rsi_period)
            
            stoch_rsi = np.zeros(n)
            k_line = np.zeros(n)
            d_line = np.zeros(n)
            
            for i in range(stoch_period - 1, n):
                start_idx = i - stoch_period + 1
                if start_idx < 0:
                    start_idx = 0
                
                rsi_window = rsi[start_idx:i+1]
                if len(rsi_window) > 0:
                    rsi_low = np.min(rsi_window)
                    rsi_high = np.max(rsi_window)
                    
                    if (rsi_high - rsi_low) > 0:
                        stoch_rsi[i] = 100 * (rsi[i] - rsi_low) / (rsi_high - rsi_low)
                    else:
                        stoch_rsi[i] = 50
            
            for i in range(k_period - 1, n):
                start_idx = i - k_period + 1
                if start_idx < 0:
                    start_idx = 0
                k_line[i] = np.mean(stoch_rsi[start_idx:i+1])
            
            for i in range(k_period + d_period - 2, n):
                start_idx = i - d_period + 1
                if start_idx < 0:
                    start_idx = 0
                d_line[i] = np.mean(k_line[start_idx:i+1])
            
            return {
                'stoch_rsi': stoch_rsi.tolist(),
                'k_line': k_line.tolist(),
                'd_line': d_line.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_stochastic_rsi: {e}")
            n = len(close)
            return {
                'stoch_rsi': [50] * n,
                'k_line': [50] * n,
                'd_line': [50] * n
            }

    def calculate_adx(self, high, low, close, period=14):
        """Calcular ADX, +DI, -DI"""
        n = len(high)
        if n < period:
            return np.zeros(n), np.zeros(n), np.zeros(n)
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        tr_smooth = self.calculate_ema(tr, period)
        plus_dm_smooth = self.calculate_ema(plus_dm, period)
        minus_dm_smooth = self.calculate_ema(minus_dm, period)
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        for i in range(n):
            if tr_smooth[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        dx = np.zeros(n)
        for i in range(n):
            if (plus_di[i] + minus_di[i]) > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = self.calculate_ema(dx, period)
        
        return adx, plus_di, minus_di

    def calculate_vwap(self, high, low, close, volume):
        """Calcular VWAP (Volume Weighted Average Price)"""
        n = len(close)
        vwap = np.zeros(n)
        cumulative_volume = 0
        cumulative_price_volume = 0
        
        for i in range(n):
            typical_price = (high[i] + low[i] + close[i]) / 3
            cumulative_price_volume += typical_price * volume[i]
            cumulative_volume += volume[i]
            
            if cumulative_volume > 0:
                vwap[i] = cumulative_price_volume / cumulative_volume
            else:
                vwap[i] = typical_price
        
        return vwap

    def calculate_ichimoku(self, high, low, close, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
        """Calcular Ichimoku Cloud"""
        n = len(close)
        
        tenkan_sen = np.zeros(n)
        kijun_sen = np.zeros(n)
        senkou_span_a = np.zeros(n)
        senkou_span_b = np.zeros(n)
        chikou_span = np.zeros(n)
        
        for i in range(n):
            if i >= tenkan_period - 1:
                tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
            
            if i >= kijun_period - 1:
                kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
            
            if i >= kijun_period - 1:
                senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
            
            if i >= senkou_span_b_period - 1:
                senkou_span_b[i] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
        
        cloud_top = np.maximum(senkou_span_a, senkou_span_b)
        cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'chikou_span': chikou_span
        }

    def calculate_supertrend(self, high, low, close, period=10, multiplier=3):
        """Calcular SuperTrend"""
        n = len(close)
        
        hl2 = (high + low) / 2
        atr = self.calculate_atr(high, low, close, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = np.zeros(n)
        direction = np.zeros(n)  # 1 = alcista, -1 = bajista
        
        supertrend[0] = upper_band[0]
        direction[0] = -1
        
        for i in range(1, n):
            if close[i-1] <= supertrend[i-1]:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            
            if close[i] > supertrend[i]:
                direction[i] = 1
            elif close[i] < supertrend[i]:
                direction[i] = -1
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band
        }

    def calculate_parabolic_sar(self, high, low, acceleration=0.02, maximum=0.2):
        """Calcular Parabolic SAR"""
        n = len(high)
        
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 = alcista, -1 = bajista
        ep = np.zeros(n)  # Extreme Point
        af = np.zeros(n)  # Acceleration Factor
        
        sar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        af[0] = acceleration
        
        for i in range(1, n):
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if trend[i-1] == 1:
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = acceleration
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            else:
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = acceleration
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
        
        return {
            'sar': sar,
            'trend': trend,
            'ep': ep,
            'af': af
        }

    def calculate_fibonacci_levels(self, high, low):
        """Calcular niveles de Fibonacci"""
        swing_high = np.max(high[-100:])
        swing_low = np.min(low[-100:])
        diff = swing_high - swing_low
        
        return {
            '0.0': swing_low,
            '0.236': swing_low + diff * 0.236,
            '0.382': swing_low + diff * 0.382,
            '0.5': swing_low + diff * 0.5,
            '0.618': swing_low + diff * 0.618,
            '0.786': swing_low + diff * 0.786,
            '1.0': swing_high,
            '1.272': swing_high + diff * 0.272,
            '1.618': swing_high + diff * 0.618
        }

    def calculate_trend_strength_maverick(self, close, length=20, mult=2.0):
        """Calcular Fuerza de Tendencia Maverick"""
        try:
            n = len(close)
            
            basis = self.calculate_sma(close, length)
            dev = np.zeros(n)
            
            for i in range(length-1, n):
                window = close[i-length+1:i+1]
                dev[i] = np.std(window) if len(window) > 1 else 0
            
            upper = basis + (dev * mult)
            lower = basis - (dev * mult)
            
            bb_width = np.zeros(n)
            for i in range(n):
                if basis[i] > 0:
                    bb_width[i] = ((upper[i] - lower[i]) / basis[i]) * 100
            
            trend_strength = np.zeros(n)
            for i in range(1, n):
                if bb_width[i] > bb_width[i-1]:
                    trend_strength[i] = bb_width[i]
                else:
                    trend_strength[i] = -bb_width[i]
            
            if n >= 50:
                historical_bb_width = bb_width[max(0, n-100):n]
                high_zone_threshold = np.percentile(historical_bb_width, 70)
            else:
                high_zone_threshold = np.percentile(bb_width, 70) if len(bb_width) > 0 else 5
            
            no_trade_zones = np.zeros(n, dtype=bool)
            strength_signals = ['NEUTRAL'] * n
            
            for i in range(10, n):
                if (bb_width[i] > high_zone_threshold and 
                    trend_strength[i] < 0 and 
                    bb_width[i] < np.max(bb_width[max(0, i-10):i])):
                    no_trade_zones[i] = True
                
                if trend_strength[i] > 0:
                    if bb_width[i] > high_zone_threshold:
                        strength_signals[i] = 'STRONG_UP'
                    else:
                        strength_signals[i] = 'WEAK_UP'
                elif trend_strength[i] < 0:
                    if bb_width[i] > high_zone_threshold:
                        strength_signals[i] = 'STRONG_DOWN'
                    else:
                        strength_signals[i] = 'WEAK_DOWN'
                else:
                    strength_signals[i] = 'NEUTRAL'
            
            return {
                'bb_width': bb_width.tolist(),
                'trend_strength': trend_strength.tolist(),
                'basis': basis.tolist(),
                'upper_band': upper.tolist(),
                'lower_band': lower.tolist(),
                'high_zone_threshold': float(high_zone_threshold),
                'no_trade_zones': no_trade_zones.tolist(),
                'strength_signals': strength_signals,
                'colors': ['green' if x > 0 else 'red' for x in trend_strength]
            }
            
        except Exception as e:
            print(f"Error en calculate_trend_strength_maverick: {e}")
            n = len(close)
            return {
                'bb_width': [0] * n,
                'trend_strength': [0] * n,
                'basis': [0] * n,
                'upper_band': [0] * n,
                'lower_band': [0] * n,
                'high_zone_threshold': 5.0,
                'no_trade_zones': [False] * n,
                'strength_signals': ['NEUTRAL'] * n,
                'colors': ['gray'] * n
            }

    def calculate_whale_signals_improved(self, df, sensitivity=1.7, min_volume_multiplier=1.5):
        """Indicador de ballenas mejorado"""
        try:
            close = df['close'].values
            low = df['low'].values
            high = df['high'].values
            volume = df['volume'].values
            
            n = len(close)
            
            whale_pump_signal = np.zeros(n)
            whale_dump_signal = np.zeros(n)
            confirmed_buy = np.zeros(n, dtype=bool)
            confirmed_sell = np.zeros(n, dtype=bool)
            extended_buy = np.zeros(n, dtype=bool)
            extended_sell = np.zeros(n, dtype=bool)
            
            for i in range(5, n-1):
                avg_volume = np.mean(volume[max(0, i-20):i+1])
                volume_ratio = volume[i] / avg_volume if avg_volume > 0 else 1
                
                price_change = (close[i] - close[i-1]) / close[i-1] * 100
                low_5 = np.min(low[max(0, i-5):i+1])
                high_5 = np.max(high[max(0, i-5):i+1])
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] < close[i-1] or price_change < -0.5) and
                    low[i] <= low_5 * 1.01):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_pump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                    
                    for j in range(i, min(n, i + 7)):
                        whale_pump_signal[j] = max(whale_pump_signal[j], whale_pump_signal[i] * 0.8)
                
                if (volume_ratio > min_volume_multiplier and 
                    (close[i] > close[i-1] or price_change > 0.5) and
                    high[i] >= high_5 * 0.99):
                    
                    volume_strength = min(3.0, volume_ratio / min_volume_multiplier)
                    whale_dump_signal[i] = min(100, volume_ratio * 20 * sensitivity * volume_strength)
                    
                    for j in range(i, min(n, i + 7)):
                        whale_dump_signal[j] = max(whale_dump_signal[j], whale_dump_signal[i] * 0.8)
            
            whale_pump_smooth = self.calculate_sma(whale_pump_signal, 3)
            whale_dump_smooth = self.calculate_sma(whale_dump_signal, 3)
            
            return {
                'whale_pump': whale_pump_smooth.tolist(),
                'whale_dump': whale_dump_smooth.tolist(),
                'confirmed_buy': confirmed_buy.tolist(),
                'confirmed_sell': confirmed_sell.tolist(),
                'extended_buy': extended_buy.tolist(),
                'extended_sell': extended_sell.tolist()
            }
            
        except Exception as e:
            print(f"Error en calculate_whale_signals_improved: {e}")
            n = len(df)
            return {
                'whale_pump': [0] * n,
                'whale_dump': [0] * n,
                'confirmed_buy': [False] * n,
                'confirmed_sell': [False] * n,
                'extended_buy': [False] * n,
                'extended_sell': [False] * n
            }

    def calculate_volume_anomaly(self, volume, close, period=20, std_multiplier=2):
        """Calcular anomalías de volumen"""
        try:
            n = len(volume)
            volume_anomaly = np.zeros(n, dtype=bool)
            volume_clusters = np.zeros(n, dtype=bool)
            volume_ratio = np.zeros(n)
            volume_signal = ['NEUTRAL'] * n
            
            for i in range(period, n):
                volume_ma = self.calculate_sma(volume[:i+1], period)
                current_volume_ma = volume_ma[i] if i < len(volume_ma) else volume[i]
                
                if current_volume_ma > 0:
                    volume_ratio[i] = volume[i] / current_volume_ma
                else:
                    volume_ratio[i] = 1
                
                if i >= period * 2:
                    window = volume[max(0, i-period*2):i+1]
                    std_volume = np.std(window) if len(window) > 1 else 0
                    
                    if volume_ratio[i] > 1 + (std_multiplier * (std_volume / current_volume_ma if current_volume_ma > 0 else 0)):
                        volume_anomaly[i] = True
                        
                        if i > 0:
                            price_change = (close[i] - close[i-1]) / close[i-1] * 100
                            if price_change > 0:
                                volume_signal[i] = 'COMPRA'
                            else:
                                volume_signal[i] = 'VENTA'
                
                if i >= 5:
                    recent_anomalies = volume_anomaly[max(0, i-4):i+1]
                    if np.sum(recent_anomalies) >= 2:
                        volume_clusters[i] = True
            
            return {
                'volume_anomaly': volume_anomaly.tolist(),
                'volume_clusters': volume_clusters.tolist(),
                'volume_ratio': volume_ratio.tolist(),
                'volume_ma': volume_ma.tolist() if 'volume_ma' in locals() else [0] * n,
                'volume_signal': volume_signal
            }
            
        except Exception as e:
            print(f"Error en calculate_volume_anomaly: {e}")
            n = len(volume)
            return {
                'volume_anomaly': [False] * n,
                'volume_clusters': [False] * n,
                'volume_ratio': [1] * n,
                'volume_ma': [0] * n,
                'volume_signal': ['NEUTRAL'] * n
            }

    def calculate_support_resistance_channels(self, high, low, close, pivot_period=10, channel_width_pct=0.05, min_strength=1):
        """Calcular canales de soporte/resistencia"""
        n = len(high)
        
        pivot_highs = np.zeros(n)
        pivot_lows = np.zeros(n)
        
        for i in range(pivot_period, n - pivot_period):
            window_high = high[i-pivot_period:i+pivot_period+1]
            window_low = low[i-pivot_period:i+pivot_period+1]
            
            if high[i] == np.max(window_high):
                pivot_highs[i] = high[i]
            
            if low[i] == np.min(window_low):
                pivot_lows[i] = low[i]
        
        pivot_points = []
        pivot_indices = []
        
        for i in range(n):
            if pivot_highs[i] > 0:
                pivot_points.append(pivot_highs[i])
                pivot_indices.append(i)
            if pivot_lows[i] > 0:
                pivot_points.append(pivot_lows[i])
                pivot_indices.append(i)
        
        if len(pivot_points) < 2:
            return [], []
        
        pivot_points = np.array(pivot_points)
        pivot_indices = np.array(pivot_indices)
        
        prd_highest = np.max(high[-300:]) if len(high) >= 300 else np.max(high)
        prd_lowest = np.min(low[-300:]) if len(low) >= 300 else np.min(low)
        cwidth = (prd_highest - prd_lowest) * channel_width_pct / 100
        
        channels = []
        
        for i in range(len(pivot_points)):
            level = pivot_points[i]
            hi = level
            lo = level
            strength = 0
            
            for j in range(len(pivot_points)):
                other_level = pivot_points[j]
                width = abs(other_level - level)
                
                if width <= cwidth:
                    hi = max(hi, other_level)
                    lo = min(lo, other_level)
                    strength += 1
            
            if strength >= min_strength:
                channels.append({
                    'high': hi,
                    'low': lo,
                    'strength': strength,
                    'mid': (hi + lo) / 2
                })
        
        channels = sorted(channels, key=lambda x: x['strength'], reverse=True)
        
        supports = []
        resistances = []
        
        for channel in channels[:6]:
            if close[-1] > channel['mid']:
                supports.append(channel['low'])
            else:
                resistances.append(channel['high'])
        
        supports = [round(float(s), 6) for s in supports if s > 0]
        resistances = [round(float(r), 6) for r in resistances if r > 0]
        
        return supports, resistances

    def detect_divergence(self, price, indicator, lookback=14):
        """Detectar divergencias entre precio e indicador"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            indicator_window = indicator[i-lookback:i+1]
            
            if (price[i] < np.min(price_window[:-1]) and 
                indicator[i] > np.min(indicator_window[:-1])):
                bullish_div[i] = True
            
            if (price[i] > np.max(price_window[:-1]) and 
                indicator[i] < np.max(indicator_window[:-1])):
                bearish_div[i] = True
        
        extended_bullish = bullish_div.copy()
        extended_bearish = bearish_div.copy()
        
        for i in range(n):
            if bullish_div[i]:
                for j in range(1, min(7, n-i)):
                    extended_bullish[i+j] = True
            if bearish_div[i]:
                for j in range(1, min(7, n-i)):
                    extended_bearish[i+j] = True
        
        return extended_bullish.tolist(), extended_bearish.tolist()

    def check_di_crossover(self, plus_di, minus_di):
        """Verificar cruces de DMI"""
        n = len(plus_di)
        di_cross_bullish = np.zeros(n, dtype=bool)
        di_cross_bearish = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                di_cross_bullish[i] = True
            
            if minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]:
                di_cross_bearish[i] = True
        
        return di_cross_bullish.tolist(), di_cross_bearish.tolist()

    def detect_candlestick_patterns(self, open_prices, high, low, close):
        """Detectar patrones de velas japonesas"""
        n = len(close)
        patterns = []
        
        for i in range(2, n):
            o, h, l, c = open_prices[i], high[i], low[i], close[i]
            o1, h1, l1, c1 = open_prices[i-1], high[i-1], low[i-1], close[i-1]
            o2, h2, l2, c2 = open_prices[i-2], high[i-2], low[i-2], close[i-2]
            
            # Hammer
            if (l < min(o, c) - (h - max(o, c)) * 2 and 
                (h - max(o, c)) < (max(o, c) - l) / 3):
                patterns.append({
                    'index': i,
                    'pattern': 'HAMMER',
                    'direction': 'BULLISH'
                })
            
            # Shooting Star
            elif (h > max(o, c) + (min(o, c) - l) * 2 and 
                  (min(o, c) - l) < (h - max(o, c)) / 3):
                patterns.append({
                    'index': i,
                    'pattern': 'SHOOTING_STAR',
                    'direction': 'BEARISH'
                })
            
            # Bullish Engulfing
            elif (c1 < o1 and c > o and c > o1 and o < c1):
                patterns.append({
                    'index': i,
                    'pattern': 'BULLISH_ENGULFING',
                    'direction': 'BULLISH'
                })
            
            # Bearish Engulfing
            elif (c1 > o1 and c < o and c < o1 and o > c1):
                patterns.append({
                    'index': i,
                    'pattern': 'BEARISH_ENGULFING',
                    'direction': 'BEARISH'
                })
        
        return patterns

    def check_multi_timeframe_obligatory(self, symbol, interval, signal_type):
        """Verificar confirmación multi-temporalidad"""
        try:
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return True
            
            menor_interval = hierarchy.get('menor')
            mayor_interval = hierarchy.get('mayor')
            
            # Verificar temporalidad menor
            if menor_interval:
                df_menor = self.get_kucoin_data(symbol, menor_interval, 50)
                if df_menor is not None and len(df_menor) > 20:
                    menor_close = df_menor['close'].values
                    menor_ma20 = self.calculate_sma(menor_close, 20)
                    
                    if signal_type == 'COMPRA' and menor_close[-1] < menor_ma20[-1]:
                        return False
                    elif signal_type == 'VENTA' and menor_close[-1] > menor_ma20[-1]:
                        return False
            
            # Verificar temporalidad mayor
            if mayor_interval:
                df_mayor = self.get_kucoin_data(symbol, mayor_interval, 50)
                if df_mayor is not None and len(df_mayor) > 20:
                    mayor_close = df_mayor['close'].values
                    mayor_ma50 = self.calculate_sma(mayor_close, 50)
                    
                    if signal_type == 'COMPRA' and mayor_close[-1] < mayor_ma50[-1]:
                        return False
                    elif signal_type == 'VENTA' and mayor_close[-1] > mayor_ma50[-1]:
                        return False
            
            return True
            
        except Exception as e:
            print(f"Error en check_multi_timeframe_obligatory: {e}")
            return True

    def should_send_telegram_alert(self, interval):
        """Determinar si se debe enviar alerta por Telegram"""
        current_time = self.get_bolivia_time()
        return self.calculate_remaining_time(interval, current_time)

    def calculate_optimal_entry_exit(self, df, signal_type, leverage=1, support_levels=None, resistance_levels=None):
        """Calcular entradas y salidas óptimas para SPOT"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            current_price = close[-1]
            atr = self.calculate_atr(high, low, close)
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            if support_levels is None or resistance_levels is None:
                support_levels, resistance_levels = self.calculate_support_resistance_channels(high, low, close)
            
            if signal_type == 'COMPRA':
                valid_supports = [s for s in support_levels if s < current_price]
                if valid_supports:
                    entry = max(valid_supports)
                else:
                    entry = current_price * 0.995
                
                if len(support_levels) > 1:
                    stop_loss = support_levels[1] if len(support_levels) > 1 else entry - (current_atr * 2)
                else:
                    stop_loss = entry - (current_atr * 2)
                
                take_profits = []
                for resistance in resistance_levels[:3]:
                    if resistance > entry:
                        take_profits.append(resistance)
                
                if not take_profits:
                    take_profits = [entry + (2 * (entry - stop_loss))]
            
            else:  # VENTA
                valid_resistances = [r for r in resistance_levels if r > current_price]
                if valid_resistances:
                    entry = min(valid_resistances)
                else:
                    entry = current_price * 1.005
                
                if len(resistance_levels) > 1:
                    stop_loss = resistance_levels[1] if len(resistance_levels) > 1 else entry + (current_atr * 2)
                else:
                    stop_loss = entry + (current_atr * 2)
                
                take_profits = []
                for support in support_levels[:3]:
                    if support < entry:
                        take_profits.append(support)
                
                if not take_profits:
                    take_profits = [entry - (2 * (stop_loss - entry))]
            
            return {
                'entry': float(entry),
                'stop_loss': float(stop_loss),
                'take_profit': [float(tp) for tp in take_profits[:3]],
                'support_levels': [float(s) for s in support_levels],
                'resistance_levels': [float(r) for r in resistance_levels],
                'atr': float(current_atr),
                'atr_percentage': float(current_atr / current_price)
            }
            
        except Exception as e:
            print(f"Error calculando entradas/salidas óptimas: {e}")
            current_price = float(df['close'].iloc[-1])
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': [current_price * 1.02],
                'support_levels': [current_price * 0.95, current_price * 0.90],
                'resistance_levels': [current_price * 1.05, current_price * 1.10],
                'atr': 0.0,
                'atr_percentage': 0.0
            }

    # ==============================================
    # NUEVAS FUNCIONES PARA RSI MAVERICK
    # ==============================================
    
    def calculate_rsi_maverick(self, close, length=20, bb_multiplier=2.0):
        """Implementación del RSI Modificado Maverick"""
        try:
            n = len(close)
            
            # Calcular media móvil y desviación estándar
            basis = np.array([np.mean(close[max(0, i-length+1):i+1]) for i in range(n)])
            dev = np.array([np.std(close[max(0, i-length+1):i+1]) for i in range(n)])
            
            # Calcular bandas
            upper = basis + (dev * bb_multiplier)
            lower = basis - (dev * bb_multiplier)
            
            # Calcular posición porcentual dentro de las bandas
            b_percent = np.zeros(n)
            for i in range(n):
                if (upper[i] - lower[i]) > 0:
                    b_percent[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
                else:
                    b_percent[i] = 0.5
            
            return b_percent.tolist()
            
        except Exception as e:
            print(f"Error en calculate_rsi_maverick: {e}")
            return [0.5] * len(close)
    
    def detect_rsi_maverick_divergence(self, price, rsi_maverick, lookback=14):
        """Detectar divergencias específicas para RSI Maverick (7 velas extendidas)"""
        n = len(price)
        bullish_div = np.zeros(n, dtype=bool)
        bearish_div = np.zeros(n, dtype=bool)
        
        for i in range(lookback, n-1):
            price_window = price[i-lookback:i+1]
            rsi_window = rsi_maverick[i-lookback:i+1]
            
            # Divergencia alcista: precio hace mínimos más bajos pero RSI Maverick hace mínimos más altos
            if (price[i] < np.min(price_window[:-1]) and 
                rsi_maverick[i] > np.min(rsi_window[:-1])):
                bullish_div[i] = True
                # Extender señal por 7 velas
                for j in range(1, min(7, n-i)):
                    bullish_div[i+j] = True
            
            # Divergencia bajista: precio hace máximos más altos pero RSI Maverick hace máximos más bajos
            if (price[i] > np.max(price_window[:-1]) and 
                rsi_maverick[i] < np.max(rsi_window[:-1])):
                bearish_div[i] = True
                # Extender señal por 7 velas
                for j in range(1, min(7, n-i)):
                    bearish_div[i+j] = True
        
        return bullish_div.tolist(), bearish_div.tolist()
    
    def get_rsi_maverick_divergence_signals(self, symbol, interval):
        """Obtener señales de divergencia del RSI Maverick"""
        cache_key = f"{symbol}_{interval}_rsi_maverick_div"
        
        if cache_key in self.rsi_maverick_divergences:
            cached_data, timestamp = self.rsi_maverick_divergences[cache_key]
            if (datetime.now() - timestamp).seconds < 300:
                return cached_data
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return {'bullish': [], 'bearish': []}
            
            close = df['close'].values
            
            # Calcular RSI Maverick
            rsi_maverick = self.calculate_rsi_maverick(close)
            
            # Detectar divergencias extendidas por 7 velas
            bullish_div, bearish_div = self.detect_rsi_maverick_divergence(close, rsi_maverick)
            
            result = {
                'bullish': bullish_div,
                'bearish': bearish_div,
                'rsi_maverick': rsi_maverick
            }
            
            self.rsi_maverick_divergences[cache_key] = (result, datetime.now())
            return result
            
        except Exception as e:
            print(f"Error en get_rsi_maverick_divergence_signals para {symbol} {interval}: {e}")
            return {'bullish': [], 'bearish': [], 'rsi_maverick': []}

    # ==============================================
    # ESTRATEGIA 1: ICHIMOKU CLOUD BREAKOUT
    # ==============================================
    def check_ichimoku_cloud_breakout_signal(self, symbol, interval):
        """Estrategia 1: Ichimoku Cloud Breakout para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Ichimoku Cloud Breakout']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            ichimoku = self.calculate_ichimoku(high, low, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            multi_tf_ok_compra = self.check_multi_timeframe_obligatory(symbol, interval, 'COMPRA')
            multi_tf_ok_venta = self.check_multi_timeframe_obligatory(symbol, interval, 'VENTA')
            
            current_price = close[-1]
            cloud_top = ichimoku['cloud_top'][-1]
            cloud_bottom = ichimoku['cloud_bottom'][-1]
            tenkan = ichimoku['tenkan_sen'][-1]
            kijun = ichimoku['kijun_sen'][-1]
            
            signal_type = None
            
            if (current_price > cloud_top and
                current_price > tenkan and
                current_price > kijun and
                tenkan > kijun and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] and
                multi_tf_ok_compra):
                
                signal_type = 'COMPRA'
                
            elif (current_price < cloud_bottom and
                  current_price < tenkan and
                  current_price < kijun and
                  tenkan < kijun and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  multi_tf_ok_venta):
                
                signal_type = 'VENTA'
            else:
                return None
            
            patterns = self.detect_candlestick_patterns(df['open'].values, high, low, close)
            recent_patterns = [p for p in patterns if p['index'] == len(close)-1]
            
            pattern_confirmed = False
            confirm_pattern = None
            for pattern in recent_patterns:
                if signal_type == 'COMPRA' and pattern['direction'] == 'BULLISH':
                    pattern_confirmed = True
                    confirm_pattern = pattern
                    break
                elif signal_type == 'VENTA' and pattern['direction'] == 'BEARISH':
                    pattern_confirmed = True
                    confirm_pattern = pattern
                    break
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            entry = current_price
            if signal_type == 'COMPRA':
                stop_loss = entry * 0.98
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                stop_loss = entry * 1.02
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_ICHIMOKU_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_ichimoku_chart(symbol, interval, df, ichimoku, ftm_data, 
                                                       confirm_pattern, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'ICHIMOKU CLOUD BREAKOUT',
                'chart': chart_buffer,
                'filters': [
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} nube Ichimoku',
                    f'Tenkan-sen {" > " if signal_type == "COMPRA" else " < "} Kijun-sen',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado',
                    f'Patrón confirmatorio: {confirm_pattern["pattern"] if pattern_confirmed else "No detectado"}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_ichimoku_cloud_breakout_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 2: FIBONACCI SUPERTREND
    # ==============================================
    def check_fibonacci_supertrend_signal(self, symbol, interval):
        """Estrategia 2: Fibonacci Supertrend para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Fibonacci Supertrend']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            supertrend_data = self.calculate_supertrend(high, low, close)
            supertrend_direction = supertrend_data['direction'][-1]
            
            fib_levels = self.calculate_fibonacci_levels(high, low)
            current_price = close[-1]
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            if (supertrend_direction == 1 and
                any(abs(current_price - fib_levels[str(level)]) / current_price < 0.01 
                    for level in [0.382, 0.5, 0.618]) and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            elif (supertrend_direction == -1 and
                  any(abs(current_price - fib_levels[str(level)]) / current_price < 0.01 
                      for level in [0.382, 0.5, 0.618]) and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            
            if not signal_type:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            if signal_type == 'COMPRA':
                nearest_level = min(fib_levels.values(), 
                                   key=lambda x: abs(x - current_price) if x < current_price else float('inf'))
                entry = nearest_level if nearest_level < current_price else current_price * 0.995
                stop_loss = entry * 0.98
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                nearest_level = min(fib_levels.values(), 
                                   key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
                entry = nearest_level if nearest_level > current_price else current_price * 1.005
                stop_loss = entry * 1.02
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_FIBONACCI_SUPERTREND_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_fibonacci_supertrend_chart(symbol, interval, df, 
                                                                   supertrend_data, fib_levels, 
                                                                   ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'FIBONACCI SUPERTREND',
                'chart': chart_buffer,
                'filters': [
                    f'SuperTrend: {"Alcista" if signal_type == "COMPRA" else "Bajista"}',
                    f'Precio en nivel Fibonacci clave',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_fibonacci_supertrend_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 3: CRYPTODIVISA
    # ==============================================
    def check_cryptodivisa_signal(self, symbol, interval):
        """Estrategia 3: CryptoDivisa (3 MAs + FTMaverick + Multi-timeframe) para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Cryptodivisa']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 200)
            if df is None or len(df) < 200:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            ma_9 = self.calculate_sma(close, 9)
            ma_53 = self.calculate_sma(close, 53)
            ma_180 = self.calculate_sma(close, 180)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            multi_tf_ok_compra = self.check_multi_timeframe_obligatory(symbol, interval, 'COMPRA')
            multi_tf_ok_venta = self.check_multi_timeframe_obligatory(symbol, interval, 'VENTA')
            
            current_price = close[-1]
            signal_type = None
            
            if (current_price > ma_9[-1] and
                ma_9[-1] > ma_53[-1] and
                ma_53[-1] > ma_180[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] and
                multi_tf_ok_compra):
                
                signal_type = 'COMPRA'
                
            elif (current_price < ma_9[-1] and
                  ma_9[-1] < ma_53[-1] and
                  ma_53[-1] < ma_180[-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  multi_tf_ok_venta):
                
                signal_type = 'VENTA'
            else:
                return None
            
            lookback = 5
            if signal_type == 'COMPRA':
                ma_cross = False
                for i in range(1, lookback + 1):
                    if (ma_9[-i] > ma_53[-i] and 
                        ma_9[-(i+1)] <= ma_53[-(i+1)]):
                        ma_cross = True
                        break
                if not ma_cross:
                    return None
                    
            elif signal_type == 'VENTA':
                ma_cross = False
                for i in range(1, lookback + 1):
                    if (ma_9[-i] < ma_53[-i] and 
                        ma_9[-(i+1)] >= ma_53[-(i+1)]):
                        ma_cross = True
                        break
                if not ma_cross:
                    return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            entry = current_price
            if signal_type == 'COMPRA':
                stop_loss = min(ma_53[-1], entry * 0.98)
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                stop_loss = max(ma_53[-1], entry * 1.02)
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_CRYPTODIVISA_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_cryptodivisa_chart(symbol, interval, df, 
                                                          ma_9, ma_53, ma_180,
                                                          ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'CRYPTODIVISA',
                'chart': chart_buffer,
                'filters': [
                    f'Alineación de MAs: {"Alcista" if signal_type == "COMPRA" else "Bajista"}',
                    f'MA9: {ma_9[-1]:.6f}, MA53: {ma_53[-1]:.6f}, MA180: {ma_180[-1]:.6f}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado',
                    f'Cruce reciente de MA9 {" > " if signal_type == "COMPRA" else " < "} MA53'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_cryptodivisa_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 4: REVERSIÓN SOPORTES RESISTENCIAS
    # ==============================================
    def check_reversion_soportes_resistencias_signal(self, symbol, interval):
        """Estrategia 4: Reversión en Soportes y Resistencias para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Reversion Soportes Resistencias']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_prices = df['open'].values
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            current_price = close[-1]
            current_open = open_prices[-1]
            current_high = high[-1]
            current_low = low[-1]
            
            signal_type = None
            target_level = None
            is_resistance = False
            
            for resistance in resistances[:3]:
                if resistance > 0:
                    if abs(current_high - resistance) / resistance < 0.005:
                        if close[-1] < open_prices[-1]:
                            signal_type = 'VENTA'
                            target_level = resistance
                            is_resistance = True
                            break
            
            if not signal_type:
                for support in supports[:3]:
                    if support > 0:
                        if abs(current_low - support) / support < 0.005:
                            if close[-1] > open_prices[-1]:
                                signal_type = 'COMPRA'
                                target_level = support
                                is_resistance = False
                                break
            
            if not signal_type:
                return None
            
            if signal_type == 'COMPRA' and ftm_data['strength_signals'][-1] not in ['STRONG_UP', 'WEAK_UP']:
                return None
            if signal_type == 'VENTA' and ftm_data['strength_signals'][-1] not in ['STRONG_DOWN', 'WEAK_DOWN']:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            patterns = self.detect_candlestick_patterns(open_prices, high, low, close)
            recent_patterns = [p for p in patterns if p['index'] == len(close)-1]
            
            pattern_confirmed = False
            confirm_pattern = None
            for pattern in recent_patterns:
                if signal_type == 'COMPRA' and pattern['direction'] == 'BULLISH':
                    pattern_confirmed = True
                    confirm_pattern = pattern
                    break
                elif signal_type == 'VENTA' and pattern['direction'] == 'BEARISH':
                    pattern_confirmed = True
                    confirm_pattern = pattern
                    break
            
            if not pattern_confirmed:
                return None
            
            if signal_type == 'COMPRA':
                entry = current_price
                stop_loss = target_level * 0.99
                take_profit = [
                    entry * 1.015,
                    target_level * 1.03 if is_resistance else entry * 1.03,
                    target_level * 1.05 if is_resistance else entry * 1.05
                ]
            else:
                entry = current_price
                stop_loss = target_level * 1.01
                take_profit = [
                    entry * 0.985,
                    target_level * 0.97 if not is_resistance else entry * 0.97,
                    target_level * 0.95 if not is_resistance else entry * 0.95
                ]
            
            signal_key = f"{symbol}_{interval}_REVERSION_{target_level:.6f}_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_support_resistance_chart(symbol, interval, df, 
                                                                supports, resistances,
                                                                target_level, signal_type,
                                                                ftm_data)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'REVERSIÓN SOPORTES/RESISTENCIAS',
                'chart': chart_buffer,
                'filters': [
                    f'Precio en {"resistencia" if is_resistance else "soporte"}: {target_level:.6f}',
                    f'Patrón de reversión: {confirm_pattern["pattern"]}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_reversion_soportes_resistencias_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 5: STOCHASTIC FIBONACCI
    # ==============================================
    def check_stochastic_fibonacci_signal(self, symbol, interval):
        """Estrategia 5: Stochastic Fibonacci para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Stochastic Fibonacci']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            stochastic = self.calculate_stochastic_rsi(close)
            k_line = stochastic['k_line']
            d_line = stochastic['d_line']
            
            fib_levels = self.calculate_fibonacci_levels(high, low)
            current_price = close[-1]
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            if (k_line[-1] < 20 and d_line[-1] < 20 and
                any(abs(current_price - fib_levels[str(level)]) / current_price < 0.01 
                    for level in [0.382, 0.5, 0.618]) and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            elif (k_line[-1] > 80 and d_line[-1] > 80 and
                  any(abs(current_price - fib_levels[str(level)]) / current_price < 0.01 
                      for level in [0.382, 0.5, 0.618]) and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            
            if not signal_type:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            if signal_type == 'COMPRA':
                nearest_level = min(fib_levels.values(), 
                                   key=lambda x: abs(x - current_price) if x < current_price else float('inf'))
                entry = nearest_level if nearest_level < current_price else current_price * 0.995
                stop_loss = entry * 0.98
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                nearest_level = min(fib_levels.values(), 
                                   key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
                entry = nearest_level if nearest_level > current_price else current_price * 1.005
                stop_loss = entry * 1.02
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_STOCHASTIC_FIB_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_stochastic_chart(symbol, interval, df, stochastic, 
                                                         ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'STOCHASTIC FIBONACCI',
                'chart': chart_buffer,
                'filters': [
                    f'Stochastic en {"sobreventa" if signal_type == "COMPRA" else "sobrecompra"}',
                    f'K-line: {k_line[-1]:.1f}, D-line: {d_line[-1]:.1f}',
                    f'Precio en nivel Fibonacci clave',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_stochastic_fibonacci_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 6: VWAP REVERSAL
    # ==============================================
    def check_vwap_reversal_signal(self, symbol, interval):
        """Estrategia 6: VWAP Reversal para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['VWAP Reversal']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            vwap = self.calculate_vwap(high, low, close, volume)
            
            volume_ma = self.calculate_sma(volume, 20)
            volume_ratio = volume[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            current_price = close[-1]
            current_vwap = vwap[-1]
            price_vwap_diff = abs(current_price - current_vwap) / current_vwap
            
            if price_vwap_diff > 0.02 or volume_ratio < 1.2:
                return None
            
            signal_type = None
            
            if (current_price > current_vwap and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            elif (current_price < current_vwap and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            if signal_type == 'COMPRA':
                entry = max(current_price, current_vwap)
                stop_loss = entry * 0.99
                take_profit = [entry * 1.01, entry * 1.02, entry * 1.03]
            else:
                entry = min(current_price, current_vwap)
                stop_loss = entry * 1.01
                take_profit = [entry * 0.99, entry * 0.98, entry * 0.97]
            
            signal_key = f"{symbol}_{interval}_VWAP_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_vwap_chart(symbol, interval, df, vwap, 
                                                   {'volume_ratio': volume_ratio}, 
                                                   ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'VWAP REVERSAL',
                'chart': chart_buffer,
                'filters': [
                    f'Precio cerca de VWAP: {price_vwap_diff*100:.1f}%',
                    f'Ratio Volumen: {volume_ratio:.2f}x',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Verificado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_vwap_reversal_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 7: MOMENTUM DIVERGENCE
    # ==============================================
    def check_momentum_divergence_signal(self, symbol, interval):
        """Estrategia 7: Momentum Divergence para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Momentum Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            rsi_traditional = self.calculate_rsi(close, 14)
            rsi_maverick = self.calculate_stochastic_rsi(close)['stoch_rsi']
            
            rsi_bullish, rsi_bearish = self.detect_divergence(close, rsi_traditional)
            rsi_maverick_bullish, rsi_maverick_bearish = self.detect_divergence(close, rsi_maverick)
            
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            
            signal_type = None
            
            if (rsi_bullish[-1] and rsi_maverick_bullish[-1] and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP'] and
                close[-1] > ma9[-1] and close[-1] > ma21[-1]):
                
                signal_type = 'COMPRA'
                
            elif (rsi_bearish[-1] and rsi_maverick_bearish[-1] and
                  volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN'] and
                  close[-1] < ma9[-1] and close[-1] < ma21[-1]):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_momentum_divergence_chart(symbol, interval, df, rsi_traditional, 
                                                                  rsi_maverick, volume_data, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_MOMENTUM_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'MOMENTUM DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    'Divergencia RSI Tradicional confirmada',
                    'Divergencia RSI Maverick confirmada',
                    'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA9 y MA21'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_momentum_divergence_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 8: ADX POWER TREND
    # ==============================================
    def check_adx_power_trend_signal(self, symbol, interval):
        """Estrategia 8: ADX Power Trend para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['ADX Power Trend']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            ma21 = self.calculate_sma(close, 21)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            signal_type = None
            
            if (adx[-1] > 30 and adx[-1] > adx[-2] and
                di_cross_bullish[-1] and
                close[-1] > ma21[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            elif (adx[-1] > 30 and adx[-1] > adx[-2] and
                  di_cross_bearish[-1] and
                  close[-1] < ma21[-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_adx_power_chart(symbol, interval, df, adx, plus_di, minus_di, 
                                                        ma21, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_ADX_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'ADX POWER TREND',
                'chart': chart_buffer,
                'filters': [
                    f'ADX > 30 y creciente: {adx[-1]:.1f}',
                    f'{"+DI > -DI" if signal_type == "COMPRA" else "-DI > +DI"} cruce confirmado',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_adx_power_trend_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 9: MA CONVERGENCE DIVERGENCE
    # ==============================================
    def check_ma_convergence_signal(self, symbol, interval):
        """Estrategia 9: MA Convergence Divergence para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['MA Convergence Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            ma50 = self.calculate_sma(close, 50)
            
            macd, signal, histogram = self.calculate_macd(close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            ma_aligned_bullish = close[-1] > ma9[-1] > ma21[-1] > ma50[-1]
            ma_aligned_bearish = close[-1] < ma9[-1] < ma21[-1] < ma50[-1]
            
            separation_ok_bullish = (ma9[-1] - ma21[-1]) > close[-1] * 0.01 and (ma21[-1] - ma50[-1]) > close[-1] * 0.01
            separation_ok_bearish = (ma21[-1] - ma9[-1]) > close[-1] * 0.01 and (ma50[-1] - ma21[-1]) > close[-1] * 0.01
            
            signal_type = None
            
            if (ma_aligned_bullish and separation_ok_bullish and
                histogram[-1] > 0 and histogram[-2] <= 0 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            elif (ma_aligned_bearish and separation_ok_bearish and
                  histogram[-1] < 0 and histogram[-2] >= 0 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(df['high'].values, df['low'].values, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_ma_convergence_chart(symbol, interval, df, ma9, ma21, ma50, 
                                                             macd, histogram, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_MA_CONVERGENCE_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'MA CONVERGENCE DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    f'Alineación MA9 > MA21 > MA50 confirmada',
                    f'Separación >1% entre medias',
                    f'Histograma MACD positivo para COMPRA/negativo para VENTA',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_ma_convergence_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 10: VOLUME SPIKE MOMENTUM
    # ==============================================
    def check_volume_spike_momentum_signal(self, symbol, interval):
        """Estrategia 10: Volume Spike Momentum para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Volume Spike Momentum']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            volume = df['volume'].values
            
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ma21 = self.calculate_sma(close, 21)
            
            stochastic_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stochastic_data['stoch_rsi']
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            recent_clusters = volume_data['volume_clusters'][-5:]
            has_cluster = sum(recent_clusters) >= 2
            
            if not has_cluster:
                return None
            
            last_signal = volume_data['volume_signal'][-1]
            
            signal_type = None
            
            if last_signal == 'COMPRA' and close[-1] > ma21[-1] and 30 < stoch_rsi[-1] < 70:
                signal_type = 'COMPRA'
            elif last_signal == 'VENTA' and close[-1] < ma21[-1] and 30 < stoch_rsi[-1] < 70:
                signal_type = 'VENTA'
            else:
                return None
            
            if signal_type == 'COMPRA' and ftm_data['strength_signals'][-1] not in ['STRONG_UP', 'WEAK_UP']:
                return None
            if signal_type == 'VENTA' and ftm_data['strength_signals'][-1] not in ['STRONG_DOWN', 'WEAK_DOWN']:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(df['high'].values, df['low'].values, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_volume_spike_chart(symbol, interval, df, volume_data, ma21, 
                                                           stoch_rsi, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_VOLUME_SPIKE_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'VOLUME SPIKE MOMENTUM',
                'chart': chart_buffer,
                'filters': [
                    f'Clúster de volumen confirmado ({sum(recent_clusters)} anomalías)',
                    f'Volumen signal: {last_signal}',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA21',
                    f'Stochastic RSI neutral: {stoch_rsi[-1]:.1f}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_volume_spike_momentum_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 11: STOCHASTIC SUPERTREND
    # ==============================================
    def check_stochastic_supertrend_signal(self, symbol, interval):
        """Estrategia 11: Stochastic Supertrend para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Stochastic Supertrend']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            stochastic = self.calculate_stochastic_rsi(close)
            k_line = stochastic['k_line']
            d_line = stochastic['d_line']
            
            supertrend_data = self.calculate_supertrend(high, low, close, period=7, multiplier=3)
            supertrend_direction = supertrend_data['direction'][-1]
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            if (k_line[-1] < 20 and d_line[-1] < 20 and
                supertrend_direction == 1 and
                close[-1] > supertrend_data['supertrend'][-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            elif (k_line[-1] > 80 and d_line[-1] > 80 and
                  supertrend_direction == -1 and
                  close[-1] < supertrend_data['supertrend'][-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            
            if not signal_type:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            current_price = close[-1]
            if signal_type == 'COMPRA':
                entry = current_price
                stop_loss = supertrend_data['supertrend'][-1]
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                entry = current_price
                stop_loss = supertrend_data['supertrend'][-1]
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_STOCHASTIC_SUPERTREND_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_stochastic_chart(symbol, interval, df, stochastic, 
                                                         ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'STOCHASTIC SUPERTREND',
                'chart': chart_buffer,
                'filters': [
                    f'Stochastic en {"sobreventa" if signal_type == "COMPRA" else "sobrecompra"}',
                    f'K-line: {k_line[-1]:.1f}, D-line: {d_line[-1]:.1f}',
                    f'SuperTrend: {"Alcista" if signal_type == "COMPRA" else "Bajista"}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_stochastic_supertrend_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 12: SUPPORT RESISTANCE BOUNCE
    # ==============================================
    def check_support_resistance_bounce_signal(self, symbol, interval):
        """Estrategia 12: Support Resistance Bounce para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Support Resistance Bounce']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            stochastic_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stochastic_data['stoch_rsi']
            k_line = stochastic_data['k_line']
            d_line = stochastic_data['d_line']
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            current_price = close[-1]
            price_tolerance = 0.02
            
            nearest_support = None
            for support in supports[:2]:
                if support < current_price and abs(current_price - support) / current_price < price_tolerance:
                    nearest_support = support
                    break
            
            nearest_resistance = None
            for resistance in resistances[:2]:
                if resistance > current_price and abs(current_price - resistance) / current_price < price_tolerance:
                    nearest_resistance = resistance
                    break
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            signal_type = None
            
            if (nearest_support is not None and
                current_price <= nearest_support * 1.01 and
                stoch_rsi[-1] < 30 and
                k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2] and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                current_price <= bb_lower[-1] * 1.02 and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            elif (nearest_resistance is not None and
                  current_price >= nearest_resistance * 0.99 and
                  stoch_rsi[-1] > 70 and
                  k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2] and
                  volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                  current_price >= bb_upper[-1] * 0.98 and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_support_resistance_bounce_chart(symbol, interval, df, 
                                                                       stochastic_data, supports, 
                                                                       resistances, volume_data, 
                                                                       bb_upper, bb_lower, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_SR_BOUNCE_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'SUPPORT RESISTANCE BOUNCE',
                'chart': chart_buffer,
                'filters': [
                    f'Rebote en {"soporte" if signal_type == "COMPRA" else "resistencia"} confirmado',
                    f'RSI Estocástico {"sobreventa" if signal_type == "COMPRA" else "sobrecompra"}',
                    f'Cruce K/D {"alcista" if signal_type == "COMPRA" else "bajista"} confirmado',
                    f'Clúster de volumen confirmado',
                    f'Precio cerca banda {"inferior" if signal_type == "COMPRA" else "superior"} BB',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_support_resistance_bounce_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 13: WHALE DMI COMBO
    # ==============================================
    def check_whale_dmi_combo_signal(self, symbol, interval):
        """Estrategia 13: Whale DMI Combo para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Whale DMI Combo']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            whale_data = self.calculate_whale_signals_improved(df)
            
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            
            whale_signal_active = whale_data['extended_buy'][-1] or whale_data['extended_sell'][-1]
            
            if not whale_signal_active:
                return None
            
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            signal_type = None
            
            if (whale_data['extended_buy'][-1] and di_cross_bullish[-1] and
                adx[-1] > 25 and close[-1] > ma9[-1] and close[-1] > ma21[-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            elif (whale_data['extended_sell'][-1] and di_cross_bearish[-1] and
                  adx[-1] > 25 and close[-1] < ma9[-1] and close[-1] < ma21[-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_whale_dmi_combo_chart(symbol, interval, df, whale_data, 
                                                              adx, plus_di, minus_di, 
                                                              ma9, ma21, ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_WHALE_DMI_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'WHALE DMI COMBO',
                'chart': chart_buffer,
                'filters': [
                    f'Señal ballenas extendida (7 velas) confirmada',
                    f'{"+DI > -DI" if signal_type == "COMPRA" else "-DI > +DI"} cruce confirmado',
                    f'ADX > 25: {adx[-1]:.1f}',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA9 y MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_whale_dmi_combo_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 14: PARABOLIC SAR MOMENTUM
    # ==============================================
    def check_parabolic_sar_momentum_signal(self, symbol, interval):
        """Estrategia 14: Parabolic SAR Momentum para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Parabolic SAR Momentum']:
            return None
        if not self.is_operational_timeframe(interval):
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            parabolic = self.calculate_parabolic_sar(high, low)
            sar_trend = parabolic['trend'][-1]
            
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            if sar_trend == 1 and adx[-1] > 25 and plus_di[-1] > minus_di[-1]:
                signal_type = 'COMPRA'
            elif sar_trend == -1 and adx[-1] > 25 and minus_di[-1] > plus_di[-1]:
                signal_type = 'VENTA'
            else:
                return None
            
            if signal_type == 'COMPRA' and ftm_data['strength_signals'][-1] not in ['STRONG_UP', 'WEAK_UP']:
                return None
            if signal_type == 'VENTA' and ftm_data['strength_signals'][-1] not in ['STRONG_DOWN', 'WEAK_DOWN']:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            current_price = close[-1]
            if signal_type == 'COMPRA':
                entry = current_price
                stop_loss = parabolic['sar'][-1]
                take_profit = [entry * 1.02, entry * 1.04, entry * 1.06]
            else:
                entry = current_price
                stop_loss = parabolic['sar'][-1]
                take_profit = [entry * 0.98, entry * 0.96, entry * 0.94]
            
            signal_key = f"{symbol}_{interval}_PARABOLIC_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            chart_buffer = self.generate_parabolic_chart(symbol, interval, df, parabolic, adx, 
                                                        plus_di, minus_di, ftm_data, signal_type)
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'PARABOLIC SAR MOMENTUM',
                'chart': chart_buffer,
                'filters': [
                    f'Parabolic SAR: {"Alcista" if signal_type == "COMPRA" else "Bajista"}',
                    f'ADX: {adx[-1]:.1f} > 25',
                    f'{"+DI" if signal_type == "COMPRA" else "-DI"} > {"-DI" if signal_type == "COMPRA" else "+DI"}',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_parabolic_sar_momentum_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 15: MULTI-TIMEFRAME CONFIRMATION
    # ==============================================
    def check_multi_timeframe_confirmation_signal(self, symbol, interval):
        """Estrategia 15: Multi-Timeframe Confirmation para SPOT"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Multi-Timeframe Confirmation']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            stochastic_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stochastic_data['stoch_rsi']
            k_line = stochastic_data['k_line']
            d_line = stochastic_data['d_line']
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            hierarchy = TIMEFRAME_HIERARCHY.get(interval, {})
            if not hierarchy:
                return None
            
            menor_df = self.get_kucoin_data(symbol, '2h', 50)
            if menor_df is None or len(menor_df) < 20:
                return None
            
            menor_close = menor_df['close'].values
            menor_stochastic_data = self.calculate_stochastic_rsi(menor_close)
            menor_stoch_rsi = menor_stochastic_data['stoch_rsi']
            
            if interval == '4h':
                mayor_tf = '8h'
            elif interval == '12h':
                mayor_tf = '1D'
            elif interval == '1D':
                mayor_tf = '1W'
            else:
                return None
            
            mayor_df = self.get_kucoin_data(symbol, mayor_tf, 50)
            if mayor_df is None or len(mayor_df) < 20:
                return None
            
            mayor_close = mayor_df['close'].values
            mayor_ma50 = self.calculate_sma(mayor_close, 50)
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            current_price = close[-1]
            current_condition = (stoch_rsi[-1] < 30 and 
                               k_line[-1] > d_line[-1] and 
                               k_line[-2] <= d_line[-2])
            
            menor_condition = menor_stoch_rsi[-1] < 40
            
            mayor_condition = current_price > mayor_ma50[-1] if len(mayor_ma50) > 0 else True
            
            signal_type = None
            
            if (current_condition and menor_condition and mayor_condition and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'COMPRA' and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
                
            current_condition_short = (stoch_rsi[-1] > 70 and 
                                     k_line[-1] < d_line[-1] and 
                                     k_line[-2] >= d_line[-2])
            
            menor_condition_short = menor_stoch_rsi[-1] > 60
            
            mayor_condition_short = current_price < mayor_ma50[-1] if len(mayor_ma50) > 0 else True
            
            if (current_condition_short and menor_condition_short and mayor_condition_short and
                volume_data['volume_clusters'][-1] and volume_data['volume_signal'][-1] == 'VENTA' and
                ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_multi_timeframe_confirmation_chart(symbol, interval, df, 
                                                                           stochastic_data, menor_stochastic_data,
                                                                           mayor_ma50, supports, 
                                                                           resistances, volume_data, 
                                                                           ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_MULTITF_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': current_price,
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'MULTI-TIMEFRAME CONFIRMATION',
                'chart': chart_buffer,
                'filters': [
                    f'RSI Estocástico {"sobreventa" if signal_type == "COMPRA" else "sobrecompra"} (TF actual)',
                    f'Cruce K/D {"alcista" if signal_type == "COMPRA" else "bajista"} confirmado',
                    f'Confirmación RSI Estocástico TF menor',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA50 TF mayor',
                    f'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_multi_timeframe_confirmation_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 16: RSI MAVERICK DMI DIVERGENCE
    # ==============================================
    def check_rsi_maverick_dmi_divergence_signal(self, symbol, interval):
        """Estrategia 16: RSI Maverick + DMI Divergence (NUEVA)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['RSI Maverick DMI Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Obtener divergencias RSI Maverick (extendidas 7 velas)
            rsi_maverick_div = self.get_rsi_maverick_divergence_signals(symbol, interval)
            bullish_div = rsi_maverick_div['bullish']
            bearish_div = rsi_maverick_div['bearish']
            
            if not bullish_div[-1] and not bearish_div[-1]:
                return None
            
            # Calcular DMI/ADX
            adx, plus_di, minus_di = self.calculate_adx(high, low, close)
            di_cross_bullish, di_cross_bearish = self.check_di_crossover(plus_di, minus_di)
            
            # Verificar condiciones de tendencia
            ma21 = self.calculate_sma(close, 21)
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            # Señal COMPRA: Divergencia alcista RSI Maverick + Cruce DMI positivo + ADX fuerte
            if (bullish_div[-1] and 
                di_cross_bullish[-1] and 
                adx[-1] > 25 and
                close[-1] > ma21[-1] and
                volume_data['volume_clusters'][-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            # Señal VENTA: Divergencia bajista RSI Maverick + Cruce DMI negativo + ADX fuerte
            elif (bearish_div[-1] and 
                  di_cross_bearish[-1] and 
                  adx[-1] > 25 and
                  close[-1] < ma21[-1] and
                  volume_data['volume_clusters'][-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_rsi_maverick_dmi_chart(symbol, interval, df, 
                                                               rsi_maverick_div['rsi_maverick'],
                                                               bullish_div, bearish_div,
                                                               adx, plus_di, minus_di,
                                                               di_cross_bullish, di_cross_bearish,
                                                               ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_RSI_MAVERICK_DMI_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'RSI MAVERICK DMI DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    f'Divergencia RSI Maverick {"alcista" if signal_type == "COMPRA" else "bajista"} (7 velas extendida)',
                    f'Cruce {"+DI > -DI" if signal_type == "COMPRA" else "-DI > +DI"} confirmado',
                    f'ADX > 25: {adx[-1]:.1f}',
                    f'Clúster de volumen confirmado',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_dmi_divergence_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 17: RSI MAVERICK STOCHASTIC CROSSOVER
    # ==============================================
    def check_rsi_maverick_stochastic_crossover_signal(self, symbol, interval):
        """Estrategia 17: RSI Maverick + Stochastic Crossover (NUEVA)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['RSI Maverick Stochastic Crossover']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Obtener divergencias RSI Maverick
            rsi_maverick_div = self.get_rsi_maverick_divergence_signals(symbol, interval)
            bullish_div = rsi_maverick_div['bullish']
            bearish_div = rsi_maverick_div['bearish']
            
            if not bullish_div[-1] and not bearish_div[-1]:
                return None
            
            # Calcular Stochastic RSI
            stochastic_data = self.calculate_stochastic_rsi(close)
            stoch_rsi = stochastic_data['stoch_rsi']
            k_line = stochastic_data['k_line']
            d_line = stochastic_data['d_line']
            
            # Verificar cruces Stochastic
            stochastic_cross_bullish = k_line[-1] > d_line[-1] and k_line[-2] <= d_line[-2]
            stochastic_cross_bearish = k_line[-1] < d_line[-1] and k_line[-2] >= d_line[-2]
            
            # Verificar niveles Stochastic
            stochastic_oversold = stoch_rsi[-1] < 30
            stochastic_overbought = stoch_rsi[-1] > 70
            
            # Verificar condiciones de tendencia
            ma21 = self.calculate_sma(close, 21)
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            # Señal COMPRA: Divergencia alcista RSI Maverick + Cruce Stochastic positivo + Oversold
            if (bullish_div[-1] and 
                stochastic_cross_bullish and
                stochastic_oversold and
                close[-1] > ma21[-1] and
                volume_data['volume_clusters'][-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            # Señal VENTA: Divergencia bajista RSI Maverick + Cruce Stochastic negativo + Overbought
            elif (bearish_div[-1] and 
                  stochastic_cross_bearish and
                  stochastic_overbought and
                  close[-1] < ma21[-1] and
                  volume_data['volume_clusters'][-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_rsi_maverick_stochastic_chart(symbol, interval, df, 
                                                                     rsi_maverick_div['rsi_maverick'],
                                                                     bullish_div, bearish_div,
                                                                     stochastic_data,
                                                                     ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_RSI_MAVERICK_STOCHASTIC_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'RSI MAVERICK STOCHASTIC CROSSOVER',
                'chart': chart_buffer,
                'filters': [
                    f'Divergencia RSI Maverick {"alcista" if signal_type == "COMPRA" else "bajista"} (7 velas extendida)',
                    f'Cruce Stochastic {"alcista" if signal_type == "COMPRA" else "bajista"} confirmado',
                    f'Stochastic RSI {"sobreventa" if signal_type == "COMPRA" else "sobrecompra"}: {stoch_rsi[-1]:.1f}',
                    f'Clúster de volumen confirmado',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_stochastic_crossover_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 18: RSI MAVERICK MACD DIVERGENCE
    # ==============================================
    def check_rsi_maverick_macd_divergence_signal(self, symbol, interval):
        """Estrategia 18: RSI Maverick + MACD Divergence (NUEVA)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['RSI Maverick MACD Divergence']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Obtener divergencias RSI Maverick
            rsi_maverick_div = self.get_rsi_maverick_divergence_signals(symbol, interval)
            bullish_div = rsi_maverick_div['bullish']
            bearish_div = rsi_maverick_div['bearish']
            
            if not bullish_div[-1] and not bearish_div[-1]:
                return None
            
            # Calcular MACD
            macd, signal, histogram = self.calculate_macd(close)
            
            # Detectar divergencias MACD
            macd_bullish, macd_bearish = self.detect_divergence(close, macd)
            
            # Verificar cruces MACD
            macd_cross_bullish = macd[-1] > signal[-1] and macd[-2] <= signal[-2]
            macd_cross_bearish = macd[-1] < signal[-1] and macd[-2] >= signal[-2]
            
            # Verificar condiciones de tendencia
            ma21 = self.calculate_sma(close, 21)
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            # Señal COMPRA: Divergencia RSI Maverick + Divergencia MACD + Cruce MACD positivo
            if (bullish_div[-1] and 
                (macd_bullish[-1] or macd_cross_bullish) and
                histogram[-1] > 0 and
                close[-1] > ma21[-1] and
                volume_data['volume_clusters'][-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            # Señal VENTA: Divergencia RSI Maverick + Divergencia MACD + Cruce MACD negativo
            elif (bearish_div[-1] and 
                  (macd_bearish[-1] or macd_cross_bearish) and
                  histogram[-1] < 0 and
                  close[-1] < ma21[-1] and
                  volume_data['volume_clusters'][-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_rsi_maverick_macd_chart(symbol, interval, df, 
                                                               rsi_maverick_div['rsi_maverick'],
                                                               bullish_div, bearish_div,
                                                               macd, signal, histogram,
                                                               macd_bullish, macd_bearish,
                                                               ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_RSI_MAVERICK_MACD_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'RSI MAVERICK MACD DIVERGENCE',
                'chart': chart_buffer,
                'filters': [
                    f'Divergencia RSI Maverick {"alcista" if signal_type == "COMPRA" else "bajista"} (7 velas extendida)',
                    f'{"Divergencia" if (macd_bullish[-1] if signal_type == "COMPRA" else macd_bearish[-1]) else "Cruce"} MACD {"alcista" if signal_type == "COMPRA" else "bajista"}',
                    f'Histograma MACD {"positivo" if signal_type == "COMPRA" else "negativo"}',
                    f'Clúster de volumen confirmado',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA21',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_macd_divergence_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 19: WHALE RSI MAVERICK COMBO
    # ==============================================
    def check_whale_rsi_maverick_combo_signal(self, symbol, interval):
        """Estrategia 19: Whale + RSI Maverick Combo para 12h y 1D (NUEVA)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['Whale RSI Maverick Combo']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Obtener señales de ballenas
            whale_data = self.calculate_whale_signals_improved(df)
            
            # Obtener divergencias RSI Maverick
            rsi_maverick_div = self.get_rsi_maverick_divergence_signals(symbol, interval)
            bullish_div = rsi_maverick_div['bullish']
            bearish_div = rsi_maverick_div['bearish']
            
            # Verificar que haya señal de ballenas activa
            whale_signal_active = whale_data['extended_buy'][-1] or whale_data['extended_sell'][-1]
            
            if not whale_signal_active:
                return None
            
            # Verificar condiciones de tendencia
            ma50 = self.calculate_sma(close, 50)
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            # Señal COMPRA: Ballenas compradoras + Divergencia RSI Maverick alcista
            if (whale_data['extended_buy'][-1] and 
                bullish_div[-1] and
                close[-1] > ma50[-1] and
                volume_data['volume_clusters'][-1] and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            # Señal VENTA: Ballenas vendedoras + Divergencia RSI Maverick bajista
            elif (whale_data['extended_sell'][-1] and 
                  bearish_div[-1] and
                  close[-1] < ma50[-1] and
                  volume_data['volume_clusters'][-1] and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_whale_rsi_maverick_chart(symbol, interval, df, 
                                                                whale_data,
                                                                rsi_maverick_div['rsi_maverick'],
                                                                bullish_div, bearish_div,
                                                                ma50, volume_data,
                                                                ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_WHALE_RSI_MAVERICK_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'WHALE RSI MAVERICK COMBO',
                'chart': chart_buffer,
                'filters': [
                    f'Señal ballenas {"compradoras" if signal_type == "COMPRA" else "vendedoras"} extendida (7 velas)',
                    f'Divergencia RSI Maverick {"alcista" if signal_type == "COMPRA" else "bajista"} (7 velas extendida)',
                    f'Precio {" > " if signal_type == "COMPRA" else " < "} MA50',
                    f'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}',
                    f'Multi-Timeframe: Confirmado para {interval}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_whale_rsi_maverick_combo_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ESTRATEGIA 20: RSI MAVERICK TREND REVERSAL
    # ==============================================
    def check_rsi_maverick_trend_reversal_signal(self, symbol, interval):
        """Estrategia 20: RSI Maverick Trend Reversal (NUEVA)"""
        if symbol not in TOP_CRYPTO_SYMBOLS:
            return None
        if interval not in STRATEGY_TIMEFRAMES['RSI Maverick Trend Reversal']:
            return None
        
        try:
            df = self.get_kucoin_data(symbol, interval, 100)
            if df is None or len(df) < 50:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Obtener divergencias RSI Maverick
            rsi_maverick_div = self.get_rsi_maverick_divergence_signals(symbol, interval)
            bullish_div = rsi_maverick_div['bullish']
            bearish_div = rsi_maverick_div['bearish']
            
            if not bullish_div[-1] and not bearish_div[-1]:
                return None
            
            # Calcular indicadores de tendencia
            ma9 = self.calculate_sma(close, 9)
            ma21 = self.calculate_sma(close, 21)
            ma50 = self.calculate_sma(close, 50)
            
            # Detectar cambio de tendencia
            trend_bullish = close[-1] > ma9[-1] > ma21[-1] > ma50[-1]
            trend_bearish = close[-1] < ma9[-1] < ma21[-1] < ma50[-1]
            trend_reversal_bullish = close[-1] > ma9[-1] and ma9[-1] > ma21[-1] and close[-2] <= ma9[-2]
            trend_reversal_bearish = close[-1] < ma9[-1] and ma9[-1] < ma21[-1] and close[-2] >= ma9[-2]
            
            # Calcular RSI tradicional para confirmación
            rsi_traditional = self.calculate_rsi(close, 14)
            
            volume = df['volume'].values
            volume_data = self.calculate_volume_anomaly(volume, close)
            
            ftm_data = self.calculate_trend_strength_maverick(close)
            if ftm_data['no_trade_zones'][-1]:
                return None
            
            signal_type = None
            
            # Señal COMPRA: Divergencia RSI Maverick + Reversión tendencia alcista + RSI tradicional no sobrecomprado
            if (bullish_div[-1] and 
                (trend_reversal_bullish or trend_bullish) and
                rsi_traditional[-1] < 60 and
                volume_data['volume_clusters'][-1] and
                volume_data['volume_signal'][-1] == 'COMPRA' and
                ftm_data['strength_signals'][-1] in ['STRONG_UP', 'WEAK_UP']):
                
                signal_type = 'COMPRA'
            
            # Señal VENTA: Divergencia RSI Maverick + Reversión tendencia bajista + RSI tradicional no sobrevendido
            elif (bearish_div[-1] and 
                  (trend_reversal_bearish or trend_bearish) and
                  rsi_traditional[-1] > 40 and
                  volume_data['volume_clusters'][-1] and
                  volume_data['volume_signal'][-1] == 'VENTA' and
                  ftm_data['strength_signals'][-1] in ['STRONG_DOWN', 'WEAK_DOWN']):
                
                signal_type = 'VENTA'
            else:
                return None
            
            multi_tf_ok = self.check_multi_timeframe_obligatory(symbol, interval, signal_type)
            if not multi_tf_ok:
                return None
            
            supports, resistances = self.calculate_support_resistance_channels(high, low, close)
            levels_data = self.calculate_optimal_entry_exit(df, signal_type, 1, supports, resistances)
            
            chart_buffer = self.generate_rsi_maverick_trend_chart(symbol, interval, df, 
                                                                rsi_maverick_div['rsi_maverick'],
                                                                bullish_div, bearish_div,
                                                                ma9, ma21, ma50,
                                                                rsi_traditional, volume_data,
                                                                ftm_data, signal_type)
            
            signal_key = f"{symbol}_{interval}_RSI_MAVERICK_TREND_{signal_type}"
            current_timestamp = int(time.time() / 60)
            
            if signal_key in self.strategy_signals:
                last_sent = self.strategy_signals[signal_key]
                if current_timestamp - last_sent < 60:
                    return None
            
            self.strategy_signals[signal_key] = current_timestamp
            
            signal_data = {
                'symbol': symbol,
                'interval': interval,
                'signal': signal_type,
                'current_price': close[-1],
                'entry': levels_data['entry'],
                'stop_loss': levels_data['stop_loss'],
                'take_profit': levels_data['take_profit'],
                'support_levels': supports[:3],
                'resistance_levels': resistances[:3],
                'strategy': 'RSI MAVERICK TREND REVERSAL',
                'chart': chart_buffer,
                'filters': [
                    f'Divergencia RSI Maverick {"alcista" if signal_type == "COMPRA" else "bajista"} (7 velas extendida)',
                    f'{"Reversión" if trend_reversal_bullish or trend_reversal_bearish else "Tendencia"} {"alcista" if signal_type == "COMPRA" else "bajista"} confirmada',
                    f'RSI Tradicional: {rsi_traditional[-1]:.1f} (no {"sobrecomprado" if signal_type == "COMPRA" else "sobrevendido"})',
                    f'Clúster de volumen confirmado',
                    f'FTMaverick: {ftm_data["strength_signals"][-1]}'
                ]
            }
            
            if self.should_send_telegram_alert(interval):
                signal_data['send_telegram'] = True
            else:
                signal_data['send_telegram'] = False
            
            return signal_data
            
        except Exception as e:
            print(f"Error en check_rsi_maverick_trend_reversal_signal para {symbol} {interval}: {e}")
            return None

    # ==============================================
    # ALERTAS DE SALIDA - SOPORTES Y RESISTENCIAS
    # ==============================================
    def check_exit_signals(self):
        """Verificar señales de salida para SPOT"""
        exit_alerts = []
        current_time = self.get_bolivia_time()
        current_timestamp = int(time.time() / 60)
        
        for symbol in TOP_CRYPTO_SYMBOLS:
            for interval in TELEGRAM_ALERT_TIMEFRAMES:
                try:
                    df = self.get_kucoin_data(symbol, interval, 50)
                    if df is None or len(df) < 20:
                        continue
                    
                    close = df['close'].values
                    high = df['high'].values
                    low = df['low'].values
                    open_prices = df['open'].values
                    
                    supports, resistances = self.calculate_support_resistance_channels(high, low, close)
                    
                    current_price = close[-1]
                    current_high = high[-1]
                    current_low = low[-1]
                    current_open = open_prices[-1]
                    
                    candle_timestamp = int(df['timestamp'].iloc[-1].timestamp() / 60)
                    
                    # Verificar toque de RESISTENCIA (para COMPRAS - tomar ganancias)
                    for resistance in resistances[:3]:
                        if resistance > 0:
                            if abs(current_high - resistance) / resistance < 0.005:
                                if close[-1] > open_prices[-1]:
                                    signal_key = f"{symbol}_{interval}_RESISTANCE_{resistance:.6f}_{candle_timestamp}"
                                    
                                    if signal_key not in self.support_resistance_alerts:
                                        ftm_data = self.calculate_trend_strength_maverick(close)
                                        
                                        chart_buffer = self.generate_support_resistance_chart(
                                            symbol, interval, df, supports, resistances,
                                            resistance, 'VENTA', ftm_data
                                        )
                                        
                                        alert = {
                                            'symbol': symbol,
                                            'interval': interval,
                                            'type': 'RESISTANCE_TOUCH_COMPRA',
                                            'message': f"🚨 {symbol} ha tocado la resistencia de {resistance:.6f} BTC en {interval} 🚨\n📈 Recomendación: Tomar ganancias totales o parciales de operación COMPRA.",
                                            'chart': chart_buffer
                                        }
                                        exit_alerts.append(alert)
                                        self.support_resistance_alerts[signal_key] = current_timestamp
                                        break
                    
                    # Verificar toque de SOPORTE (para VENTAS - tomar ganancias)
                    for support in supports[:3]:
                        if support > 0:
                            if abs(current_low - support) / support < 0.005:
                                if close[-1] < open_prices[-1]:
                                    signal_key = f"{symbol}_{interval}_SUPPORT_{support:.6f}_{candle_timestamp}"
                                    
                                    if signal_key not in self.support_resistance_alerts:
                                        ftm_data = self.calculate_trend_strength_maverick(close)
                                        
                                        chart_buffer = self.generate_support_resistance_chart(
                                            symbol, interval, df, supports, resistances,
                                            support, 'COMPRA', ftm_data
                                        )
                                        
                                        alert = {
                                            'symbol': symbol,
                                            'interval': interval,
                                            'type': 'SUPPORT_TOUCH_VENTA',
                                            'message': f"🚨 {symbol} ha tocado el soporte de {support:.6f} BTC en {interval} 🚨\n📉 Recomendación: Tomar ganancias totales o parciales de operación VENTA.",
                                            'chart': chart_buffer
                                        }
                                        exit_alerts.append(alert)
                                        self.support_resistance_alerts[signal_key] = current_timestamp
                                        break
                    
                    keys_to_remove = []
                    for key, timestamp in self.support_resistance_alerts.items():
                        if current_timestamp - timestamp > 1440:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.support_resistance_alerts[key]
                    
                except Exception as e:
                    print(f"Error verificando soportes/resistencias para {symbol} {interval}: {e}")
                    continue
        
        return exit_alerts

    # ==============================================
    # GENERACIÓN DE GRÁFICOS PARA NUEVAS ESTRATEGIAS
    # ==============================================
    
    def generate_rsi_maverick_dmi_chart(self, symbol, interval, df, rsi_maverick, 
                                       bullish_div, bearish_div,
                                       adx, plus_di, minus_di,
                                       di_cross_bullish, di_cross_bearish,
                                       ftm_data, signal_type):
        """Generar gráfico para RSI Maverick + DMI Divergence"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de precios
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'RSI Maverick + DMI Divergence - {symbol} - {interval} - Señal {signal_type}')
            ax1.set_ylabel('Precio (BTC)')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de RSI Maverick
            ax2.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            for i in range(len(dates_matplotlib)):
                if bullish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='green', s=50, marker='^', label='Div Alcista' if i == 0 else "")
                if bearish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='red', s=50, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico de ADX/DMI
            ax3.plot(dates_matplotlib, adx[-50:], 'black', linewidth=2, label='ADX')
            ax3.plot(dates_matplotlib, plus_di[-50:], 'green', linewidth=1, label='+DI')
            ax3.plot(dates_matplotlib, minus_di[-50:], 'red', linewidth=1, label='-DI')
            
            # Marcar cruces DMI
            for i in range(len(dates_matplotlib)):
                if di_cross_bullish[-50+i]:
                    ax3.scatter(dates_matplotlib[i], plus_di[-50+i], color='green', s=80, marker='^', label='Cruce +DI' if i == 0 else "")
                if di_cross_bearish[-50+i]:
                    ax3.scatter(dates_matplotlib[i], minus_di[-50+i], color='red', s=80, marker='v', label='Cruce -DI' if i == 0 else "")
            
            ax3.set_ylabel('ADX/DMI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Maverick DMI: {e}")
            return None
    
    def generate_rsi_maverick_stochastic_chart(self, symbol, interval, df, rsi_maverick, 
                                              bullish_div, bearish_div,
                                              stochastic_data, ftm_data, signal_type):
        """Generar gráfico para RSI Maverick + Stochastic Crossover"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de precios
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'RSI Maverick + Stochastic Crossover - {symbol} - {interval} - Señal {signal_type}')
            ax1.set_ylabel('Precio (BTC)')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de RSI Maverick
            ax2.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            for i in range(len(dates_matplotlib)):
                if bullish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='green', s=50, marker='^', label='Div Alcista' if i == 0 else "")
                if bearish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='red', s=50, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico de Stochastic RSI
            ax3.plot(dates_matplotlib, stochastic_data['stoch_rsi'][-50:], 'blue', linewidth=1, label='Stoch RSI')
            ax3.plot(dates_matplotlib, stochastic_data['k_line'][-50:], 'green', linewidth=1, label='%K')
            ax3.plot(dates_matplotlib, stochastic_data['d_line'][-50:], 'red', linewidth=1, label='%D')
            
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            
            ax3.set_ylabel('Stochastic RSI')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Maverick Stochastic: {e}")
            return None
    
    def generate_rsi_maverick_macd_chart(self, symbol, interval, df, rsi_maverick, 
                                        bullish_div, bearish_div,
                                        macd, signal, histogram,
                                        macd_bullish, macd_bearish,
                                        ftm_data, signal_type):
        """Generar gráfico para RSI Maverick + MACD Divergence"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de precios
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            ax1.set_title(f'RSI Maverick + MACD Divergence - {symbol} - {interval} - Señal {signal_type}')
            ax1.set_ylabel('Precio (BTC)')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de RSI Maverick
            ax2.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            for i in range(len(dates_matplotlib)):
                if bullish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='green', s=50, marker='^', label='Div Alcista' if i == 0 else "")
                if bearish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='red', s=50, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico de MACD
            ax3.plot(dates_matplotlib, macd[-50:], 'blue', linewidth=1, label='MACD')
            ax3.plot(dates_matplotlib, signal[-50:], 'red', linewidth=1, label='Señal')
            
            # Histograma MACD
            colors = ['green' if h >= 0 else 'red' for h in histogram[-50:]]
            ax3.bar(dates_matplotlib, histogram[-50:], color=colors, alpha=0.6, width=0.8)
            
            # Marcar divergencias MACD
            for i in range(len(dates_matplotlib)):
                if macd_bullish[-50+i]:
                    ax3.scatter(dates_matplotlib[i], macd[-50+i], color='green', s=80, marker='^', label='Div MACD Alcista' if i == 0 else "")
                if macd_bearish[-50+i]:
                    ax3.scatter(dates_matplotlib[i], macd[-50+i], color='red', s=80, marker='v', label='Div MACD Bajista' if i == 0 else "")
            
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Maverick MACD: {e}")
            return None
    
    def generate_whale_rsi_maverick_chart(self, symbol, interval, df, whale_data,
                                         rsi_maverick, bullish_div, bearish_div,
                                         ma50, volume_data, ftm_data, signal_type):
        """Generar gráfico para Whale + RSI Maverick Combo"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de precios
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            # MA50
            ax1.plot(dates_matplotlib, ma50[-50:], 'orange', linewidth=2, label='MA50')
            
            ax1.set_title(f'Whale + RSI Maverick Combo - {symbol} - {interval} - Señal {signal_type}')
            ax1.set_ylabel('Precio (BTC)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de RSI Maverick
            ax2.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            for i in range(len(dates_matplotlib)):
                if bullish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='green', s=50, marker='^', label='Div Alcista' if i == 0 else "")
                if bearish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='red', s=50, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico de señales de ballenas
            ax3.bar(dates_matplotlib, whale_data['whale_pump'][-50:], color='green', alpha=0.7, label='Ballenas Compradoras')
            ax3.bar(dates_matplotlib, whale_data['whale_dump'][-50:], color='red', alpha=0.7, label='Ballenas Vendedoras')
            
            # Marcar señales extendidas
            for i in range(len(dates_matplotlib)):
                if whale_data['extended_buy'][-50+i]:
                    ax3.scatter(dates_matplotlib[i], max(whale_data['whale_pump'][-50:]) * 1.1, 
                              color='green', s=100, marker='^', label='Señal Compra Extendida' if i == 0 else "")
                if whale_data['extended_sell'][-50+i]:
                    ax3.scatter(dates_matplotlib[i], max(whale_data['whale_dump'][-50:]) * 1.1, 
                              color='red', s=100, marker='v', label='Señal Venta Extendida' if i == 0 else "")
            
            ax3.set_ylabel('Señales Ballenas')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico de volumen
            volumes = df['volume'].iloc[-50:].values
            volume_colors = []
            for i in range(len(dates_matplotlib)):
                signal = volume_data['volume_signal'][-50+i] if len(volume_data['volume_signal']) > -50+i else 'NEUTRAL'
                if signal == 'COMPRA':
                    volume_colors.append('green')
                elif signal == 'VENTA':
                    volume_colors.append('red')
                else:
                    volume_colors.append('gray')
            
            ax4.bar(dates_matplotlib, volumes, color=volume_colors, alpha=0.6)
            
            # Marcar clusters de volumen
            for i in range(len(dates_matplotlib)):
                if volume_data['volume_clusters'][-50+i]:
                    ax4.scatter(dates_matplotlib[i], max(volumes) * 1.1, 
                              color='orange', s=50, marker='*', label='Cluster Volumen' if i == 0 else "")
            
            ax4.set_ylabel('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico Whale RSI Maverick: {e}")
            return None
    
    def generate_rsi_maverick_trend_chart(self, symbol, interval, df, rsi_maverick, 
                                         bullish_div, bearish_div,
                                         ma9, ma21, ma50,
                                         rsi_traditional, volume_data,
                                         ftm_data, signal_type):
        """Generar gráfico para RSI Maverick Trend Reversal"""
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
            
            dates = df['timestamp'].iloc[-50:].values
            dates_matplotlib = mdates.date2num(dates)
            
            # Gráfico de precios con MAs
            for i in range(len(dates_matplotlib)):
                open_price = df['open'].iloc[-50+i]
                close_price = df['close'].iloc[-50+i]
                high_price = df['high'].iloc[-50+i]
                low_price = df['low'].iloc[-50+i]
                
                color = 'green' if close_price >= open_price else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [low_price, high_price], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [open_price, close_price], color=color, linewidth=3)
            
            # Medias móviles
            ax1.plot(dates_matplotlib, ma9[-50:], 'red', linewidth=1, label='MA9')
            ax1.plot(dates_matplotlib, ma21[-50:], 'blue', linewidth=1, label='MA21')
            ax1.plot(dates_matplotlib, ma50[-50:], 'orange', linewidth=2, label='MA50')
            
            ax1.set_title(f'RSI Maverick Trend Reversal - {symbol} - {interval} - Señal {signal_type}')
            ax1.set_ylabel('Precio (BTC)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de RSI Maverick
            ax2.plot(dates_matplotlib, rsi_maverick[-50:], 'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            for i in range(len(dates_matplotlib)):
                if bullish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='green', s=50, marker='^', label='Div Alcista' if i == 0 else "")
                if bearish_div[-50+i]:
                    ax2.scatter(dates_matplotlib[i], rsi_maverick[-50+i], color='red', s=50, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.set_ylabel('RSI Maverick')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Gráfico de RSI Tradicional
            ax3.plot(dates_matplotlib, rsi_traditional[-50:], 'purple', linewidth=2, label='RSI Tradicional')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            
            ax3.set_ylabel('RSI Tradicional')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gráfico de volumen
            volumes = df['volume'].iloc[-50:].values
            volume_colors = []
            for i in range(len(dates_matplotlib)):
                signal = volume_data['volume_signal'][-50+i] if len(volume_data['volume_signal']) > -50+i else 'NEUTRAL'
                if signal == 'COMPRA':
                    volume_colors.append('green')
                elif signal == 'VENTA':
                    volume_colors.append('red')
                else:
                    volume_colors.append('gray')
            
            ax4.bar(dates_matplotlib, volumes, color=volume_colors, alpha=0.6)
            
            # Marcar clusters de volumen
            for i in range(len(dates_matplotlib)):
                if volume_data['volume_clusters'][-50+i]:
                    ax4.scatter(dates_matplotlib[i], max(volumes) * 1.1, 
                              color='orange', s=50, marker='*', label='Cluster Volumen' if i == 0 else "")
            
            ax4.set_ylabel('Volumen')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generando gráfico RSI Maverick Trend: {e}")
            return None

    # ==============================================
    # SISTEMA DE GENERACIÓN DE SEÑALES COMPLETO
    # ==============================================
    def generate_strategy_signals(self):
        """Generar señales para todas las estrategias"""
        all_signals = []
        
        intervals_to_check = ['4h', '12h', '1D', '1W']
        
        current_time = self.get_bolivia_time()
        
        for interval in intervals_to_check:
            should_check = self.calculate_remaining_time(interval, current_time)
            if not should_check:
                continue
            
            for symbol in TOP_CRYPTO_SYMBOLS:
                try:
                    # Estrategia 1: Ichimoku Cloud Breakout
                    if interval in STRATEGY_TIMEFRAMES['Ichimoku Cloud Breakout']:
                        signal = self.check_ichimoku_cloud_breakout_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 2: Fibonacci Supertrend
                    if interval in STRATEGY_TIMEFRAMES['Fibonacci Supertrend']:
                        signal = self.check_fibonacci_supertrend_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 3: Cryptodivisa
                    if interval in STRATEGY_TIMEFRAMES['Cryptodivisa']:
                        signal = self.check_cryptodivisa_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 4: Reversion Soportes Resistencias
                    if interval in STRATEGY_TIMEFRAMES['Reversion Soportes Resistencias']:
                        signal = self.check_reversion_soportes_resistencias_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 5: Stochastic Fibonacci
                    if interval in STRATEGY_TIMEFRAMES['Stochastic Fibonacci']:
                        signal = self.check_stochastic_fibonacci_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 6: VWAP Reversal
                    if interval in STRATEGY_TIMEFRAMES['VWAP Reversal']:
                        signal = self.check_vwap_reversal_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 7: Momentum Divergence
                    if interval in STRATEGY_TIMEFRAMES['Momentum Divergence']:
                        signal = self.check_momentum_divergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 8: ADX Power Trend
                    if interval in STRATEGY_TIMEFRAMES['ADX Power Trend']:
                        signal = self.check_adx_power_trend_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 9: MA Convergence Divergence
                    if interval in STRATEGY_TIMEFRAMES['MA Convergence Divergence']:
                        signal = self.check_ma_convergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 10: Volume Spike Momentum
                    if interval in STRATEGY_TIMEFRAMES['Volume Spike Momentum']:
                        signal = self.check_volume_spike_momentum_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 11: Stochastic Supertrend
                    if interval in STRATEGY_TIMEFRAMES['Stochastic Supertrend']:
                        signal = self.check_stochastic_supertrend_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 12: Support Resistance Bounce
                    if interval in STRATEGY_TIMEFRAMES['Support Resistance Bounce']:
                        signal = self.check_support_resistance_bounce_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 13: Whale DMI Combo
                    if interval in STRATEGY_TIMEFRAMES['Whale DMI Combo']:
                        signal = self.check_whale_dmi_combo_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 14: Parabolic SAR Momentum
                    if interval in STRATEGY_TIMEFRAMES['Parabolic SAR Momentum']:
                        signal = self.check_parabolic_sar_momentum_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 15: Multi-Timeframe Confirmation
                    if interval in STRATEGY_TIMEFRAMES['Multi-Timeframe Confirmation']:
                        signal = self.check_multi_timeframe_confirmation_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # NUEVAS ESTRATEGIAS
                    
                    # Estrategia 16: RSI Maverick DMI Divergence
                    if interval in STRATEGY_TIMEFRAMES['RSI Maverick DMI Divergence']:
                        signal = self.check_rsi_maverick_dmi_divergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 17: RSI Maverick Stochastic Crossover
                    if interval in STRATEGY_TIMEFRAMES['RSI Maverick Stochastic Crossover']:
                        signal = self.check_rsi_maverick_stochastic_crossover_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 18: RSI Maverick MACD Divergence
                    if interval in STRATEGY_TIMEFRAMES['RSI Maverick MACD Divergence']:
                        signal = self.check_rsi_maverick_macd_divergence_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 19: Whale RSI Maverick Combo
                    if interval in STRATEGY_TIMEFRAMES['Whale RSI Maverick Combo']:
                        signal = self.check_whale_rsi_maverick_combo_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                    # Estrategia 20: RSI Maverick Trend Reversal
                    if interval in STRATEGY_TIMEFRAMES['RSI Maverick Trend Reversal']:
                        signal = self.check_rsi_maverick_trend_reversal_signal(symbol, interval)
                        if signal:
                            all_signals.append(signal)
                    
                except Exception as e:
                    print(f"Error generando señales para {symbol} {interval}: {e}")
                    continue
        
        return all_signals

# Instancia global del indicador
indicator = TradingIndicator()

def send_telegram_alert(alert_data):
    """Enviar alerta por Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        filters_text = '\n'.join(['• ' + f for f in alert_data.get('filters', [])])
        
        recommendation = "Swing Trading"
        if alert_data['interval'] in ['12h', '1D']:
            recommendation = "Swing Trading / Inversión"
        elif alert_data['interval'] == '1W':
            recommendation = "Inversión Spot Largo Plazo"
        
        message = f"""
🚨 Alerta {alert_data['signal']} {alert_data['symbol']} en {alert_data['interval']} 🚨
Estrategia: {alert_data['strategy']}
Precio actual: {alert_data['current_price']:.6f} BTC | Entrada: {alert_data['entry']:.6f} BTC
Stop Loss: {alert_data['stop_loss']:.6f} BTC | Take Profit: {alert_data['take_profit'][0]:.6f} BTC
Filtros: 
{filters_text}
Recomendación: {recommendation}
"""
        
        if 'chart' in alert_data and alert_data['chart']:
            asyncio.run(bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=alert_data['chart'],
                caption=message
            ))
        else:
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message
            ))
        
        print(f"Alerta enviada a Telegram: {alert_data['symbol']} {alert_data['interval']} {alert_data['signal']} - {alert_data['strategy']}")
        
    except Exception as e:
        print(f"Error enviando alerta a Telegram: {e}")

def background_strategy_checker():
    """Verificador de estrategias en segundo plano"""
    print("Background strategy checker iniciado para PAXG/BTC SPOT...")
    
    interval_wait_times = {
        '4h': 660,
        '12h': 1380,
        '1D': 1380,
        '1W': 1920
    }
    
    last_checks = {interval: datetime.now() for interval in interval_wait_times.keys()}
    
    while True:
        try:
            current_time = datetime.now()
            
            for interval, wait_time in interval_wait_times.items():
                if (current_time - last_checks[interval]).seconds >= wait_time:
                    print(f"Verificando estrategias para intervalo {interval}...")
                    
                    if indicator.calculate_remaining_time(interval, current_time):
                        signals = indicator.generate_strategy_signals()
                        interval_signals = [s for s in signals if s['interval'] == interval]
                        
                        for signal in interval_signals:
                            signal_key = f"{signal['symbol']}_{signal['interval']}_{signal['strategy']}_{signal['signal']}"
                            
                            if signal_key not in indicator.strategy_signals:
                                if signal.get('send_telegram', True):
                                    send_telegram_alert(signal)
                                indicator.strategy_signals[signal_key] = current_time
                            else:
                                last_sent = indicator.strategy_signals[signal_key]
                                if (current_time - last_sent).seconds > 7200:
                                    if signal.get('send_telegram', True):
                                        send_telegram_alert(signal)
                                    indicator.strategy_signals[signal_key] = current_time
                    
                    # Verificar señales de salida
                    exit_signals = indicator.check_exit_signals()
                    for exit_signal in exit_signals:
                        exit_key = f"{exit_signal['symbol']}_{exit_signal['interval']}_{exit_signal['type']}"
                        
                        if exit_key not in indicator.strategy_signals:
                            send_telegram_alert(exit_signal)
                            indicator.strategy_signals[exit_key] = current_time
                    
                    last_checks[interval] = current_time
            
            time.sleep(10)
            
        except Exception as e:
            print(f"Error en background_strategy_checker: {e}")
            time.sleep(60)

# Iniciar verificador de estrategias
try:
    strategy_thread = Thread(target=background_strategy_checker, daemon=True)
    strategy_thread.start()
    print("Background strategy checker iniciado correctamente para PAXG/BTC SPOT")
except Exception as e:
    print(f"Error iniciando background strategy checker: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

@app.route('/api/signals')
def get_signals():
    """Endpoint para obtener señales de trading para PAXG/BTC"""
    try:
        symbol = request.args.get('symbol', 'PAXG-BTC')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 1))
        
        df = indicator.get_kucoin_data(symbol, interval, 100)
        if df is None or len(df) < 50:
            return jsonify({'error': 'No hay datos disponibles'}), 400
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        whale_data = indicator.calculate_whale_signals_improved(df)
        adx, plus_di, minus_di = indicator.calculate_adx(high, low, close, 14)
        rsi_traditional = indicator.calculate_rsi(close, 14)
        stochastic_data = indicator.calculate_stochastic_rsi(close)
        
        # Calcular RSI Maverick
        rsi_maverick = indicator.calculate_rsi_maverick(close)
        rsi_maverick_div = indicator.get_rsi_maverick_divergence_signals(symbol, interval)
        
        ma_9 = indicator.calculate_sma(close, 9)
        ma_21 = indicator.calculate_sma(close, 21)
        ma_50 = indicator.calculate_sma(close, 50)
        ma_200 = indicator.calculate_sma(close, 200)
        
        macd, macd_signal, macd_histogram = indicator.calculate_macd(close)
        
        bb_upper, bb_middle, bb_lower = indicator.calculate_bollinger_bands(close)
        
        volume_data = indicator.calculate_volume_anomaly(volume, close)
        
        ftm_data = indicator.calculate_trend_strength_maverick(close)
        
        supports, resistances = indicator.calculate_support_resistance_channels(high, low, close)
        
        current_price = float(close[-1])
        signal_type = 'NEUTRAL'
        signal_score = 0
        
        bullish_conditions = 0
        bearish_conditions = 0
        
        if rsi_traditional[-1] < 30:
            bullish_conditions += 1
        elif rsi_traditional[-1] > 70:
            bearish_conditions += 1
        
        if adx[-1] > 25 and plus_di[-1] > minus_di[-1]:
            bullish_conditions += 1
        elif adx[-1] > 25 and minus_di[-1] > plus_di[-1]:
            bearish_conditions += 1
        
        if macd[-1] > macd_signal[-1]:
            bullish_conditions += 1
        elif macd[-1] < macd_signal[-1]:
            bearish_conditions += 1
        
        signal_score = min(100, max(bullish_conditions, bearish_conditions) * 33.3)
        
        if bullish_conditions > bearish_conditions and signal_score >= 65:
            signal_type = 'COMPRA'
        elif bearish_conditions > bullish_conditions and signal_score >= 65:
            signal_type = 'VENTA'
        
        levels_data = indicator.calculate_optimal_entry_exit(
            df, signal_type, leverage, supports, resistances
        )
        
        response_data = {
            'symbol': symbol,
            'current_price': current_price,
            'signal': signal_type,
            'signal_score': float(signal_score),
            'entry': float(levels_data['entry']),
            'stop_loss': float(levels_data['stop_loss']),
            'take_profit': [float(tp) for tp in levels_data['take_profit'][:3]],
            'support_levels': [float(s) for s in supports[:3]],
            'resistance_levels': [float(r) for r in resistances[:3]],
            'atr': float(levels_data['atr']),
            'atr_percentage': float(levels_data['atr_percentage']),
            'volume': float(volume[-1]),
            'volume_ma': float(np.mean(volume[-20:])),
            'adx': float(adx[-1]),
            'plus_di': float(plus_di[-1]),
            'minus_di': float(minus_di[-1]),
            'whale_pump': float(whale_data['whale_pump'][-1]),
            'whale_dump': float(whale_data['whale_dump'][-1]),
            'rsi_traditional': float(rsi_traditional[-1]),
            'stoch_rsi': float(stochastic_data['stoch_rsi'][-1]),
            'stoch_k': float(stochastic_data['k_line'][-1]),
            'stoch_d': float(stochastic_data['d_line'][-1]),
            'rsi_maverick': float(rsi_maverick[-1]) if rsi_maverick else 0.5,
            'rsi_maverick_bullish_div': bool(rsi_maverick_div['bullish'][-1]) if rsi_maverick_div['bullish'] else False,
            'rsi_maverick_bearish_div': bool(rsi_maverick_div['bearish'][-1]) if rsi_maverick_div['bearish'] else False,
            'ma200_condition': 'above' if current_price > float(ma_200[-1]) else 'below',
            'data': [],
            'indicators': {
                'whale_pump': [float(x) for x in whale_data['whale_pump'][-50:]],
                'whale_dump': [float(x) for x in whale_data['whale_dump'][-50:]],
                'adx': [float(x) for x in adx[-50:]],
                'plus_di': [float(x) for x in plus_di[-50:]],
                'minus_di': [float(x) for x in minus_di[-50:]],
                'rsi_traditional': [float(x) for x in rsi_traditional[-50:]],
                'stoch_rsi': [float(x) for x in stochastic_data['stoch_rsi'][-50:]],
                'stoch_k': [float(x) for x in stochastic_data['k_line'][-50:]],
                'stoch_d': [float(x) for x in stochastic_data['d_line'][-50:]],
                'rsi_maverick': [float(x) for x in rsi_maverick[-50:]] if rsi_maverick else [0.5] * 50,
                'rsi_maverick_bullish_div': [bool(x) for x in rsi_maverick_div['bullish'][-50:]] if rsi_maverick_div['bullish'] else [False] * 50,
                'rsi_maverick_bearish_div': [bool(x) for x in rsi_maverick_div['bearish'][-50:]] if rsi_maverick_div['bearish'] else [False] * 50,
                'ma_9': [float(x) for x in ma_9[-50:]],
                'ma_21': [float(x) for x in ma_21[-50:]],
                'ma_50': [float(x) for x in ma_50[-50:]],
                'ma_200': [float(x) for x in ma_200[-50:]],
                'macd': [float(x) for x in macd[-50:]],
                'macd_signal': [float(x) for x in macd_signal[-50:]],
                'macd_histogram': [float(x) for x in macd_histogram[-50:]],
                'bb_upper': [float(x) for x in bb_upper[-50:]],
                'bb_middle': [float(x) for x in bb_middle[-50:]],
                'bb_lower': [float(x) for x in bb_lower[-50:]],
                'volume_anomaly': [bool(x) for x in volume_data['volume_anomaly'][-50:]],
                'volume_clusters': [bool(x) for x in volume_data['volume_clusters'][-50:]],
                'volume_ratio': [float(x) for x in volume_data['volume_ratio'][-50:]],
                'volume_ma': [float(x) for x in volume_data['volume_ma'][-50:]],
                'volume_signal': volume_data['volume_signal'][-50:],
                'trend_strength': [float(x) for x in ftm_data['trend_strength'][-50:]],
                'bb_width': [float(x) for x in ftm_data['bb_width'][-50:]],
                'no_trade_zones': [bool(x) for x in ftm_data['no_trade_zones'][-50:]],
                'strength_signals': ftm_data['strength_signals'][-50:],
                'high_zone_threshold': float(ftm_data['high_zone_threshold']),
                'colors': ftm_data['colors'][-50:]
            }
        }
        
        for i in range(max(0, len(df) - 50), len(df)):
            candle = df.iloc[i]
            response_data['data'].append({
                'timestamp': candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error en /api/signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/api/strategy_signals')
def get_strategy_signals():
    """Endpoint para obtener señales de las estrategias"""
    try:
        signals = indicator.generate_strategy_signals()
        return jsonify({'signals': signals})
        
    except Exception as e:
        print(f"Error en /api/strategy_signals: {e}")
        return jsonify({'signals': []})

@app.route('/api/bolivia_time')
def get_bolivia_time():
    """Endpoint para obtener la hora actual de Bolivia"""
    bolivia_tz = pytz.timezone('America/La_Paz')
    current_time = datetime.now(bolivia_tz)
    return jsonify({
        'time': current_time.strftime('%H:%M:%S'),
        'date': current_time.strftime('%Y-%m-%d'),
        'timezone': 'America/La_Paz'
    })

@app.route('/api/generate_report')
def generate_report():
    """Generar reporte técnico completo"""
    try:
        symbol = request.args.get('symbol', 'PAXG-BTC')
        interval = request.args.get('interval', '4h')
        leverage = int(request.args.get('leverage', 1))
        
        signal_data_response = get_signals()
        signal_data = signal_data_response.get_json()
        
        if 'error' in signal_data:
            return jsonify({'error': 'No hay datos para generar el reporte'}), 400
        
        fig = plt.figure(figsize=(14, 20))
        
        ax1 = plt.subplot(11, 1, 1)
        if 'data' in signal_data and signal_data['data']:
            dates = [datetime.strptime(d['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(d['timestamp'], str) 
                    else d['timestamp'] for d in signal_data['data']]
            opens = [d['open'] for d in signal_data['data']]
            highs = [d['high'] for d in signal_data['data']]
            lows = [d['low'] for d in signal_data['data']]
            closes = [d['close'] for d in signal_data['data']]
            
            dates_matplotlib = mdates.date2num(dates)
            
            for i in range(len(dates_matplotlib)):
                color = 'green' if closes[i] >= opens[i] else 'red'
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [lows[i], highs[i]], color='black', linewidth=1)
                ax1.plot([dates_matplotlib[i], dates_matplotlib[i]], [opens[i], closes[i]], color=color, linewidth=3)
            
            ax1.axhline(y=signal_data['entry'], color='blue', linestyle='--', alpha=0.7, label='Entrada')
            ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.7, label='Stop Loss')
            for i, tp in enumerate(signal_data['take_profit'][:3]):
                ax1.axhline(y=tp, color='green', linestyle='--', alpha=0.7, label=f'TP{i+1}')
        
        ax1.set_title(f'{symbol} - Análisis Técnico Completo ({interval})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio (BTC)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
        
        ax2 = plt.subplot(11, 1, 2, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_maverick' in signal_data['indicators']:
            rsi_dates = dates_matplotlib[-len(signal_data['indicators']['rsi_maverick']):]
            ax2.plot(rsi_dates, signal_data['indicators']['rsi_maverick'], 
                    'blue', linewidth=2, label='RSI Maverick')
            
            # Marcar divergencias
            if 'rsi_maverick_bullish_div' in signal_data['indicators']:
                for i in range(len(rsi_dates)):
                    if signal_data['indicators']['rsi_maverick_bullish_div'][i]:
                        ax2.scatter(rsi_dates[i], signal_data['indicators']['rsi_maverick'][i], 
                                  color='green', s=30, marker='^', label='Div Alcista' if i == 0 else "")
                    if signal_data['indicators']['rsi_maverick_bearish_div'][i]:
                        ax2.scatter(rsi_dates[i], signal_data['indicators']['rsi_maverick'][i], 
                                  color='red', s=30, marker='v', label='Div Bajista' if i == 0 else "")
            
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
            ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
            ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)
        ax2.set_ylabel('RSI Maverick')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(11, 1, 3, sharex=ax1)
        if 'indicators' in signal_data and 'stoch_rsi' in signal_data['indicators']:
            stoch_dates = dates_matplotlib[-len(signal_data['indicators']['stoch_rsi']):]
            ax3.plot(stoch_dates, signal_data['indicators']['stoch_rsi'], 
                    'blue', linewidth=1, label='RSI Estocástico')
            ax3.plot(stoch_dates, signal_data['indicators']['stoch_k'], 
                    'green', linewidth=1, label='%K')
            ax3.plot(stoch_dates, signal_data['indicators']['stoch_d'], 
                    'red', linewidth=1, label='%D')
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(y=20, color='green', linestyle='--', alpha=0.3)
        ax3.set_ylabel('RSI Estocástico')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(11, 1, 4, sharex=ax1)
        if 'indicators' in signal_data and 'adx' in signal_data['indicators']:
            adx_dates = dates_matplotlib[-len(signal_data['indicators']['adx']):]
            ax4.plot(adx_dates, signal_data['indicators']['adx'], 
                    'black', linewidth=2, label='ADX')
            ax4.plot(adx_dates, signal_data['indicators']['plus_di'], 
                    'green', linewidth=1, label='+DI')
            ax4.plot(adx_dates, signal_data['indicators']['minus_di'], 
                    'red', linewidth=1, label='-DI')
            ax4.axhline(y=25, color='yellow', linestyle='--', alpha=0.7, label='Umbral 25')
        ax4.set_ylabel('ADX/DMI')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(11, 1, 5, sharex=ax1)
        if 'indicators' in signal_data and 'rsi_traditional' in signal_data['indicators']:
            rsi_trad_dates = dates_matplotlib[-len(signal_data['indicators']['rsi_traditional']):]
            ax5.plot(rsi_trad_dates, signal_data['indicators']['rsi_traditional'], 
                    'cyan', linewidth=2, label='RSI Tradicional')
            ax5.axhline(y=80, color='red', linestyle='--', alpha=0.3)
            ax5.axhline(y=20, color='green', linestyle='--', alpha=0.3)
            ax5.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
        ax5.set_ylabel('RSI Tradicional')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(11, 1, 6, sharex=ax1)
        if 'indicators' in signal_data and 'macd' in signal_data['indicators']:
            macd_dates = dates_matplotlib[-len(signal_data['indicators']['macd']):]
            ax6.plot(macd_dates, signal_data['indicators']['macd'], 
                    'blue', linewidth=1, label='MACD')
            ax6.plot(macd_dates, signal_data['indicators']['macd_signal'], 
                    'red', linewidth=1, label='Señal')
            
            colors = ['green' if x > 0 else 'red' for x in signal_data['indicators']['macd_histogram']]
            ax6.bar(macd_dates, signal_data['indicators']['macd_histogram'], 
                   color=colors, alpha=0.6, label='Histograma')
            
            ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax6.set_ylabel('MACD')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        ax7 = plt.subplot(11, 1, 7, sharex=ax1)
        if 'indicators' in signal_data and 'whale_pump' in signal_data['indicators']:
            whale_dates = dates_matplotlib[-len(signal_data['indicators']['whale_pump']):]
            ax7.bar(whale_dates, signal_data['indicators']['whale_pump'], 
                   color='green', alpha=0.7, label='Ballenas Compradoras')
            ax7.bar(whale_dates, [-x for x in signal_data['indicators']['whale_dump']], 
                   color='red', alpha=0.7, label='Ballenas Vendedoras')
        ax7.set_ylabel('Fuerza Ballenas')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        ax8 = plt.subplot(11, 1, 8, sharex=ax1)
        if 'indicators' in signal_data and 'volume_ratio' in signal_data['indicators']:
            volume_dates = dates_matplotlib[-len(signal_data['indicators']['volume_ratio']):]
            
            colors = []
            volume_signal = signal_data['indicators'].get('volume_signal', ['NEUTRAL'] * 50)
            for i, signal in enumerate(volume_signal[-50:]):
                if signal == 'COMPRA':
                    colors.append('green')
                elif signal == 'VENTA':
                    colors.append('red')
                else:
                    colors.append('gray')
            
            volumes = [d['volume'] for d in signal_data['data'][-50:]]
            ax8.bar(volume_dates, volumes, color=colors, alpha=0.6, label='Volumen')
            
            ax8.plot(volume_dates, signal_data['indicators']['volume_ma'][-50:], 
                    'yellow', linewidth=1, label='MA Volumen')
        ax8.set_ylabel('Volumen')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        ax9 = plt.subplot(11, 1, 9, sharex=ax1)
        if 'indicators' in signal_data and 'trend_strength' in signal_data['indicators']:
            trend_dates = dates_matplotlib[-len(signal_data['indicators']['trend_strength']):]
            trend_strength = signal_data['indicators']['trend_strength'][-50:]
            colors = signal_data['indicators']['colors'][-50:]
            
            for i in range(len(trend_dates)):
                ax9.bar(trend_dates[i], trend_strength[i], color=colors[i], alpha=0.7, width=0.8)
            
            if 'high_zone_threshold' in signal_data['indicators']:
                threshold = signal_data['indicators']['high_zone_threshold']
                ax9.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, 
                           label=f'Umbral Alto ({threshold:.1f}%)')
                ax9.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.7)
        ax9.set_ylabel('Fuerza Tendencia %')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        ax10 = plt.subplot(11, 1, 10, sharex=ax1)
        if 'indicators' in signal_data and 'bb_upper' in signal_data['indicators']:
            bb_dates = dates_matplotlib[-len(signal_data['indicators']['bb_upper']):]
            ax10.plot(bb_dates, closes[-50:], 'blue', linewidth=1, label='Precio')
            ax10.plot(bb_dates, signal_data['indicators']['bb_upper'][-50:], 
                    'orange', alpha=0.7, linewidth=1, label='BB Superior')
            ax10.plot(bb_dates, signal_data['indicators']['bb_middle'][-50:], 
                    'orange', alpha=0.5, linewidth=1, label='BB Media')
            ax10.plot(bb_dates, signal_data['indicators']['bb_lower'][-50:], 
                    'orange', alpha=0.7, linewidth=1, label='BB Inferior')
            ax10.fill_between(bb_dates, signal_data['indicators']['bb_lower'][-50:], 
                           signal_data['indicators']['bb_upper'][-50:], 
                           color='orange', alpha=0.1)
        ax10.set_ylabel('Bollinger Bands')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        ax11 = plt.subplot(11, 1, 11)
        ax11.axis('off')
        
        signal_info = f"""
        SEÑAL: {signal_data['signal']}
        SCORE: {signal_data['signal_score']:.1f}%
        
        PRECIO ACTUAL: {signal_data['current_price']:.6f} BTC
        ENTRADA: {signal_data['entry']:.6f} BTC
        STOP LOSS: {signal_data['stop_loss']:.6f} BTC
        TAKE PROFIT: {signal_data['take_profit'][0]:.6f} BTC
        
        ATR: {signal_data['atr']:.6f} ({signal_data['atr_percentage']*100:.1f}%)
        
        INDICADORES:
        RSI Maverick: {signal_data['rsi_maverick']:.3f}
        Divergencia: {"Alcista" if signal_data['rsi_maverick_bullish_div'] else "Bajista" if signal_data['rsi_maverick_bearish_div'] else "Ninguna"}
        RSI Tradicional: {signal_data['rsi_traditional']:.1f}
        RSI Estocástico: {signal_data['stoch_rsi']:.1f}
        ADX: {signal_data['adx']:.1f}
        +DI: {signal_data['plus_di']:.1f}
        -DI: {signal_data['minus_di']:.1f}
        """
        
        ax11.text(0.1, 0.9, signal_info, transform=ax11.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'report_{symbol}_{interval}_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
    except Exception as e:
        print(f"Error generando reporte: {e}")
        return jsonify({'error': 'Error generando reporte'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({'error': 'Servicio no disponible temporalmente'}), 503

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
