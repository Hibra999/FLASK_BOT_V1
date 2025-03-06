from flask import Flask, render_template, jsonify, request, Response
import ccxt
import pandas as pd
import numpy as np
import time
import math
import logging
import warnings
from datetime import datetime, timedelta
import threading
import json
import os
import io
import csv

# Ensure pandas_ta is installed
try:
    import pandas_ta as pta
except ImportError:
    import pip
    pip.main(['install', 'pandas_ta'])
    import pandas_ta as pta

# Import the Telegram bot (make sure to have python-telegram-bot installed)
try:
    from telegram_bot import TelegramBot
except ImportError:
    import pip
    pip.main(['install', 'python-telegram-bot==13.7'])  # Use older version to avoid conflicts
    from telegram_bot import TelegramBot

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration storage
CONFIG_FILE = 'bot_config.json'

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

# -------------------------------
# TRADING BOT CLASS
# -------------------------------
class TradingBot:
    def __init__(self):
        # Load config
        self.config = load_config()
        
        # Configuration
        self.modo_simulacion = self.config.get('simulation_mode', True)
        self.precision = 5
        self.initial_capital = self.config.get('initial_capital', 2000.0)
        self.saldo_money = self.initial_capital
        self.saldo_monedas = 0.0
        self.status = ""
        self.ultimo_precio = 1.0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.df_simulacion = pd.DataFrame([])
        self.df_order = pd.DataFrame([])
        self.ultimo_procesado = 0
        self.last_buy_price = None
        self.operations_log = []
        self.start_time = time.time()
        self.operation_interval = self.config.get('operation_interval', 3600)  # Default: 1 hour
        self.last_operation_time = time.time()
        self.next_operation_time = self.last_operation_time + self.operation_interval
        self.max_candles = 25000  # Increased candle limit
        
        # SMA Parameters
        self.sma_short_period = self.config.get('sma_short_period', 10)
        self.sma_long_period = self.config.get('sma_long_period', 50)
        self.strategy = self.config.get('strategy', 'NFI')  # NFI or SMA
        
        # Backtesting results
        self.backtesting_results = {}
        self.multi_token_results = []
        
        # API Credentials (with defaults from config)
        self.api_key = self.config.get('api_key', "GpjnxAeNtT")
        self.api_secret = self.config.get('api_secret', '2da27dea7a224f1304534dc267841db1')
        
        # Nuevas variables para gestionar estado
        self.is_backtesting = True  # Flag para distinguir backtesting vs operaciones reales
        self.state_file = 'bot_state.json'
        self.recovering = False
        self.load_state()  # Cargar estado previo si existe
        
        # Telegram bot setup
        self.telegram_config = self.config.get('telegram', {
            'enabled': False,
            'token': '',
            'chat_id': ''
        })
        self.telegram_bot = TelegramBot(
            token=self.telegram_config.get('token'),
            chat_id=self.telegram_config.get('chat_id'),
            enabled=self.telegram_config.get('enabled', False),
            trading_bot=self  # Referencia a la instancia del bot para comandos
        )
        
        # CCXT exchange setup
        self.setup_exchange()
        
        # NFI strategy parameters
        self.NFI_PARAMS = self.config.get('nfi_params', {
            "buy_condition_1_enable": True,
            "buy_condition_2_enable": True,
            "buy_condition_3_enable": True,
            "buy_condition_4_enable": True,
            "buy_condition_5_enable": True,
            "buy_condition_6_enable": True,
            "buy_condition_7_enable": True,
            "buy_condition_8_enable": True,
            "buy_condition_9_enable": True,
            "buy_condition_10_enable": True,
            "buy_condition_11_enable": True,
            "buy_condition_12_enable": True,
            "buy_condition_13_enable": True,
            "buy_condition_14_enable": True,
            "buy_condition_15_enable": True,
            "buy_condition_16_enable": True,
            "buy_condition_17_enable": True,
            "buy_condition_18_enable": True,
            "buy_condition_19_enable": True,
            "buy_condition_20_enable": True,
            "buy_condition_21_enable": True,
            "sell_condition_1_enable": True,
            "sell_condition_2_enable": True,
            "sell_condition_3_enable": True,
            "sell_condition_4_enable": True,
            "sell_condition_5_enable": True,
            "sell_condition_6_enable": True,
            "sell_condition_7_enable": True,
            "sell_condition_8_enable": True,
            "ewo_low": -8.5,
            "ewo_high": 4.3,
            "low_offset_sma": 0.955,
            "low_offset_ema": 0.929,
            "low_offset_trima": 0.949,
            "low_offset_t3": 0.975,
            "low_offset_kama": 0.972,
            "high_offset_ema": 1.047,
            "buy_rsi_18": 26,
            "buy_chop_min_19": 58.2,
            "buy_rsi_1h_min_19": 65.3,
            "buy_rsi_15": 30,
            "buy_mfi_1": 26.0,
            "buy_rsi_1": 36.0,
            "buy_mfi_9": 30.0,
            "buy_mfi_11": 38.0,
            "sell_rsi_bb_2": 81.0,
            "sell_rsi_main_3": 82.0,
            "sell_dual_rsi_rsi_4": 73.4,
            "sell_dual_rsi_rsi_1h_4": 79.6,
            "sell_ema_relative_5": 0.024,
            "sell_rsi_under_6": 79.0,
            "sell_rsi_1h_7": 81.7,
            "sell_bb_relative_8": 1.1
        })
        
        # Initialize tokens
        self.all_tokens = self.obtener_tokens_ccxt()
        self.tokens_disponibles = self.all_tokens.copy()
        self.current_token = self.tokens_disponibles[0] if self.tokens_disponibles else ""
        
        # Start background thread for updates
        self.running = True
        self.update_thread = threading.Thread(target=self.background_update, daemon=True)
        self.update_thread.start()
    
    def save_state(self):
        """Guarda el estado del bot para evitar pérdida de datos en caso de apagado inesperado"""
        state = {
            'mode': 'Simulación' if self.modo_simulacion else 'Real',
            'token': self.current_token,
            'status': self.status,
            'saldo_money': self.saldo_money,
            'saldo_monedas': self.saldo_monedas,
            'last_buy_price': self.last_buy_price,
            'last_operation_time': self.last_operation_time,
            'next_operation_time': self.next_operation_time,
            'operations': self.operations_log[-10:],  # Guardar últimas 10 operaciones
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'strategy': self.strategy,
            'sma_short_period': self.sma_short_period,
            'sma_long_period': self.sma_long_period
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4, default=str)
            return True
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
            return False

    def load_state(self):
        """Carga el estado previo del bot si existe"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Si estábamos en modo real y con posición abierta
                if state.get('mode') == 'Real' and state.get('status') == 'compra':
                    self.recovering = True
                    self.modo_simulacion = False
                    self.current_token = state.get('token', self.current_token)
                    self.status = state.get('status', '')
                    self.saldo_money = state.get('saldo_money', self.initial_capital)
                    self.saldo_monedas = state.get('saldo_monedas', 0.0)
                    self.last_buy_price = state.get('last_buy_price')
                    self.operations_log = state.get('operations', [])
                    self.wins = state.get('wins', 0)
                    self.losses = state.get('losses', 0)
                    self.draws = state.get('draws', 0)
                    self.strategy = state.get('strategy', 'NFI')
                    self.sma_short_period = state.get('sma_short_period', 10)
                    self.sma_long_period = state.get('sma_long_period', 50)
                    
                    logger.info(f"Recuperado estado de sesión anterior, posición abierta: {self.status}")
                    
                    # Notificar por Telegram que estamos recuperando de un apagado
                    if self.telegram_bot.enabled:
                        self.telegram_bot.send_message("⚠️ *Alerta de Recuperación*\nBot reiniciado después de un apagado inesperado. Recuperando estado anterior.")
                
                return True
        except Exception as e:
            logger.error(f"Error cargando estado: {e}")
            return False

    def safe_stop(self):
        """Detiene el bot de forma segura, asegurando que se completen operaciones"""
        # Guardar estado actual primero
        self.save_state()
        
        # Verificar si hay una posición abierta en modo real
        if not self.modo_simulacion and self.status == 'compra' and self.saldo_monedas > 0:
            logger.warning("¡Deteniendo bot con posición abierta en modo real!")
            # Notificar al usuario por Telegram
            if self.telegram_bot.enabled:
                self.telegram_bot.send_message("⚠️ *Advertencia: Bot detenido con posición abierta*\nEl bot se detuvo mientras una posición aún estaba abierta. Es posible que necesite cerrarla manualmente.")
        
        self.running = False
        
        # Esperar a que el hilo de fondo se detenga
        if hasattr(self, 'update_thread') and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        logger.info("Bot detenido de forma segura")
        return True
    
    def setup_exchange(self):
        """Setup CCXT exchange with current credentials"""
        try:
            self.exchange_ccxt = ccxt.bitso({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            logger.info("Exchange setup complete")
            return True
        except Exception as e:
            logger.error(f"Error setting up exchange: {e}")
            return False
    
    def update_api_credentials(self, api_key, api_secret):
        """Update API credentials and reinitialize exchange"""
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Update config
            self.config['api_key'] = api_key
            self.config['api_secret'] = api_secret
            save_config(self.config)
            
            # Reinitialize exchange
            return self.setup_exchange()
        except Exception as e:
            logger.error(f"Error updating API credentials: {e}")
            return False
    
    def update_telegram_config(self, enabled, token, chat_id):
        """Update Telegram bot configuration"""
        try:
            self.telegram_config = {
                'enabled': enabled,
                'token': token,
                'chat_id': chat_id
            }
            
            # Update config file
            self.config['telegram'] = self.telegram_config
            save_config(self.config)
            
            # Reinitialize Telegram bot
            self.telegram_bot = TelegramBot(
                token=token,
                chat_id=chat_id,
                enabled=enabled,
                trading_bot=self
            )
            
            return True
        except Exception as e:
            logger.error(f"Error updating Telegram config: {e}")
            return False
    
    # ------------------------------
    # UTILITY FUNCTIONS
    # ------------------------------
    def truncate(self, number, decimals=0):
        if decimals == 0:
            return math.trunc(number)
        factor = 10.0 ** decimals
        return math.trunc(number * factor) / factor
    
    def reset_vars(self):
        self.saldo_money = self.initial_capital
        self.saldo_monedas = 0.0
        self.status = ""
        self.df_simulacion = pd.DataFrame([])
        self.df_order = pd.DataFrame([])
        self.ultimo_procesado = 0
        self.ultimo_precio = 1.0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.operations_log = []
        self.last_buy_price = None
    
    def typical_price(self, df):
        return (df['High'] + df['Low'] + df['Close']) / 3.0
    
    def bollinger_bands(self, series, window=20, stds=2):
        mid = series.rolling(window).mean()
        stdev = series.rolling(window).std()
        upper = mid + stds * stdev
        lower = mid - stds * stdev
        return pd.DataFrame({'bb_lowerband': lower, 'bb_middleband': mid, 'bb_upperband': upper})
    
    def chopiness(self, df, window=14):
        high_low = (df['High'] - df['Low']).rolling(window).sum()
        highest = df['High'].rolling(window).max()
        lowest = df['Low'].rolling(window).min()
        rangema = highest - lowest
        chop = 100.0 * np.log10((high_low) / (rangema + 1e-9)) / np.log10(window)
        return chop
    
    # ------------------------------
    # SMA STRATEGY FUNCTIONS
    # ------------------------------
    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    def implement_sma_strategy(self, df, short_period, long_period):
        """Implement SMA crossover strategy"""
        # Calculate short and long SMAs
        df['SMA_Short'] = self.calculate_sma(df['Close'], short_period)
        df['SMA_Long'] = self.calculate_sma(df['Close'], long_period)
        
        # Generate buy/sell signals
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1  # Buy signal
        df.loc[df['SMA_Short'] < df['SMA_Long'], 'Signal'] = -1  # Sell signal
        
        return df

    def run_sma_backtest(self, token, limit, capital, sma_short, sma_long):
        """Run SMA strategy backtest on a single token"""
        # Reset variables for this backtest
        self.reset_vars()
        self.is_backtesting = True
        self.saldo_money = capital
        
        # Load historical data
        self.cargar_ordenes(token, limit)
        
        if self.df_order.empty:
            return {
                'success': False,
                'error': 'No data available for this token'
            }
            
        # Apply SMA strategy
        df = self.df_order.copy()
        df = self.implement_sma_strategy(df, sma_short, sma_long)
        
        # Run backtest with SMA strategy
        result = self.backtest_sma_strategy(token, df, capital, sma_short, sma_long)
        self.backtesting_results[token] = result
        return result

    def run_multi_token_backtest(self, tokens, limit, capital, sma_short, sma_long):
        """Run backtests on multiple tokens and compare results"""
        results = []
        
        for token in tokens:
            # Run backtest for this token
            result = self.run_sma_backtest(token, limit, capital, sma_short, sma_long)
            
            if result.get('success', True):  # If not explicitly failed
                results.append(result)
        
        # Sort results by profit percentage (descending)
        results.sort(key=lambda x: x.get('profit_percent', -float('inf')), reverse=True)
        
        self.multi_token_results = results
        return results

    def backtest_sma_strategy(self, token, df, initial_capital, sma_short, sma_long):
        """Backtest SMA strategy on a token"""
        # Initialize variables
        money = initial_capital
        coins = 0
        trades = []
        positions = []
        last_signal = 0
        start_time = df.iloc[0]['Date']
        end_time = df.iloc[-1]['Date']
        
        for idx, row in df.iterrows():
            current_signal = row['Signal']
            close_price = row['Close']
            date = row['Date']
            
            # Buy signal (1) when previous signal was sell (-1) or no position (0)
            if current_signal == 1 and last_signal <= 0 and money > 0:
                # Ensure minimum investment of 20 USD
                amount_to_invest = max(20, money)
                amount_to_invest = min(amount_to_invest, money)  # Don't invest more than available
                
                coins_bought = self.truncate((amount_to_invest / close_price) * 0.99, self.precision)
                cost = coins_bought * close_price
                
                if cost > 0 and coins_bought > 0:
                    money -= cost
                    coins += coins_bought
                    
                    trade = {
                        'date': date,
                        'type': 'buy',
                        'price': close_price,
                        'amount': coins_bought,
                        'cost': cost,
                        'balance': money + (coins * close_price)
                    }
                    trades.append(trade)
                    positions.append({
                        'open_date': date,
                        'close_date': None,
                        'open_price': close_price,
                        'close_price': None,
                        'amount': coins_bought,
                        'profit': None,
                        'profit_percent': None,
                        'duration': None
                    })
            
            # Sell signal (-1) when previous signal was buy (1)
            elif current_signal == -1 and last_signal == 1 and coins > 0:
                revenue = coins * close_price
                profit = revenue - trades[-1]['cost'] if trades else 0
                profit_percent = (profit / trades[-1]['cost']) * 100 if trades else 0
                
                money += revenue
                
                # Update the last open position
                if positions and positions[-1]['close_date'] is None:
                    positions[-1]['close_date'] = date
                    positions[-1]['close_price'] = close_price
                    positions[-1]['profit'] = profit
                    positions[-1]['profit_percent'] = profit_percent
                    positions[-1]['duration'] = (date - positions[-1]['open_date']).total_seconds() / 3600  # Hours
                
                trade = {
                    'date': date,
                    'type': 'sell',
                    'price': close_price,
                    'amount': coins,
                    'revenue': revenue,
                    'profit': profit,
                    'profit_percent': profit_percent,
                    'balance': money
                }
                trades.append(trade)
                coins = 0
            
            last_signal = current_signal
        
        # Force sell at the end if still holding
        if coins > 0:
            close_price = df.iloc[-1]['Close']
            revenue = coins * close_price
            profit = revenue - trades[-1]['cost'] if trades else 0
            profit_percent = (profit / trades[-1]['cost']) * 100 if trades else 0
            
            money += revenue
            
            # Update the last open position
            if positions and positions[-1]['close_date'] is None:
                positions[-1]['close_date'] = end_time
                positions[-1]['close_price'] = close_price
                positions[-1]['profit'] = profit
                positions[-1]['profit_percent'] = profit_percent
                positions[-1]['duration'] = (end_time - positions[-1]['open_date']).total_seconds() / 3600  # Hours
            
            trade = {
                'date': end_time,
                'type': 'sell',
                'price': close_price,
                'amount': coins,
                'revenue': revenue,
                'profit': profit,
                'profit_percent': profit_percent,
                'balance': money
            }
            trades.append(trade)
        
        # Calculate results
        total_profit = money - initial_capital
        profit_percent = (total_profit / initial_capital) * 100
        
        # Calculate buy & hold profit for comparison
        first_price = df.iloc[0]['Close']
        last_price = df.iloc[-1]['Close']
        buy_hold_profit_percent = ((last_price - first_price) / first_price) * 100
        
        # Calculate win rate
        wins = sum(1 for p in positions if p.get('profit', 0) > 0)
        total_trades = len(positions)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate drawdown
        balances = [t.get('balance', initial_capital) for t in trades]
        peak_balance = initial_capital
        max_drawdown = 0
        
        for balance in balances:
            peak_balance = max(peak_balance, balance)
            if peak_balance > 0:  # Avoid division by zero
                drawdown = (peak_balance - balance) / peak_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate additional metrics
        avg_profit = sum(p.get('profit', 0) for p in positions) / total_trades if total_trades > 0 else 0
        avg_profit_percent = sum(p.get('profit_percent', 0) for p in positions) / total_trades if total_trades > 0 else 0
        avg_duration = sum(p.get('duration', 0) for p in positions) / total_trades if total_trades > 0 else 0
        
        return {
            'token': token,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'initial_capital': initial_capital,
            'final_balance': money,
            'profit': total_profit,
            'profit_percent': profit_percent,
            'buy_hold_profit_percent': buy_hold_profit_percent,
            'trades': trades,
            'positions': positions,
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_profit_percent': avg_profit_percent,
            'avg_duration': avg_duration,
            'max_drawdown': max_drawdown,
            'start_date': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_days': (end_time - start_time).total_seconds() / (24 * 3600)
        }

    def get_backtest_chart_data(self, token, sma_short, sma_long):
        """Get chart data for backtesting visualization"""
        if self.df_order.empty:
            return None
        
        # Apply SMA strategy
        df = self.df_order.copy()
        df = self.implement_sma_strategy(df, sma_short, sma_long)
        
        # Format data for chart
        chart_data = {}
        
        # Convert timestamps to milliseconds for JS
        df['timestamp'] = df['Date'].astype(int) // 10**6
        
        # OHLC data
        chart_data['candles'] = df[['timestamp', 'Open', 'High', 'Low', 'Close']].values.tolist()
        
        # SMA data
        df_sma_short = df.dropna(subset=['SMA_Short'])
        df_sma_long = df.dropna(subset=['SMA_Long'])
        
        chart_data['sma_short'] = df_sma_short[['timestamp', 'SMA_Short']].values.tolist()
        chart_data['sma_long'] = df_sma_long[['timestamp', 'SMA_Long']].values.tolist()
        
        # Signals data
        buy_signals = df[(df['Signal'] == 1) & (df['Signal'].shift(1) != 1)].copy()
        sell_signals = df[(df['Signal'] == -1) & (df['Signal'].shift(1) != -1)].copy()
        
        chart_data['buy_signals'] = buy_signals[['timestamp', 'Close']].values.tolist()
        chart_data['sell_signals'] = sell_signals[['timestamp', 'Close']].values.tolist()
        
        return chart_data
        
    # ------------------------------
    # TOKEN AND BALANCE FUNCTIONS
    # ------------------------------
    def obtener_tokens_ccxt(self):
        try:
            self.exchange_ccxt.load_markets()
            all_markets = list(self.exchange_ccxt.markets.keys())
            tokens_filtrados = [m for m in all_markets if m.endswith('/USD')]
            tokens_filtrados.sort()
            return tokens_filtrados
        except Exception as e:
            logger.error(f"Error obteniendo tokens: {e}")
            return []
    
    def get_real_balance(self):
        try:
            balance = self.exchange_ccxt.fetch_balance()
            usd_balance = None
            if 'USD' in balance:
                usd_balance = balance['USD'].get('total', None)
            if usd_balance is None:
                for key, val in balance.items():
                    if key == 'info':
                        continue
                    if 'USD' in key:
                        usd_balance = val.get('total', 0)
                        break
            return usd_balance
        except Exception as e:
            logger.error(f"Error fetching real balance: {e}")
            return None
    
    def get_all_balances(self):
        """Get all account balances"""
        try:
            balance = self.exchange_ccxt.fetch_balance()
            result = {}
            
            # Filter only relevant balances (non-zero)
            for currency, data in balance.items():
                if currency == 'info' or currency == 'free' or currency == 'used' or currency == 'total':
                    continue
                
                total = data.get('total', 0)
                if total > 0:
                    result[currency] = {
                        'free': data.get('free', 0),
                        'used': data.get('used', 0),
                        'total': total
                    }
            
            return result
        except Exception as e:
            logger.error(f"Error fetching all balances: {e}")
            return {}
    
    # ------------------------------
    # DATA LOADING AND UPDATING
    # ------------------------------
    def cargar_ordenes(self, token, limit_candles):
        """Load historical data with better error handling and progress logging"""
        try:
            # Enforce maximum limit
            limit_candles = min(limit_candles, self.max_candles)
            
            # For large requests, we need to fetch in chunks
            max_chunk_size = 1000  # CCXT typical limit per request
            chunks_needed = math.ceil(limit_candles / max_chunk_size)
            
            if chunks_needed > 1:
                logger.info(f"Fetching {limit_candles} candles in {chunks_needed} chunks")
                all_ohlcv = []
                
                # Start from the oldest data
                since = int((datetime.now() - timedelta(hours=limit_candles)).timestamp() * 1000)
                
                for i in range(chunks_needed):
                    logger.info(f"Fetching chunk {i+1}/{chunks_needed}")
                    chunk = self.exchange_ccxt.fetch_ohlcv(
                        symbol=token, 
                        timeframe='1h', 
                        since=since, 
                        limit=max_chunk_size
                    )
                    
                    if not chunk:
                        break
                        
                    all_ohlcv.extend(chunk)
                    
                    # Update since for next chunk
                    last_timestamp = chunk[-1][0]
                    since = last_timestamp + 1
                    
                    # Avoid rate limiting
                    time.sleep(1)
                
                # Limit to the requested number of candles
                ohlcv = all_ohlcv[-limit_candles:] if all_ohlcv else []
            else:
                # For smaller requests, fetch directly
                ohlcv = self.exchange_ccxt.fetch_ohlcv(symbol=token, timeframe='1h', limit=limit_candles)
            
            if not ohlcv:
                logger.error(f"No OHLCV data received for {token}")
                return
            
            # Process the data
            df_temp = pd.DataFrame(ohlcv, columns=['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_temp['Date'] = pd.to_datetime(df_temp['OpenTime'], unit='ms')
            df_temp.sort_values('Date', inplace=True)
            df_temp.reset_index(drop=True, inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_temp[col] = df_temp[col].astype(float)
            
            # Calculate indicators
            logger.info(f"Calculating indicators for {len(df_temp)} candles")
            df_temp['ema_20'] = df_temp.ta.ema(length=20, close='Close')
            df_temp['sma_20_nfi'] = df_temp.ta.sma(length=20, close='Close')
            df_temp['trima_20'] = df_temp.ta.trima(length=20, close='Close')
            df_temp['t3_20'] = df_temp.ta.t3(length=20, close='Close')
            df_temp['kama_20'] = df_temp.ta.kama(length=20, close='Close')
            df_temp['rsi'] = df_temp.ta.rsi(length=14, close='Close')
            df_temp['mfi'] = df_temp.ta.mfi(high=df_temp['High'], low=df_temp['Low'], close=df_temp['Close'], volume=df_temp['Volume'], length=14)
            df_temp['ema_5'] = df_temp.ta.ema(length=5, close='Close')
            df_temp['ema_35'] = df_temp.ta.ema(length=35, close='Close')
            df_temp['ewo'] = (df_temp['ema_5'] - df_temp['ema_35']) / df_temp['Close'] * 100
            
            # Add SMA indicators for SMA strategy
            df_temp['SMA_Short'] = df_temp.ta.sma(length=self.sma_short_period, close='Close')
            df_temp['SMA_Long'] = df_temp.ta.sma(length=self.sma_long_period, close='Close')
            
            tp = self.typical_price(df_temp)
            bb = self.bollinger_bands(tp, window=20, stds=2)
            df_temp['bb_lowerband'] = bb['bb_lowerband']
            df_temp['bb_upperband'] = bb['bb_upperband']
            df_temp['chop'] = self.chopiness(df_temp, window=14)
            df_temp['Minimo'] = df_temp['Close']
            df_temp['Maximo'] = df_temp['Close']
            
            self.df_order = df_temp.copy()
            self.ultimo_precio = float(df_temp.iloc[-1]['Close']) if not df_temp.empty else 1.0
            
            logger.info(f"Successfully loaded {len(df_temp)} candles for {token}")
            
            # Send status update to Telegram
            if self.telegram_bot.enabled:
                status_data = self.get_status_data()
                self.telegram_bot.send_status_update(status_data)
                
        except Exception as e:
            logger.error(f"Error al obtener OHLCV: {e}")
    
    def actualizar_datos(self, token):
        if self.df_order.empty:
            return
        
        try:
            ohlcv = self.exchange_ccxt.fetch_ohlcv(symbol=token, timeframe='1h', limit=1)
        except Exception as e:
            logger.error(f"Error al obtener la vela más reciente: {e}")
            return
        
        if not ohlcv:
            return
        
        nueva = ohlcv[0]
        fecha = pd.to_datetime(nueva[0], unit='ms')
        close = float(nueva[4])
        self.ultimo_precio = close
        
        if not self.df_order.empty and self.df_order.iloc[-1]['Date'] == fecha:
            self.df_order.at[self.df_order.index[-1], 'Open'] = float(nueva[1])
            self.df_order.at[self.df_order.index[-1], 'High'] = float(nueva[2])
            self.df_order.at[self.df_order.index[-1], 'Low'] = float(nueva[3])
            self.df_order.at[self.df_order.index[-1], 'Close'] = close
            self.df_order.at[self.df_order.index[-1], 'Volume'] = float(nueva[5])
            self.df_order.at[self.df_order.index[-1], 'Minimo'] = min(self.df_order.iloc[-1]['Minimo'], close)
            self.df_order.at[self.df_order.index[-1], 'Maximo'] = max(self.df_order.iloc[-1]['Maximo'], close)
        else:
            last_row = self.df_order.iloc[-1].copy() if not self.df_order.empty else None
            new_row = last_row.to_dict() if last_row is not None else {}
            new_row['Date'] = fecha
            new_row['Open'] = float(nueva[1])
            new_row['High'] = float(nueva[2])
            new_row['Low'] = float(nueva[3])
            new_row['Close'] = close
            new_row['Volume'] = float(nueva[5])
            new_row['Minimo'] = min(last_row.get('Minimo', close), close) if last_row is not None else close
            new_row['Maximo'] = max(last_row.get('Maximo', close), close) if last_row is not None else close
            self.df_order = pd.concat([self.df_order, pd.DataFrame([new_row])], ignore_index=True)
    
    # ------------------------------
    # TRADING FUNCTIONS
    # ------------------------------
    def ejecutar_compra(self, token, cantidad, precio, fecha, is_backtesting=False):
        # Ensure minimum investment of 20 USD
        if cantidad * precio < 20 and not is_backtesting:
            logger.warning(f"Minimum investment required is 20 USD, skipping buy: {cantidad * precio}USD")
            return False
            
        if self.modo_simulacion:
            self.saldo_money -= cantidad * precio
            self.saldo_monedas += cantidad
            self.status = 'compra'
            self.ultimo_precio = precio
            self.last_buy_price = precio
            
            op_entry = {
                "dt": fecha,
                "Date": fecha.strftime("%Y-%m-%d %H:%M"),
                "Token": token,
                "Operation": "Compra",
                "Price": precio,
                "Quantity": cantidad,
                "Profit": "",
                "is_backtesting": is_backtesting
            }
            self.operations_log.append(op_entry)
            
            nueva = {'Date': fecha, 'Close': precio, 'Ultimo_Status': self.status, 'Ultimo_Precio': precio}
            self.df_simulacion = pd.concat([self.df_simulacion, pd.DataFrame([nueva])], ignore_index=True)
            
            # Enviar notificación Telegram solo si no es operación de backtesting
            if self.telegram_bot.enabled and not is_backtesting:
                self.telegram_bot.send_trade_notification("Compra", token, precio, cantidad)
            
            return True
        else:
            try:
                # Ensure minimum investment of 20 USD
                if cantidad * precio < 20:
                    logger.warning(f"Minimum investment required is 20 USD, skipping buy: {cantidad * precio}USD")
                    return False
                    
                order = self.exchange_ccxt.create_limit_buy_order(token, cantidad, precio)
                logger.info(f"Orden de compra real => {order}")
                
                # Record the operation
                op_entry = {
                    "dt": fecha,
                    "Date": fecha.strftime("%Y-%m-%d %H:%M"),
                    "Token": token,
                    "Operation": "Compra",
                    "Price": precio,
                    "Quantity": cantidad,
                    "Profit": "",
                    "Order_ID": order.get('id', ''),
                    "is_backtesting": False  # Operaciones reales nunca son backtesting
                }
                self.operations_log.append(op_entry)
                
                # Update status
                self.status = 'compra'
                self.ultimo_precio = precio
                self.last_buy_price = precio
                self.saldo_monedas += cantidad
                
                # Guardar estado después de la operación
                self.save_state()
                
                # Send Telegram notification
                if self.telegram_bot.enabled:
                    self.telegram_bot.send_trade_notification("Compra", token, precio, cantidad)
                    
                return True
            except Exception as e:
                logger.error(f"Error compra real: {e}")
                if self.telegram_bot.enabled:
                    self.telegram_bot.send_message(f"❌ *Error en orden de compra*\n{str(e)}")
                return False
    
    def ejecutar_venta(self, token, cantidad, precio, fecha, is_backtesting=False):
        if self.modo_simulacion:
            self.saldo_money += cantidad * precio
            self.saldo_monedas -= cantidad
            self.status = 'venta'
            self.ultimo_precio = precio
            profit = 0.0
            
            if self.last_buy_price is not None:
                profit = (precio - self.last_buy_price) * cantidad
                if precio > self.last_buy_price:
                    self.wins += 1
                elif precio == self.last_buy_price:
                    self.draws += 1
                else:
                    self.losses += 1
            
            op_entry = {
                "dt": fecha,
                "Date": fecha.strftime("%Y-%m-%d %H:%M"),
                "Token": token,
                "Operation": "Venta",
                "Price": precio,
                "Quantity": cantidad,
                "Profit": profit,
                "is_backtesting": is_backtesting
            }
            self.operations_log.append(op_entry)
            
            nueva = {'Date': fecha, 'Close': precio, 'Ultimo_Status': self.status, 'Ultimo_Precio': precio}
            self.df_simulacion = pd.concat([self.df_simulacion, pd.DataFrame([nueva])], ignore_index=True)
            self.last_buy_price = None
            
            # Send Telegram notification
            if self.telegram_bot.enabled and not is_backtesting:
                self.telegram_bot.send_trade_notification("Venta", token, precio, cantidad, profit)
                
            return True
        else:
            try:
                order = self.exchange_ccxt.create_limit_sell_order(token, cantidad, precio)
                logger.info(f"Orden de venta real => {order}")
                
                # Calculate profit if there was a previous buy
                profit = None
                if self.last_buy_price is not None:
                    profit = (precio - self.last_buy_price) * cantidad
                    if precio > self.last_buy_price:
                        self.wins += 1
                    elif precio == self.last_buy_price:
                        self.draws += 1
                    else:
                        self.losses += 1
                
                # Record the operation
                op_entry = {
                    "dt": fecha,
                    "Date": fecha.strftime("%Y-%m-%d %H:%M"),
                    "Token": token,
                    "Operation": "Venta",
                    "Price": precio,
                    "Quantity": cantidad,
                    "Profit": profit,
                    "Order_ID": order.get('id', ''),
                    "is_backtesting": False
                }
                self.operations_log.append(op_entry)
                
                # Update status
                self.status = 'venta'
                self.ultimo_precio = precio
                self.saldo_monedas -= cantidad
                self.last_buy_price = None
                
                # Guardar estado después de la operación
                self.save_state()
                
                # Send Telegram notification
                if self.telegram_bot.enabled:
                    self.telegram_bot.send_trade_notification("Venta", token, precio, cantidad, profit)
                    
                return True
            except Exception as e:
                logger.error(f"Error venta real: {e}")
                if self.telegram_bot.enabled:
                    self.telegram_bot.send_message(f"❌ *Error en orden de venta*\n{str(e)}")
                return False
    
    # ------------------------------
    # NFI STRATEGY IMPLEMENTATION
    # ------------------------------
    def simular_trading_nfi(self, token):
        if self.df_order.empty:
            return
        
        # Determinar si estamos en backtesting o en operaciones en vivo
        real_time_operation = not self.is_backtesting
        
        p = self.NFI_PARAMS
        
        for idx, row in self.df_order.iloc[self.ultimo_procesado:].iterrows():
            fecha = row['Date']
            close = row['Close']
            volume = row['Volume']
            rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
            mfi = row['mfi'] if not pd.isna(row['mfi']) else 50
            ewo = row['ewo'] if not pd.isna(row['ewo']) else 0
            
            ema_20 = row['ema_20'] if not pd.isna(row['ema_20']) else close
            sma_20nfi = row['sma_20_nfi'] if not pd.isna(row['sma_20_nfi']) else close
            trima_20 = row['trima_20'] if not pd.isna(row['trima_20']) else close
            t3_20 = row['t3_20'] if not pd.isna(row['t3_20']) else close
            kama_20 = row['kama_20'] if not pd.isna(row['kama_20']) else close
            
            ema_offset_buy = ema_20 * p['low_offset_ema']
            sma_offset_buy = sma_20nfi * p['low_offset_sma']
            trima_offset_buy = trima_20 * p['low_offset_trima']
            t3_offset_buy = t3_20 * p['low_offset_t3']
            kama_offset_buy = kama_20 * p['low_offset_kama']
            
            # Initialize buy conditions list
            buy_conds = []
            
            # Check all buy conditions
            if p["buy_condition_1_enable"]:
                c1 = (close < ema_offset_buy) and (mfi < 27) and (close < ema_20) and (((ewo < p["ewo_low"]) or (ewo > p["ewo_high"]))) and (volume > 0)
                buy_conds.append(c1)
            if p["buy_condition_2_enable"]:
                c2 = (close < sma_offset_buy) and (mfi < 30) and (close < ema_20) and (((ewo < p["ewo_low"]) or (ewo > p["ewo_high"]))) and (volume > 0)
                buy_conds.append(c2)
            if p["buy_condition_3_enable"]:
                c3 = (row['bb_lowerband'] < close) and (mfi < 35) and (trima_20 > 0) and (((ewo < p["ewo_low"]) or (ewo > p["ewo_high"])))
                buy_conds.append(c3)
            if p["buy_condition_4_enable"]:
                c4 = (close < t3_offset_buy) and (rsi < 40) and (mfi < 35) and (volume > 0)
                buy_conds.append(c4)
            if p["buy_condition_5_enable"]:
                c5 = (close < kama_offset_buy) and (rsi < p["buy_rsi_1"]) and (mfi < p["buy_mfi_1"]) and (volume > 0)
                buy_conds.append(c5)
            if p["buy_condition_6_enable"]:
                c6 = (close < ema_offset_buy) and (mfi < 49) and (volume > 0)
                buy_conds.append(c6)
            if p["buy_condition_7_enable"]:
                c7 = (close < trima_offset_buy) and (rsi < 35) and (volume > 0)
                buy_conds.append(c7)
            if p["buy_condition_8_enable"]:
                c8 = (close < t3_offset_buy) and (rsi < 36) and (row['chop'] < 60) and (volume > 0)
                buy_conds.append(c8)
            if p["buy_condition_9_enable"]:
                c9 = (close < sma_offset_buy) and (mfi < p["buy_mfi_9"]) and (volume > 0)
                buy_conds.append(c9)
            if p["buy_condition_10_enable"]:
                c10 = (close < ema_offset_buy) and (rsi < 35) and (volume > 0)
                buy_conds.append(c10)
            if p["buy_condition_11_enable"]:
                c11 = (close < sma_offset_buy) and (mfi < p["buy_mfi_11"]) and (volume > 0)
                buy_conds.append(c11)
            if p["buy_condition_12_enable"]:
                c12 = (close < ema_offset_buy) and (ewo > 2) and (volume > 0)
                buy_conds.append(c12)
            if p["buy_condition_13_enable"]:
                c13 = (close < sma_offset_buy) and (ewo < -7) and (volume > 0)
                buy_conds.append(c13)
            if p["buy_condition_14_enable"]:
                c14 = (close < ema_offset_buy) and (close < sma_offset_buy) and (rsi < 40) and (volume > 0)
                buy_conds.append(c14)
            if p["buy_condition_15_enable"]:
                c15 = (close < ema_offset_buy) and (rsi < p["buy_rsi_15"]) and (volume > 0)
                buy_conds.append(c15)
            if p["buy_condition_16_enable"]:
                c16 = (close < ema_offset_buy) and (ewo > p["ewo_low"]) and (volume > 0)
                buy_conds.append(c16)
            if p["buy_condition_17_enable"]:
                c17 = (close < sma_offset_buy) and (ewo < -10) and (volume > 0)
                buy_conds.append(c17)
            if p["buy_condition_18_enable"]:
                c18 = (close < ema_offset_buy) and (rsi < p["buy_rsi_18"]) and (volume > 0)
                buy_conds.append(c18)
            if p["buy_condition_19_enable"]:
                c19 = (row['chop'] < p["buy_chop_min_19"]) and (rsi < p["buy_rsi_1h_min_19"])
                buy_conds.append(c19)
            if p["buy_condition_20_enable"]:
                c20 = (close < ema_offset_buy) and (rsi < 26) and (volume > 0)
                buy_conds.append(c20)
            if p["buy_condition_21_enable"]:
                c21 = (close < sma_offset_buy) and (rsi < 23) and (volume > 0)
                buy_conds.append(c21)
            
            buy_signal = any(buy_conds)
            
            # Execute buy if conditions met
            if (self.status == "" or self.status == "venta") and self.saldo_money > 0:
                if buy_signal:
                    cant = self.truncate((self.saldo_money / close) * 0.99, self.precision)
                    self.ejecutar_compra(token, cant, close, fecha, is_backtesting=not real_time_operation)
            
            # Check sell conditions
            if self.status == "compra" and self.saldo_monedas > 0:
                sell_conds = []
                if p["sell_condition_1_enable"]:
                    sc1 = (close > (ema_20 * p["high_offset_ema"])) and (volume > 0)
                    sell_conds.append(sc1)
                if p["sell_condition_2_enable"]:
                    sc2 = (rsi > p["sell_rsi_bb_2"])
                    sell_conds.append(sc2)
                if p["sell_condition_3_enable"]:
                    sc3 = (rsi > p["sell_rsi_main_3"])
                    sell_conds.append(sc3)
                if p["sell_condition_4_enable"]:
                    sc4 = (rsi > p["sell_dual_rsi_rsi_4"]) and (rsi > (p["sell_dual_rsi_rsi_1h_4"] - 5))
                    sell_conds.append(sc4)
                if p["sell_condition_5_enable"]:
                    sc5 = (close > (ema_20 + (ema_20 * p["sell_ema_relative_5"]))) and (rsi > 50)
                    sell_conds.append(sc5)
                if p["sell_condition_6_enable"]:
                    sc6 = (rsi > p["sell_rsi_under_6"])
                    sell_conds.append(sc6)
                if p["sell_condition_7_enable"]:
                    sc7 = (rsi > p["sell_rsi_1h_7"])
                    sell_conds.append(sc7)
                if p["sell_condition_8_enable"]:
                    sc8 = (close > (row['bb_upperband'] * p["sell_bb_relative_8"])) and (volume > 0)
                    sell_conds.append(sc8)
                sell_signal = any(sell_conds)
                
                # Execute sell if conditions met
                if sell_signal:
                    self.ejecutar_venta(token, self.truncate(self.saldo_monedas, self.precision), close, fecha, is_backtesting=not real_time_operation)
            
            self.ultimo_procesado = idx + 1
        
        # Force sell any remaining position at the end
        if self.saldo_monedas > 0 and self.is_backtesting:
            self.ejecutar_venta(token, self.truncate(self.saldo_monedas, self.precision), self.ultimo_precio, fecha, is_backtesting=True)
        
        # Si es un backtesting inicial, marcar como completado
        if self.is_backtesting:
            self.is_backtesting = False
            
    # ------------------------------
    # SMA STRATEGY TRADING
    # ------------------------------
    def simular_trading_sma(self, token):
        """Simulate trading using SMA crossover strategy"""
        if self.df_order.empty:
            return
        
        # Determine if we're in backtesting or live operations
        real_time_operation = not self.is_backtesting
        
        # Apply SMA strategy
        df = self.df_order.copy()
        df = self.implement_sma_strategy(df, self.sma_short_period, self.sma_long_period)
        
        # Iterate through data points starting from last processed
        for idx, row in df.iloc[self.ultimo_procesado:].iterrows():
            fecha = row['Date']
            close = row['Close']
            signal = row['Signal']
            
            # Buy signal
            if (self.status == "" or self.status == "venta") and signal == 1 and self.saldo_money > 0:
                # Ensure minimum investment of 20 USD
                amount_to_invest = max(20, self.saldo_money)
                amount_to_invest = min(amount_to_invest, self.saldo_money)
                
                cant = self.truncate((amount_to_invest / close) * 0.99, self.precision)
                self.ejecutar_compra(token, cant, close, fecha, is_backtesting=not real_time_operation)
            
            # Sell signal
            elif self.status == "compra" and signal == -1 and self.saldo_monedas > 0:
                self.ejecutar_venta(token, self.truncate(self.saldo_monedas, self.precision), close, fecha, is_backtesting=not real_time_operation)
            
            self.ultimo_procesado = idx + 1
        
        # Force sell any remaining position at the end of backtesting
        if self.saldo_monedas > 0 and self.is_backtesting:
            self.ejecutar_venta(token, self.truncate(self.saldo_monedas, self.precision), self.ultimo_precio, fecha, is_backtesting=True)
        
        # Mark backtesting as completed
        if self.is_backtesting:
            self.is_backtesting = False
    
    # ------------------------------
    # STRATEGY MANAGEMENT AND CONFIG
    # ------------------------------
    def update_strategy_params(self, strategy, sma_short=None, sma_long=None):
        """Update the strategy and SMA parameters"""
        changed = False
        
        if strategy in ['NFI', 'SMA']:
            if self.strategy != strategy:
                self.strategy = strategy
                changed = True
        
        if sma_short is not None and sma_short != self.sma_short_period:
            self.sma_short_period = sma_short
            changed = True
            
        if sma_long is not None and sma_long != self.sma_long_period:
            self.sma_long_period = sma_long
            changed = True
            
        if changed:
            # Update config
            self.config['strategy'] = self.strategy
            self.config['sma_short_period'] = self.sma_short_period
            self.config['sma_long_period'] = self.sma_long_period
            save_config(self.config)
            self.save_state()
            
        return changed
    
    def aplicar_configuracion(self, token, limit, cap):
        """Aplicar configuración y ejecutar simulación"""
        # Verificar si hay una operación en curso en modo real
        if not self.modo_simulacion and self.status == 'compra':
            logger.warning("Intento de cambiar configuración con posición abierta en modo real")
            return False, "No se puede cambiar la configuración con una posición abierta en modo real. Cierre la posición primero."
        
        self.current_token = token
        self.initial_capital = cap
        self.reset_vars()
        self.is_backtesting = True  # Activar modo backtesting
        self.cargar_ordenes(token, limit)
        
        if self.modo_simulacion:
            # Modo simulación: ejecutar backtesting normalmente
            if self.strategy == 'NFI':
                self.simular_trading_nfi(token)
            else:  # SMA strategy
                self.simular_trading_sma(token)
        else:
            # Modo real: simular primero, luego ejecutar según posición actual
            modo_simulacion_backup = self.modo_simulacion
            self.modo_simulacion = True
            
            if self.strategy == 'NFI':
                self.simular_trading_nfi(token)
            else:  # SMA strategy
                self.simular_trading_sma(token)
            
            # Guardar última señal de simulación
            sim_status = self.status
            sim_last_buy_price = self.last_buy_price
            
            # Restaurar modo real y actualizar balance real
            self.modo_simulacion = False
            real_balance = self.get_real_balance()
            if real_balance is None:
                real_balance = self.initial_capital
            
            self.saldo_money = real_balance
            self.saldo_monedas = 0.0
            
            # Si la simulación terminó en "compra", intentar replicar en real
            if sim_status == "compra" and sim_last_buy_price is not None:
                # Ensure minimum investment of 20 USD
                investment = max(20, self.saldo_money * 0.99)
                cant = self.truncate(investment / sim_last_buy_price, self.precision)
                
                logger.info(f"Iniciando en modo real replicando posición COMPRA: Cantidad={cant} a precio {sim_last_buy_price}")
                self.ejecutar_compra(token, cant, sim_last_buy_price, datetime.now())
                self.status = "compra"
            else:
                self.status = sim_status
            
            # Guardar configuración actualizada
            self.config['current_token'] = token
            self.config['initial_capital'] = cap
            self.config['simulation_mode'] = self.modo_simulacion
            save_config(self.config)
        
        return True, "Configuración aplicada correctamente"
    
    def background_update(self):
        """Hilo de fondo para actualizaciones automatizadas"""
        while self.running:
            current_time = time.time()
            if current_time >= self.next_operation_time:
                if self.current_token:
                    try:
                        self.actualizar_datos(self.current_token)
                        
                        # Solo ejecutar trading si no estamos en backtesting
                        if not self.is_backtesting:
                            if self.strategy == 'NFI':
                                self.simular_trading_nfi(self.current_token)
                            else:  # SMA strategy
                                self.simular_trading_sma(self.current_token)
                        
                        self.last_operation_time = current_time
                        self.next_operation_time = self.last_operation_time + self.operation_interval
                        
                        # Guardar estado después de cada actualización
                        self.save_state()
                        
                        # Log de la actualización
                        logger.info(f"Actualización automática completada para {self.current_token}")
                        
                        # Enviar actualización de estado por Telegram
                        if self.telegram_bot.enabled:
                            status_data = self.get_status_data()
                            self.telegram_bot.send_status_update(status_data)
                    except Exception as e:
                        logger.error(f"Error en actualización automática: {e}")
                        if self.telegram_bot.enabled:
                            self.telegram_bot.send_message(f"❌ *Error en actualización*\n{str(e)}")
                
                # Verificar nuevamente cada minuto
                time.sleep(60)
            
            time.sleep(60)
    
    def get_trading_trend(self):
        """Determine current trend based on indicators"""
        if self.df_order.empty:
            return "Neutral"
        
        if self.strategy == 'NFI':
            # Get the most recent data
            last_rows = self.df_order.tail(10)
            
            # Check for uptrend
            uptrend = (last_rows['Close'].iloc[-1] > last_rows['ema_20'].iloc[-1]) and \
                      (last_rows['Close'].iloc[-1] > last_rows['Close'].iloc[-5])
            
            # Check for downtrend
            downtrend = (last_rows['Close'].iloc[-1] < last_rows['ema_20'].iloc[-1]) and \
                        (last_rows['Close'].iloc[-1] < last_rows['Close'].iloc[-5])
        else:  # SMA strategy
            # Get the most recent data with SMA values
            last_rows = self.df_order.dropna(subset=['SMA_Short', 'SMA_Long']).tail(10)
            
            if last_rows.empty:
                return "Neutral"
                
            # Check for uptrend (short SMA above long SMA)
            uptrend = last_rows['SMA_Short'].iloc[-1] > last_rows['SMA_Long'].iloc[-1]
            
            # Check for downtrend (short SMA below long SMA)
            downtrend = last_rows['SMA_Short'].iloc[-1] < last_rows['SMA_Long'].iloc[-1]
        
        if uptrend:
            return "Alta"
        elif downtrend:
            return "Baja"
        else:
            return "Neutral"
    
    def get_final_decision(self):
        """Get the final trading decision"""
        if self.df_order.empty:
            return "Sin datos suficientes"
        
        if self.status == "compra":
            return "Mantener - Posición actual de compra"
        
        if self.strategy == 'NFI':
            # Get indicators from last row
            last_row = self.df_order.iloc[-1]
            
            if self.status == "venta" or self.status == "":
                trend = self.get_trading_trend()
                if trend == "Alta":
                    # Check if buy conditions are met
                    if last_row['Close'] < last_row['ema_20'] * self.NFI_PARAMS['low_offset_ema']:
                        return "Comprar - Tendencia alcista y precio en zona de compra"
                    else:
                        return "Esperar - Tendencia alcista pero precio no en zona de compra"
                elif trend == "Baja":
                    return "Esperar - Tendencia bajista"
                else:
                    return "Mantener - Sin tendencia clara"
        else:  # SMA strategy
            # Get SMA values from last row with valid data
            df_valid = self.df_order.dropna(subset=['SMA_Short', 'SMA_Long'])
            
            if df_valid.empty:
                return "Sin datos SMA suficientes"
                
            last_row = df_valid.iloc[-1]
            sma_short = last_row['SMA_Short']
            sma_long = last_row['SMA_Long']
            
            if self.status == "venta" or self.status == "":
                # Determine crossing recently happened
                last_rows = df_valid.tail(3)
                crossing_up = False
                
                if len(last_rows) >= 3:
                    # Check if there was a recent crossing (short SMA crossed above long SMA)
                    if (last_rows['SMA_Short'].iloc[-3] <= last_rows['SMA_Long'].iloc[-3] and 
                        last_rows['SMA_Short'].iloc[-1] > last_rows['SMA_Long'].iloc[-1]):
                        crossing_up = True
                
                if sma_short > sma_long:
                    if crossing_up:
                        return "Comprar - Cruce reciente de SMA (corto > largo)"
                    else:
                        return "Comprar - SMA corto por encima del SMA largo"
                else:
                    return "Esperar - SMA corto por debajo del SMA largo"
            
        return "Mantener - Sin señal clara"
    
    def get_sma_values(self):
        """Get SMA values for display and analysis"""
        if self.df_order.empty:
            return {
                'success': False,
                'error': 'No data available'
            }
        
        # Use the last valid row with SMA data
        df_valid = self.df_order.dropna(subset=['SMA_Short', 'SMA_Long'])
        
        if df_valid.empty:
            return {
                'success': False,
                'error': 'No SMA data calculated yet'
            }
            
        last_row = df_valid.iloc[-1]
        price = last_row['Close']
        sma_short = last_row['SMA_Short']
        sma_long = last_row['SMA_Long']
        
        # Determine trend and signal
        trend = "Alta" if sma_short > sma_long else "Baja"
        
        # Calculate distances
        distance_short = abs(price - sma_short) / price * 100  # Percentage
        distance_long = abs(price - sma_long) / price * 100  # Percentage
        
        return {
            'success': True,
            'price': price,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'short_period': self.sma_short_period,
            'long_period': self.sma_long_period,
            'trend': trend,
            'distance_short_percent': distance_short,
            'distance_long_percent': distance_long
        }
    
    def get_chart_data(self):
        """Get data for chart rendering"""
        if self.df_order.empty:
            return {}
        
        # Format OHLC data for chart
        chart_data = self.df_order.tail(100).copy()
        
        # Convert timestamps to milliseconds for JS
        chart_data['timestamp'] = chart_data['Date'].astype(int) // 10**6
        
        # Format operations for chart - MOSTRAR TODAS LAS OPERACIONES
        operations = []
        for op in self.operations_log:
            # Incluir operaciones tanto de backtesting como reales
            timestamp = op['dt'].timestamp() * 1000 if isinstance(op['dt'], datetime) else pd.to_datetime(op['dt']).timestamp() * 1000
            
            # Determinar tipo de operación (compra/venta)
            op_type = 'buy' if op['Operation'] == 'Compra' else 'sell'
            
            # Añadir un indicador para operaciones de backtesting vs reales
            operations.append({
                'timestamp': timestamp,
                'price': op['Price'],
                'type': op_type,
                'is_backtesting': op.get('is_backtesting', True)
            })
        
        # SMA data if using SMA strategy
        sma_data = {}
        if self.strategy == 'SMA':
            sma_short = chart_data[['timestamp', 'SMA_Short']].dropna().values.tolist()
            sma_long = chart_data[['timestamp', 'SMA_Long']].dropna().values.tolist()
            sma_data = {
                'sma_short': sma_short,
                'sma_long': sma_long
            }
        
        return {
            'prices': chart_data[['timestamp', 'Open', 'High', 'Low', 'Close']].values.tolist(),
            'volume': chart_data[['timestamp', 'Volume']].values.tolist(),
            'ema20': chart_data[['timestamp', 'ema_20']].dropna().values.tolist(),
            'operations': operations,
            'sma_data': sma_data
        }
    
    def get_status_data(self):
        """Get current status data"""
        saldo_total = self.saldo_money
        if self.saldo_monedas > 0:
            saldo_total += self.saldo_monedas * self.ultimo_precio
            
        trend = self.get_trading_trend()
        decision = self.get_final_decision()
        
        # Get SMA data if applicable
        sma_data = None
        if self.strategy == 'SMA':
            sma_data = self.get_sma_values()
        
        return {
            'current_token': self.current_token,
            'current_price': self.ultimo_precio,
            'saldo_money': self.saldo_money,
            'saldo_monedas': self.saldo_monedas,
            'saldo_total': saldo_total,
            'mode': 'Simulación' if self.modo_simulacion else 'Real',
            'status': self.status.capitalize() if self.status else 'Inicial',
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'trend': trend,
            'decision': decision,
            'next_operation': self.next_operation_time,
            'uptime': int(time.time() - self.start_time),
            'telegram_enabled': self.telegram_bot.enabled,
            'strategy': self.strategy,
            'sma_data': sma_data
        }
    
    def get_operations_data(self):
        """Get operations log data"""
        return self.operations_log
    
    def get_telegram_config(self):
        """Get Telegram configuration"""
        return {
            'enabled': self.telegram_config.get('enabled', False),
            'token': self.telegram_config.get('token', ''),
            'chat_id': self.telegram_config.get('chat_id', '')
        }
    
    def get_strategy_data(self):
        """Get current strategy data"""
        return {
            'strategy': self.strategy,
            'sma_short_period': self.sma_short_period,
            'sma_long_period': self.sma_long_period
        }
    
    def get_multi_token_results(self):
        """Get results of multi-token backtesting"""
        return self.multi_token_results


# Initialize the trading bot
bot = TradingBot()

# ------------------------------
# FLASK ROUTES
# ------------------------------
@app.route('/')
def index():
    tokens = bot.tokens_disponibles
    status_data = bot.get_status_data()
    strategy_data = bot.get_strategy_data()
    return render_template('index.html', tokens=tokens, status=status_data, strategy=strategy_data)

@app.route('/api/status')
def api_status():
    return jsonify(bot.get_status_data())

@app.route('/api/chart')
def api_chart():
    return jsonify(bot.get_chart_data())

@app.route('/api/operations')
def api_operations():
    return jsonify(bot.get_operations_data())

@app.route('/api/toggle_mode')
def api_toggle_mode():
    bot.modo_simulacion = not bot.modo_simulacion
    # Update config
    bot.config['simulation_mode'] = bot.modo_simulacion
    save_config(bot.config)
    return jsonify({'success': True, 'mode': 'Simulación' if bot.modo_simulacion else 'Real'})

@app.route('/api/apply_config', methods=['POST'])
def api_apply_config():
    try:
        data = request.json
        token = data.get('token')
        limit = int(data.get('limit', 500))
        capital = float(data.get('capital', 2000))
        
        # Aplicar límites de configuración
        limit = min(limit, bot.max_candles)
        
        success, message = bot.aplicar_configuracion(token, limit, capital)
        
        if success:
            return jsonify({
                'success': True,
                'status': bot.get_status_data(),
                'chart': bot.get_chart_data(),
                'operations': bot.get_operations_data()
            })
        else:
            return jsonify({'success': False, 'error': message})
    except Exception as e:
        logger.error(f"Error applying configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_strategy', methods=['POST'])
def api_update_strategy():
    try:
        data = request.json
        strategy = data.get('strategy')
        sma_short = int(data.get('sma_short', 10))
        sma_long = int(data.get('sma_long', 50))
        
        # Validate SMA periods
        if sma_short >= sma_long:
            return jsonify({'success': False, 'error': 'El período corto debe ser menor que el período largo'})
            
        success = bot.update_strategy_params(strategy, sma_short, sma_long)
        
        if success:
            # Apply new strategy to current token
            bot.cargar_ordenes(bot.current_token, 500)
            
            return jsonify({
                'success': True,
                'strategy': bot.get_strategy_data(),
                'status': bot.get_status_data(),
                'chart': bot.get_chart_data()
            })
        else:
            return jsonify({'success': True, 'message': 'No changes needed'})
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    try:
        data = request.json
        token = data.get('token')
        limit = int(data.get('limit', 500))
        capital = float(data.get('capital', 2000))
        sma_short = int(data.get('sma_short', 10))
        sma_long = int(data.get('sma_long', 50))
        
        # Run backtest
        result = bot.run_sma_backtest(token, limit, capital, sma_short, sma_long)
        
        # Get chart data for backtest visualization
        chart_data = bot.get_backtest_chart_data(token, sma_short, sma_long)
        
        return jsonify({
            'success': True,
            'result': result,
            'chart_data': chart_data
        })
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_multi_backtest', methods=['POST'])
def api_run_multi_backtest():
    try:
        data = request.json
        tokens = data.get('tokens', [])
        limit = int(data.get('limit', 500))
        capital = float(data.get('capital', 2000))
        sma_short = int(data.get('sma_short', 10))
        sma_long = int(data.get('sma_long', 50))
        
        # If no tokens provided, use top 5 available
        if not tokens:
            tokens = bot.tokens_disponibles[:5]
            
        # Run multi-token backtest
        results = bot.run_multi_token_backtest(tokens, limit, capital, sma_short, sma_long)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f"Error running multi-token backtest: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search_token')
def api_search_token():
    search_term = request.args.get('term', '').upper()
    filtered_tokens = [t for t in bot.all_tokens if search_term in t]
    return jsonify(filtered_tokens)

@app.route('/api/update_api_keys', methods=['POST'])
def api_update_api_keys():
    try:
        data = request.json
        api_key = data.get('api_key', '')
        api_secret = data.get('api_secret', '')
        
        if not api_key or not api_secret:
            return jsonify({'success': False, 'error': 'API key and secret are required'})
        
        success = bot.update_api_credentials(api_key, api_secret)
        
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error updating API keys: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_balances')
def api_get_balances():
    try:
        if bot.modo_simulacion:
            return jsonify({
                'success': True, 
                'simulation_mode': True,
                'simulation_balance': {
                    'money': bot.saldo_money,
                    'coins': bot.saldo_monedas,
                    'total': bot.saldo_money + (bot.saldo_monedas * bot.ultimo_precio)
                }
            })
        else:
            balances = bot.get_all_balances()
            return jsonify({
                'success': True, 
                'simulation_mode': False,
                'balances': balances
            })
    except Exception as e:
        logger.error(f"Error fetching balances: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_telegram_config', methods=['POST'])
def api_update_telegram_config():
    try:
        data = request.json
        enabled = data.get('enabled', False)
        token = data.get('token', '')
        chat_id = data.get('chat_id', '')
        
        success = bot.update_telegram_config(enabled, token, chat_id)
        
        # Send a test message if enabled
        if success and enabled and token and chat_id:
            bot.telegram_bot.send_message("✅ *Telegram bot configured successfully*\nYou will now receive trading notifications.")
        
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error updating Telegram config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_telegram_config')
def api_get_telegram_config():
    return jsonify(bot.get_telegram_config())

@app.route('/api/stop_bot')
def api_stop_bot():
    """Detener el bot de forma segura"""
    success = bot.safe_stop()
    return jsonify({'success': success})

@app.route('/api/restart_bot')
def api_restart_bot():
    """Reiniciar el bot"""
    if not bot.running:
        bot.running = True
        bot.update_thread = threading.Thread(target=bot.background_update, daemon=True)
        bot.update_thread.start()
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'El bot ya está en ejecución'})

@app.route('/api/export_operations')
def api_export_operations():
    """Exportar operaciones como CSV"""
    operations = bot.get_operations_data()
    
    # Convertir a cadena CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Escribir encabezados
    writer.writerow(['Fecha', 'Token', 'Operación', 'Precio', 'Cantidad', 'Beneficio', 'Tipo'])
    
    # Escribir datos
    for op in operations:
        writer.writerow([
            op['Date'],
            op['Token'],
            op['Operation'],
            op['Price'],
            op['Quantity'],
            op['Profit'] if op['Profit'] != "" else "",
            "Simulación" if op.get('is_backtesting', True) else "Real"
        ])
    
    # Devolver CSV como adjunto
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=operaciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@app.route('/api/export_backtest_results')
def api_export_backtest_results():
    """Exportar resultados de backtesting como CSV"""
    results = bot.multi_token_results
    
    if not results:
        return jsonify({'success': False, 'error': 'No hay resultados de backtesting disponibles'})
    
    # Convertir a cadena CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Escribir encabezados
    writer.writerow([
        'Token', 'SMA Corto', 'SMA Largo', 'Capital Inicial', 'Balance Final', 
        'Ganancia', 'Ganancia %', 'Buy & Hold %', 'Operaciones', 
        'Victorias', 'Derrotas', 'Tasa de Victoria %', 'Drawdown Máximo %'
    ])
    
    # Escribir datos
    for result in results:
        writer.writerow([
            result.get('token', ''),
            result.get('sma_short', ''),
            result.get('sma_long', ''),
            result.get('initial_capital', 0),
            result.get('final_balance', 0),
            result.get('profit', 0),
            f"{result.get('profit_percent', 0):.2f}",
            f"{result.get('buy_hold_profit_percent', 0):.2f}",
            result.get('total_trades', 0),
            result.get('wins', 0),
            result.get('losses', 0),
            f"{result.get('win_rate', 0):.2f}",
            f"{result.get('max_drawdown', 0):.2f}"
        ])
    
    # Devolver CSV como adjunto
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

if __name__ == '__main__':
    # Start with default configuration for BTC/USD
    if 'BTC/USD' in bot.tokens_disponibles:
        bot.aplicar_configuracion('BTC/USD', 500, 2000)
    elif bot.tokens_disponibles:
        bot.aplicar_configuracion(bot.tokens_disponibles[0], 500, 2000)
    
    app.run(debug=True, threaded=True)