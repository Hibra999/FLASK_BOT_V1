import logging
import requests
import json
import time
import threading
from datetime import datetime

class TelegramBot:
    """
    Versión mejorada del bot de Telegram con comandos similares a Freqtrade.
    """
    def __init__(self, token=None, chat_id=None, enabled=False, trading_bot=None):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and token and chat_id
        self.logger = logging.getLogger(__name__)
        self.trading_bot = trading_bot  # Referencia a la instancia del bot
        self.polling = False
        
        if self.enabled:
            self.logger.info("Telegram notifications enabled")
            self.send_message("🤖 *Trading Bot Inicializado*\nEl bot está ahora en funcionamiento.\n\nComandos disponibles:\n/status - Ver estado del bot\n/profit - Ver información de ganancias\n/performance - Ver estadísticas de rendimiento\n/balance - Ver balance actual\n/help - Obtener ayuda")
            
            # Iniciar sondeo de mensajes si está habilitado
            self.start_polling()
    
    def start_polling(self):
        """Iniciar un hilo en segundo plano para sondear mensajes"""
        if not self.enabled:
            return
            
        self.polling = True
        self.poll_thread = threading.Thread(target=self._poll_messages, daemon=True)
        self.poll_thread.start()
    
    def _poll_messages(self):
        """Sondear nuevos mensajes periódicamente"""
        last_update_id = 0
        
        while self.polling:
            try:
                updates = self._get_updates(last_update_id)
                
                if updates and 'result' in updates and updates['result']:
                    last_update_id = updates['result'][-1]['update_id'] + 1
                    
                    for update in updates['result']:
                        if 'message' in update and 'text' in update['message']:
                            # Procesar comandos
                            self._process_command(update['message'])
            except Exception as e:
                self.logger.error(f"Error polling messages: {e}")
            
            # Sondear cada 5 segundos
            time.sleep(5)
    
    def _get_updates(self, offset=0):
        """Obtener actualizaciones de la API de Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                "offset": offset,
                "timeout": 30
            }
            
            response = requests.get(url, params=params, timeout=60)
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting updates: {e}")
            return None
    
    def _process_command(self, message):
        """Procesar un mensaje de comando"""
        if 'text' not in message:
            return
            
        text = message['text']
        chat_id = message['chat']['id']
        
        # Solo responder a mensajes del chat ID autorizado
        if str(chat_id) != str(self.chat_id):
            return
        
        # Procesar comandos
        if text.startswith('/status'):
            self._cmd_status()
        elif text.startswith('/profit'):
            self._cmd_profit()
        elif text.startswith('/performance'):
            self._cmd_performance()
        elif text.startswith('/balance'):
            self._cmd_balance()
        elif text.startswith('/stop'):
            self._cmd_stop()
        elif text.startswith('/start'):
            self._cmd_start()
        elif text.startswith('/help'):
            self._cmd_help()
        elif text.startswith('/chart'):
            self._cmd_chart()
    
    def _cmd_status(self):
        """Enviar estado del bot"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        status_data = self.trading_bot.get_status_data()
        self.send_status_update(status_data)
    
    def _cmd_profit(self):
        """Enviar información de ganancias"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        # Calcular ganancia
        status_data = self.trading_bot.get_status_data()
        initial_capital = self.trading_bot.initial_capital
        current_value = status_data['saldo_total']
        profit_abs = current_value - initial_capital
        profit_pct = (profit_abs / initial_capital) * 100
        
        message = f"💰 *Resumen de Ganancias*\n\n"
        message += f"Capital inicial: ${initial_capital:.2f}\n"
        message += f"Valor actual: ${current_value:.2f}\n"
        message += f"Ganancia absoluta: ${profit_abs:.2f}\n"
        message += f"Ganancia relativa: {profit_pct:.2f}%\n\n"
        message += f"Victorias: {status_data['wins']}\n"
        message += f"Derrotas: {status_data['losses']}\n"
        
        self.send_message(message)
    
    def _cmd_performance(self):
        """Enviar información de rendimiento"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        # Obtener registro de operaciones
        operations = self.trading_bot.get_operations_data()
        
        if not operations:
            self.send_message("No hay operaciones de trading registradas aún.")
            return
            
        # Filtrar solo operaciones reales
        real_ops = [op for op in operations if not op.get('is_backtesting', True)]
        
        # Calcular métricas de rendimiento
        total_ops = len(real_ops)
        buy_ops = len([op for op in real_ops if op['Operation'] == 'Compra'])
        sell_ops = len([op for op in real_ops if op['Operation'] == 'Venta'])
        
        profits = [op['Profit'] for op in real_ops if op['Operation'] == 'Venta' and op['Profit'] != ""]
        if profits:
            avg_profit = sum(profits) / len(profits)
            best_profit = max(profits) if profits else 0
            worst_loss = min(profits) if profits else 0
        else:
            avg_profit = best_profit = worst_loss = 0
            
        message = f"📊 *Estadísticas de Rendimiento*\n\n"
        message += f"Total operaciones reales: {total_ops}\n"
        message += f"Operaciones de compra: {buy_ops}\n"
        message += f"Operaciones de venta: {sell_ops}\n\n"
        message += f"Ganancia media por operación: ${avg_profit:.2f}\n"
        message += f"Mejor operación: ${best_profit:.2f}\n"
        message += f"Peor operación: ${worst_loss:.2f}\n"
        
        self.send_message(message)
    
    def _cmd_balance(self):
        """Enviar información de balance"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        # Obtener info de balance
        status_data = self.trading_bot.get_status_data()
        
        message = f"💼 *Información de Balance*\n\n"
        message += f"Modo: {status_data['mode']}\n"
        message += f"Token: {status_data['current_token']}\n"
        message += f"Balance USD: ${status_data['saldo_money']:.2f}\n"
        message += f"Balance Token: {status_data['saldo_monedas']:.5f}\n"
        message += f"Balance Total: ${status_data['saldo_total']:.2f}\n"
        
        self.send_message(message)
    
    def _cmd_stop(self):
        """Detener el bot de trading"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        # Implementar una parada segura
        self.send_message("⏹ Deteniendo el bot de trading...")
        
        if self.trading_bot.safe_stop():
            self.send_message("✅ Bot de trading detenido correctamente")
        else:
            self.send_message("❌ Error al detener el bot de trading")
    
    def _cmd_start(self):
        """Iniciar el bot de trading"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
            
        # Implementar inicio del bot
        if not self.trading_bot.running:
            self.trading_bot.running = True
            self.trading_bot.update_thread = threading.Thread(target=self.trading_bot.background_update, daemon=True)
            self.trading_bot.update_thread.start()
            self.send_message("✅ Bot de trading iniciado correctamente")
        else:
            self.send_message("⚠️ El bot de trading ya está en ejecución")
    
    def _cmd_chart(self):
        """Enviar información sobre el gráfico actual"""
        if not self.trading_bot:
            self.send_message("❌ Referencia al bot de trading no disponible")
            return
        
        token = self.trading_bot.current_token
        price = self.trading_bot.ultimo_precio
        trend = self.trading_bot.get_trading_trend()
        decision = self.trading_bot.get_final_decision()
        
        trend_emoji = "🟢" if trend == "Alta" else "🔴" if trend == "Baja" else "⚪️"
        
        message = f"📈 *Análisis de {token}*\n\n"
        message += f"Precio actual: ${price:.2f}\n"
        message += f"Tendencia: {trend_emoji} {trend}\n"
        message += f"Decisión: {decision}\n\n"
        message += f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message)
    
    def _cmd_help(self):
        """Enviar información de ayuda"""
        message = "🤖 *Comandos del Bot de Trading*\n\n"
        message += "/status - Ver estado actual del bot\n"
        message += "/profit - Ver información de ganancias\n"
        message += "/performance - Ver estadísticas de rendimiento\n"
        message += "/balance - Ver balance actual\n"
        message += "/chart - Ver análisis actual del gráfico\n"
        message += "/stop - Detener el bot de trading\n"
        message += "/start - Iniciar el bot de trading\n"
        message += "/help - Mostrar este mensaje de ayuda\n"
        
        self.send_message(message)
    
    def send_message(self, message, parse_mode="Markdown"):
        """Envía un mensaje utilizando la API de Telegram directamente"""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            return True
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_notification(self, operation_type, token, price, quantity, profit=None):
        """Envía una notificación de operación de trading"""
        if not self.enabled:
            return
        
        if operation_type.lower() == "buy" or operation_type.lower() == "compra":
            emoji = "🟢"
            title = "ORDEN DE COMPRA EJECUTADA"
        else:
            emoji = "🔴"
            title = "ORDEN DE VENTA EJECUTADA"
        
        message = f"{emoji} *{title}*\n\n"
        message += f"*Token:* {token}\n"
        message += f"*Precio:* ${price:.2f}\n"
        message += f"*Cantidad:* {quantity:.5f}\n"
        
        if profit is not None and (operation_type.lower() == "venta" or operation_type.lower() == "sell"):
            profit_emoji = "✅" if profit > 0 else "❌"
            message += f"*Beneficio:* {profit_emoji} ${profit:.2f}\n"
        
        # Añadir marca de tiempo
        message += f"\n*Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_message(message)
    
    def send_status_update(self, status_data):
        """Envía una actualización de estado del bot"""
        if not self.enabled:
            return
        
        mode_emoji = "🔄" if status_data['mode'] == "Simulación" else "💵"
        trend_emoji = "📈" if status_data['trend'] == "Alta" else "📉" if status_data['trend'] == "Baja" else "📊"
        
        message = "📊 *Estado del Bot de Trading*\n\n"
        message += f"*Token:* {status_data['current_token']}\n"
        message += f"*Precio:* ${status_data['current_price']:.2f}\n"
        message += f"*Balance:* ${status_data['saldo_total']:.2f}\n"
        message += f"*Monedas:* {status_data['saldo_monedas']:.5f}\n"
        message += f"*Modo:* {mode_emoji} {status_data['mode']}\n"
        message += f"*Estado:* {status_data['status']}\n"
        message += f"*Tendencia:* {trend_emoji} {status_data['trend']}\n"
        message += f"*V/D:* ✅ {status_data['wins']} / ❌ {status_data['losses']}\n\n"
        message += f"*Decisión:* {status_data['decision']}"
        
        self.send_message(message)
    
    def stop(self):
        """Detener sondeo de mensajes"""
        self.polling = False
        
        if hasattr(self, 'poll_thread') and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=5)