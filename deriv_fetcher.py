"""
Deriv API data fetcher for NEXUS Trading System.
Fixed version with emoji removed for Windows compatibility.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time
import websocket
import threading
from typing import Optional, Dict, List, Any, Callable
import requests
import random

logger = logging.getLogger(__name__)

class DerivDataFetcher:
    """
    Fetches real and historical data from Deriv API.
    Uses WebSocket for real-time and historical data.
    """
    
    INSTRUMENT_MAP = {
        'EUR/USD': 'frxEURUSD',
        'GBP/USD': 'frxGBPUSD',
        'USD/JPY': 'frxUSDJPY',
        'AUD/USD': 'frxAUDUSD',
        'USD/CAD': 'frxUSDCAD',
        'NZD/USD': 'frxNZDUSD',
        'USD/CHF': 'frxUSDCHF',
        'XAU/USD': 'frxXAUUSD'
    }
    
    REVERSE_MAP = {v: k for k, v in INSTRUMENT_MAP.items()}
    
    TIMEFRAME_MAP = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '1d': 86400
    }
    
    def __init__(self, api_token: str = None, app_id: str = "1089"):
        """
        Initialize Deriv API fetcher.
        
        Args:
            api_token: Your Deriv API token (required for real data)
            app_id: Deriv app ID (default 1089)
        """
        self.api_token = api_token
        self.app_id = app_id
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.authorized = False
        self.subscriptions = {}
        self.price_callbacks = {}
        self.last_prices = {}
        
        # Connection thread
        self.ws_thread = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Cache
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600
        
        # Pending requests
        self.pending_requests = {}
        self.request_id = 0
        
        # Auto-connect if token provided
        if api_token:
            logger.info("Initializing Deriv fetcher with API token")
            self.connect_websocket()
        else:
            logger.warning("No API token provided - using synthetic data only")
    
    def connect_websocket(self):
        """Establish WebSocket connection for real-time data."""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_ws_message(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.ws_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed: {close_status_code}")
            self.ws_connected = False
            self.authorized = False
            
            # Attempt to reconnect
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                logger.info(f"Reconnecting... attempt {self.reconnect_attempts}")
                time.sleep(5)
                self.connect_websocket()
            else:
                logger.error("Max reconnection attempts reached")
        
        def on_open(ws):
            logger.info("WebSocket connected to Deriv")
            self.ws_connected = True
            self.reconnect_attempts = 0
            
            # Authorize if token provided
            if self.api_token:
                auth_msg = json.dumps({
                    "authorize": self.api_token
                })
                self.ws.send(auth_msg)
                logger.info("Authorization sent")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run in separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Wait for connection
        for i in range(10):
            if self.ws_connected:
                break
            time.sleep(0.5)
    
    def _handle_ws_message(self, data: Dict):
        """Handle incoming WebSocket messages."""
        # Handle authorization response
        if 'authorize' in data:
            if 'error' in data:
                logger.error(f"Authorization failed: {data['error']}")
                self.authorized = False
            else:
                logger.info("Successfully authorized with Deriv")
                self.authorized = True
                # Get account currency
                if 'authorize' in data and 'currency' in data['authorize']:
                    logger.info(f"Account currency: {data['authorize']['currency']}")
        
        # Handle tick updates
        elif 'tick' in data:
            tick = data['tick']
            symbol = tick['symbol']
            if symbol in self.REVERSE_MAP:
                instrument = self.REVERSE_MAP[symbol]
                price_info = {
                    'instrument': instrument,
                    'price': float(tick['quote']),
                    'timestamp': datetime.fromtimestamp(tick['epoch']),
                    'bid': float(tick.get('bid', tick['quote'])),
                    'ask': float(tick.get('ask', tick['quote']))
                }
                
                self.last_prices[instrument] = price_info
                
                # Call callbacks
                if instrument in self.price_callbacks:
                    for callback in self.price_callbacks[instrument]:
                        try:
                            callback(price_info)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
        
        # Handle candle history
        elif 'candles' in data:
            req_id = data.get('req_id')
            if req_id and req_id in self.pending_requests:
                self.pending_requests[req_id] = data['candles']
        
        # Handle error
        elif 'error' in data:
            logger.error(f"Deriv API error: {data['error']}")
    
    def _send_request(self, request: Dict, timeout: int = 10) -> Optional[Any]:
        """Send request via WebSocket and wait for response."""
        if not self.ws_connected or not self.authorized:
            logger.warning("WebSocket not ready for requests")
            return None
        
        self.request_id += 1
        request['req_id'] = self.request_id
        self.pending_requests[self.request_id] = None
        
        try:
            self.ws.send(json.dumps(request))
            
            # Wait for response
            start = time.time()
            while time.time() - start < timeout:
                if self.pending_requests[self.request_id] is not None:
                    response = self.pending_requests[self.request_id]
                    del self.pending_requests[self.request_id]
                    return response
                time.sleep(0.1)
            
            # Timeout
            del self.pending_requests[self.request_id]
            logger.warning(f"Request timeout: {request.get('ticks_history', 'unknown')}")
            return None
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def fetch_historical_data(self, 
                            instrument: str, 
                            start_date: datetime,
                            end_date: Optional[datetime] = None,
                            timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLC data from Deriv.
        Returns real data if connected, otherwise falls back to synthetic.
        
        Args:
            instrument: Your instrument name (e.g., 'EUR/USD')
            start_date: Start date
            end_date: End date (defaults to now)
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Check cache first
        cache_key = f"{instrument}_{timeframe}_{start_date.date()}_{end_date.date()}"
        if cache_key in self.cache:
            cache_time = self.cache_expiry.get(cache_key, 0)
            if time.time() - cache_time < self.cache_duration:
                logger.info(f"Returning cached data for {instrument}")
                return self.cache[cache_key].copy()
        
        # Try to get real data if connected
        if self.ws_connected and self.authorized:
            try:
                symbol = self.INSTRUMENT_MAP.get(instrument)
                if not symbol:
                    logger.error(f"Unknown instrument: {instrument}")
                    return self._generate_fallback_data(instrument, start_date, end_date, timeframe)
                
                granularity = self.TIMEFRAME_MAP.get(timeframe, 3600)
                total_seconds = (end_date - start_date).total_seconds()
                count = min(int(total_seconds / granularity) + 100, 5000)
                
                logger.info(f"Fetching {count} {timeframe} candles for {instrument} from {start_date} to {end_date}")
                
                request = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "end": int(end_date.timestamp()),
                    "start": int(start_date.timestamp()),
                    "style": "candles",
                    "granularity": granularity,
                    "count": count
                }
                
                candles = self._send_request(request, timeout=30)
                
                if candles and len(candles) > 0:
                    df_list = []
                    for c in candles:
                        df_list.append({
                            'date': datetime.fromtimestamp(c['epoch']),
                            'open': float(c['open']),
                            'high': float(c['high']),
                            'low': float(c['low']),
                            'close': float(c['close']),
                            'volume': int(c.get('volume', 1000000))
                        })
                    
                    df = pd.DataFrame(df_list)
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Filter to exact range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if len(df) > 0:
                        logger.info(f"Got {len(df)} real candles for {instrument}")
                        
                        # Cache
                        self.cache[cache_key] = df.copy()
                        self.cache_expiry[cache_key] = time.time()
                        
                        return df
                
                logger.warning("No real data received, using fallback")
                
            except Exception as e:
                logger.error(f"Error fetching real data: {e}")
        
        # Fallback to synthetic
        return self._generate_fallback_data(instrument, start_date, end_date, timeframe)
    
    def _generate_fallback_data(self, instrument: str, start_date: datetime, 
                               end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Generate synthetic data when API fails."""
        logger.warning(f"Generating fallback data for {instrument}")
        
        timeframe_seconds = self.TIMEFRAME_MAP.get(timeframe, 3600)
        total_seconds = (end_date - start_date).total_seconds()
        n_points = max(100, int(total_seconds / timeframe_seconds) + 1)
        
        dates = pd.date_range(start=start_date, end=end_date, periods=n_points)
        
        # Get base price
        try:
            import config
            base_price = config.INSTRUMENTS.get(instrument, {}).get('base_price', 100)
            volatility = config.INSTRUMENTS.get(instrument, {}).get('volatility', 0.01)
        except:
            base_price = 100
            volatility = 0.01
        
        # Generate realistic price movement
        np.random.seed(hash(f"{instrument}_{time.time()}") % 2**32)
        
        # Mean-reverting random walk
        returns = np.zeros(n_points)
        for i in range(1, n_points):
            returns[i] = 0.01 * np.random.randn() + 0.1 * returns[i-1]
        
        price = base_price * np.exp(np.cumsum(returns * volatility))
        
        # Generate OHLC
        df = pd.DataFrame({
            'open': price * (1 + 0.001 * np.random.randn(n_points)),
            'high': price * (1 + 0.002 * np.abs(np.random.randn(n_points))),
            'low': price * (1 - 0.002 * np.abs(np.random.randn(n_points))),
            'close': price,
            'volume': (1000000 * (1 + 0.5 * np.abs(returns))).astype(int)
        }, index=dates)
        
        logger.info(f"Generated {len(df)} fallback rows for {instrument}")
        
        return df
    
    def get_realtime_price(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get current real-time price via WebSocket.
        
        Args:
            instrument: Your instrument name
        
        Returns:
            Dictionary with price info or None
        """
        # Check if we have recent cached price
        if instrument in self.last_prices:
            last_update = self.last_prices[instrument]['timestamp']
            if (datetime.now() - last_update).total_seconds() < 60:
                return self.last_prices[instrument]
        
        symbol = self.INSTRUMENT_MAP.get(instrument)
        if not symbol:
            return None
        
        if not self.ws_connected or not self.authorized:
            return None
        
        try:
            # Request a single tick
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "end": "latest",
                "start": 1,
                "style": "ticks",
                "count": 1
            }
            
            ticks = self._send_request(request, timeout=5)
            
            if ticks and len(ticks) > 0:
                tick = ticks[0]
                price_info = {
                    'instrument': instrument,
                    'price': float(tick['quote']),
                    'timestamp': datetime.fromtimestamp(tick['epoch']),
                    'bid': float(tick.get('bid', tick['quote'])),
                    'ask': float(tick.get('ask', tick['quote']))
                }
                self.last_prices[instrument] = price_info
                return price_info
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting realtime price: {e}")
            return None
    
    def subscribe_to_price(self, instrument: str, callback: Callable) -> bool:
        """
        Subscribe to real-time price updates via WebSocket.
        
        Args:
            instrument: Your instrument name
            callback: Function to call with each price update
        
        Returns:
            True if successful
        """
        if not self.ws_connected or not self.authorized:
            logger.warning("Cannot subscribe - WebSocket not ready")
            return False
        
        symbol = self.INSTRUMENT_MAP.get(instrument)
        if not symbol:
            logger.error(f"Unknown instrument: {instrument}")
            return False
        
        # Store callback
        if instrument not in self.price_callbacks:
            self.price_callbacks[instrument] = []
        self.price_callbacks[instrument].append(callback)
        
        # Subscribe if not already subscribed
        if instrument not in self.subscriptions:
            subscribe_msg = json.dumps({
                "ticks": symbol,
                "subscribe": 1
            })
            self.ws.send(subscribe_msg)
            self.subscriptions[instrument] = symbol
            logger.info(f"Subscribed to {instrument} real-time prices")
            
            # Request initial price
            initial_price = self.get_realtime_price(instrument)
            if initial_price:
                callback(initial_price)
        
        return True
    
    def unsubscribe_from_price(self, instrument: str, callback: Optional[Callable] = None):
        """Unsubscribe from real-time price updates."""
        if instrument in self.price_callbacks:
            if callback and callback in self.price_callbacks[instrument]:
                self.price_callbacks[instrument].remove(callback)
            
            # If no callbacks left, unsubscribe
            if not self.price_callbacks[instrument] or callback is None:
                if instrument in self.subscriptions:
                    unsubscribe_msg = json.dumps({
                        "ticks": self.subscriptions[instrument],
                        "subscribe": 0
                    })
                    self.ws.send(unsubscribe_msg)
                    del self.subscriptions[instrument]
                    logger.info(f"Unsubscribed from {instrument}")
                
                del self.price_callbacks[instrument]
    
    def get_available_instruments(self) -> List[str]:
        """Get list of available instruments."""
        return list(self.INSTRUMENT_MAP.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            'connected': self.ws_connected,
            'authorized': self.authorized,
            'subscriptions': list(self.subscriptions.keys()),
            'cached_instruments': list(set(k.split('_')[0] for k in self.cache.keys()))
        }
    
    def close(self):
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
        logger.info("Deriv fetcher closed")