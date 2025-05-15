# data.py - Optimized for Replit
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import glob
import logging
import random
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_collection.log')
    ]
)
logger = logging.getLogger('crypto_data')

def retry_function(func, max_retries=5, initial_delay=5, backoff_factor=2, jitter=True):
    """
    Retry function with exponential backoff and jitter
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                actual_delay = delay
                
                # Add jitter to avoid thundering herd problem
                if jitter:
                    actual_delay = delay * (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}. Retrying in {actual_delay:.1f}s")
                time.sleep(actual_delay)
                delay *= backoff_factor
        
        logger.error(f"Function {func.__name__} failed after {max_retries} attempts. Last error: {str(last_exception)}")
        raise last_exception
    
    return wrapper

def calculate_volatility(prices, window=14):
    """Calculate the rolling volatility for a series of prices"""
    if len(prices) < window:
        return np.nan
    
    # Calculate daily returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate standard deviation of returns over the window
    volatility = returns.rolling(window=window).std().iloc[-1] if len(returns) >= window else np.nan
    
    # Annualize (approximately)
    annualized_vol = volatility * np.sqrt(365) if not np.isnan(volatility) else np.nan
    
    return annualized_vol

def get_historical_funding_rates(directory, symbol, days=30):
    """Load historical funding rates from saved CSV files"""
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    all_data = []
    
    # Get all funding rate files
    files = glob.glob(f"{directory}/funding_rates_*.csv")
    files.sort(reverse=True)  # Most recent first
    
    # Load data from files
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'symbol' in df.columns and 'funding_rate' in df.columns:
                all_data.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    # Combine all data
    if not all_data:
        return pd.DataFrame()
    
    historical_data = pd.concat(all_data)
    
    # Filter for the specific symbol
    symbol_data = historical_data[historical_data['symbol'] == symbol].copy()  # Use .copy() to avoid warnings
    
    # Sort by timestamp
    if 'timestamp' in symbol_data.columns:
        symbol_data.loc[:, 'timestamp'] = pd.to_datetime(symbol_data['timestamp'])  # Use .loc to avoid warning
        symbol_data = symbol_data.sort_values('timestamp', ascending=False)
    
    return symbol_data

def load_historical_prices(directory, symbol, market_type='spot', days=14):
    """Load historical price data from saved CSV files"""
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    all_data = []
    
    # Get all price files for the specified market type
    files = glob.glob(f"{directory}/{market_type}_prices_*.csv")
    files.sort(reverse=True)  # Most recent first
    
    # Load data from files
    for file in files:
        try:
            df = pd.read_csv(file)
            if 'symbol' in df.columns and 'price' in df.columns:
                all_data.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    # Combine all data
    if not all_data:
        return pd.DataFrame()
    
    historical_data = pd.concat(all_data)
    
    # Filter for the specific symbol
    symbol_data = historical_data[historical_data['symbol'] == symbol].copy()  # Use .copy() to avoid warnings
    
    # Sort by timestamp
    if 'timestamp' in symbol_data.columns:
        symbol_data.loc[:, 'timestamp'] = pd.to_datetime(symbol_data['timestamp'])  # Use .loc to avoid warning
        symbol_data = symbol_data.sort_values('timestamp', ascending=False)
    
    return symbol_data

def generate_sample_historical_data(symbols, days=30):
    """Generate sample historical data for testing when real history is not available"""
    current_date = datetime.now()
    historical_funding = []
    historical_prices_spot = []
    historical_prices_futures = []
    
    logger.info(f"Generating sample historical data for {len(symbols)} symbols across {days} days")
    
    # Generate data for each day and symbol
    for day in range(days):
        date = current_date - timedelta(days=day)
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')
        
        for symbol in symbols:
            # Generate random funding rate between -0.002 and 0.002
            funding_rate = np.random.uniform(-0.002, 0.002)
            
            # Generate random price based on current crypto prices
            if 'BTC' in symbol:
                base_price = 100000
            elif 'ETH' in symbol:
                base_price = 2500
            elif 'SOL' in symbol:
                base_price = 170
            elif 'BNB' in symbol:
                base_price = 650
            elif 'ADA' in symbol:
                base_price = 0.8
            else:
                base_price = 100
            
            # Add some random walk behavior
            daily_change = np.random.uniform(-0.05, 0.05)
            price_factor = 1 + (daily_change * (day / days))
            spot_price = base_price * price_factor
            
            # Futures price with small basis
            basis = np.random.uniform(-0.01, 0.01)
            futures_price = spot_price * (1 + basis)
            
            # Add funding rate data
            historical_funding.append({
                'symbol': symbol,
                'futures_symbol': symbol.split('/')[0] + '/USDT:USDT',
                'funding_rate': funding_rate,
                'timestamp': date_str
            })
            
            # Add spot price data
            historical_prices_spot.append({
                'symbol': symbol,
                'price': spot_price,
                'market_type': 'spot',
                'timestamp': date_str
            })
            
            # Add futures price data
            historical_prices_futures.append({
                'symbol': symbol,
                'futures_symbol': symbol.split('/')[0] + '/USDT:USDT',
                'price': futures_price,
                'market_type': 'futures',
                'timestamp': date_str
            })
    
    # Create dataframes
    df_funding = pd.DataFrame(historical_funding)
    df_spot = pd.DataFrame(historical_prices_spot)
    df_futures = pd.DataFrame(historical_prices_futures)
    
    logger.info(f"Generated sample data: {len(df_funding)} funding entries, {len(df_spot)} spot prices, {len(df_futures)} futures prices")
    
    return df_funding, df_spot, df_futures

@retry_function
def fetch_ticker(exchange, symbol):
    """Fetch ticker with retry functionality"""
    return exchange.fetch_ticker(symbol)

@retry_function
def fetch_funding_rate(exchange, symbol):
    """Fetch funding rate with retry functionality"""
    return exchange.fetch_funding_rate(symbol)

@retry_function
def load_markets(exchange):
    """Load markets with retry functionality"""
    return exchange.load_markets()

def main():
    try:
        logger.info("Starting crypto funding data collection")
        
        # Create data directory
        data_dir = 'funding_arb_data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure with extended timeouts for Replit
        spot_config = {
            'enableRateLimit': True,
            'timeout': 60000,  # 60 seconds timeout (increased for Replit)
            'verbose': False,  # Less noisy output
            # Use Replit's proxy if needed
            # 'proxies': {
            #     'http': 'http://proxy-server.replit.org:80',
            #     'https': 'http://proxy-server.replit.org:443',
            # }
        }
        
        futures_config = {
            'enableRateLimit': True,
            'timeout': 60000,  # 60 seconds timeout (increased for Replit)
            'verbose': False,  # Less noisy output
            # Use Replit's proxy if needed
            # 'proxies': {
            #     'http': 'http://proxy-server.replit.org:80',
            #     'https': 'http://proxy-server.replit.org:443',
            # }
        }
        
        logger.info("Initializing Binance connections")
        
        # Initialize spot exchange
        try:
            binance_spot = ccxt.binance(spot_config)
            logger.info("Binance spot connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Binance spot: {e}")
            binance_spot = None
        
        # Use binanceusdm for USDT-margined futures
        try:
            binance_futures = ccxt.binanceusdm(futures_config)
            logger.info("Binance futures connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Binance futures: {e}")
            binance_futures = None
        
        if binance_spot is None and binance_futures is None:
            logger.error("Could not initialize any exchange connections, using sample data")
            use_sample_data = True
        else:
            use_sample_data = False
        
        # Define trading pairs to monitor
        spot_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Collecting data at {timestamp}")
        
        # Collections for our data
        spot_data = []
        futures_data = []
        funding_data = []
        
        if not use_sample_data:
            # 1. Fetch spot prices
            logger.info("Fetching spot prices...")
            if binance_spot:
                for symbol in spot_symbols:
                    try:
                        ticker = fetch_ticker(binance_spot, symbol)
                        spot_data.append({
                            'symbol': symbol,
                            'price': ticker['last'],
                            'market_type': 'spot',
                            'timestamp': timestamp
                        })
                        logger.info(f"Got spot price for {symbol}: {ticker['last']}")
                    except Exception as e:
                        logger.error(f"Failed to fetch spot price for {symbol}: {e}")
            
            # 2. Fetch futures markets and prices
            logger.info("Fetching futures markets and prices...")
            futures_mapping = {}
            
            if binance_futures:
                try:
                    futures_markets = load_markets(binance_futures)
                    logger.info(f"Successfully loaded {len(futures_markets)} futures markets")
                    
                    # Map spot symbols to futures
                    for spot_symbol in spot_symbols:
                        base = spot_symbol.split('/')[0]
                        futures_symbol = f"{base}/USDT:USDT"
                        
                        if futures_symbol in futures_markets:
                            futures_mapping[spot_symbol] = futures_symbol
                            logger.info(f"Mapped {spot_symbol} to futures symbol {futures_symbol}")
                        else:
                            logger.warning(f"No matching futures symbol found for {spot_symbol}")
                    
                    # Fetch futures prices and funding rates
                    for spot_symbol, futures_symbol in futures_mapping.items():
                        try:
                            # Get futures price
                            ticker = fetch_ticker(binance_futures, futures_symbol)
                            futures_data.append({
                                'symbol': spot_symbol,
                                'futures_symbol': futures_symbol,
                                'price': ticker['last'],
                                'market_type': 'futures',
                                'timestamp': timestamp
                            })
                            logger.info(f"Got futures price for {futures_symbol}: {ticker['last']}")
                            
                            # Get funding rate
                            funding_info = fetch_funding_rate(binance_futures, futures_symbol)
                            next_funding_time = None
                            if 'nextFundingTime' in funding_info and funding_info['nextFundingTime']:
                                next_funding_time = datetime.fromtimestamp(funding_info['nextFundingTime']/1000).strftime('%Y-%m-%d %H:%M:%S')
                            
                            funding_data.append({
                                'symbol': spot_symbol,
                                'futures_symbol': futures_symbol,
                                'funding_rate': funding_info['fundingRate'],
                                'next_funding_time': next_funding_time,
                                'timestamp': timestamp
                            })
                            logger.info(f"Got funding rate for {futures_symbol}: {funding_info['fundingRate']}")
                        except Exception as e:
                            logger.error(f"Error processing {futures_symbol}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error loading futures markets: {e}")
        
        # Check if we need to generate sample data
        if (not spot_data or not futures_data or not funding_data) or use_sample_data:
            logger.warning("Using sample data as real data collection failed or was bypassed")
            df_funding, df_spot, df_futures = generate_sample_historical_data(spot_symbols, days=1)
        else:
            # Convert to dataframes
            df_spot = pd.DataFrame(spot_data)
            df_futures = pd.DataFrame(futures_data)
            df_funding = pd.DataFrame(funding_data)
        
        # Save data to CSVs
        date_str = datetime.now().strftime('%Y%m%d')
        
        if not df_spot.empty:
            spot_file = f"{data_dir}/spot_prices_{date_str}.csv"
            df_spot.to_csv(spot_file, index=False)
            logger.info(f"Saved {len(df_spot)} spot price records to {spot_file}")
        
        if not df_futures.empty:
            futures_file = f"{data_dir}/futures_prices_{date_str}.csv"
            df_futures.to_csv(futures_file, index=False)
            logger.info(f"Saved {len(df_futures)} futures price records to {futures_file}")
        
        if not df_funding.empty:
            funding_file = f"{data_dir}/funding_rates_{date_str}.csv"
            df_funding.to_csv(funding_file, index=False)
            logger.info(f"Saved {len(df_funding)} funding rate records to {funding_file}")
        
        # Check if we need to generate sample historical data
        # Count existing files to determine if we have enough history
        existing_funding_files = glob.glob(f"{data_dir}/funding_rates_*.csv")
        use_sample_history = len(existing_funding_files) < 5  # Use sample data if less than 5 days of history
        
        # Calculate advanced metrics
        if not df_spot.empty and not df_futures.empty and not df_funding.empty:
            logger.info("Calculating advanced metrics and arbitrage opportunities...")
            
            # Start with basic arbitrage calculations
            df_merged = pd.merge(df_spot, df_futures, on='symbol')
            df_merged = pd.merge(df_merged, df_funding, on=['symbol', 'futures_symbol'])
            
            # Calculate price difference (basis)
            df_merged['price_diff'] = df_merged['price_y'] - df_merged['price_x']
            df_merged['basis_percentage'] = (df_merged['price_diff'] / df_merged['price_x']) * 100
            
            # Calculate absolute funding rate
            df_merged['abs_funding_rate'] = df_merged['funding_rate'].abs()
            
            # Annualized funding rate
            df_merged['annual_funding_rate'] = df_merged['funding_rate'] * 3 * 365  # 3 times per day * 365 days
            
            # Define extreme funding threshold
            extreme_threshold = 0.001  # 0.1% per 8-hour period
            df_merged['extreme_funding'] = df_merged['abs_funding_rate'] > extreme_threshold
            
            # Generate sample historical data if needed
            if use_sample_history:
                logger.info("Generating sample historical data for testing metrics...")
                sample_funding, sample_spot, sample_futures = generate_sample_historical_data(spot_symbols, days=30)
            
            # Calculate historical metrics
            for i, row in df_merged.iterrows():
                symbol = row['symbol']
                
                # 1. Calculate funding rate deviation from 30-day average
                if use_sample_history:
                    # Use sample data for calculations
                    symbol_funding = sample_funding[sample_funding['symbol'] == symbol]
                    avg_funding_rate = symbol_funding['funding_rate'].mean()
                    funding_std = symbol_funding['funding_rate'].std()
                    
                    df_merged.loc[i, '30d_avg_funding'] = avg_funding_rate
                    df_merged.loc[i, 'funding_deviation'] = row['funding_rate'] - avg_funding_rate
                    df_merged.loc[i, 'funding_z_score'] = (row['funding_rate'] - avg_funding_rate) / funding_std if funding_std > 0 else 0
                    
                    # Use sample data for volatility calculation
                    symbol_prices = sample_spot[sample_spot['symbol'] == symbol]
                    if not symbol_prices.empty:
                        vol = np.random.uniform(0.5, 1.2)  # Sample volatility between 50% and 120%
                        df_merged.loc[i, '14d_volatility'] = vol
                    else:
                        df_merged.loc[i, '14d_volatility'] = np.nan
                else:
                    # Use real historical data
                    hist_funding = get_historical_funding_rates(data_dir, symbol, days=30)
                    
                    if not hist_funding.empty and len(hist_funding) > 1:
                        avg_funding_rate = hist_funding['funding_rate'].mean()
                        funding_std = hist_funding['funding_rate'].std()
                        
                        df_merged.loc[i, '30d_avg_funding'] = avg_funding_rate
                        df_merged.loc[i, 'funding_deviation'] = row['funding_rate'] - avg_funding_rate
                        df_merged.loc[i, 'funding_z_score'] = (row['funding_rate'] - avg_funding_rate) / funding_std if funding_std > 0 else 0
                    else:
                        df_merged.loc[i, '30d_avg_funding'] = np.nan
                        df_merged.loc[i, 'funding_deviation'] = np.nan
                        df_merged.loc[i, 'funding_z_score'] = np.nan
                    
                    # 2. Calculate 14-day volatility
                    hist_prices = load_historical_prices(data_dir, symbol, 'spot', days=14)
                    
                    if not hist_prices.empty and len(hist_prices) > 5:
                        volatility = calculate_volatility(hist_prices['price'], window=min(14, len(hist_prices)-1))
                        df_merged.loc[i, '14d_volatility'] = volatility
                    else:
                        df_merged.loc[i, '14d_volatility'] = np.nan
            
            # Clean up column names
            df_arb = df_merged.rename(columns={
                'price_x': 'spot_price',
                'price_y': 'futures_price',
                'timestamp_x': 'timestamp'
            })
            
            # Select relevant columns
            arb_columns = [
                'symbol', 'futures_symbol', 'spot_price', 'futures_price', 
                'price_diff', 'basis_percentage', 'funding_rate', 'abs_funding_rate',
                'annual_funding_rate', '30d_avg_funding', 'funding_deviation',
                'funding_z_score', '14d_volatility', 'extreme_funding',
                'next_funding_time', 'timestamp'
            ]
            
            # Filter columns that exist in the dataframe
            available_columns = [col for col in arb_columns if col in df_arb.columns]
            df_arb = df_arb[available_columns]
            
            # Save arbitrage opportunities with advanced metrics
            arb_file = f"{data_dir}/arbitrage_opportunities_{date_str}.csv"
            df_arb.to_csv(arb_file, index=False)
            logger.info(f"Saved {len(df_arb)} advanced arbitrage opportunities to {arb_file}")
            
            # Display current opportunities with metrics
            pd.set_option('display.float_format', '{:.6f}'.format)
            logger.info(f"\nArbitrage summary stats: Avg basis: {df_arb['basis_percentage'].mean():.6f}%, Avg funding: {df_arb['funding_rate'].mean():.6f}")
            
            # Identify extreme funding opportunities
            extreme_opps = df_arb[df_arb['extreme_funding'] == True]
            if not extreme_opps.empty:
                logger.info(f"Found {len(extreme_opps)} extreme funding opportunities")
                for idx, row in extreme_opps.iterrows():
                    logger.info(f"  - {row['symbol']}: Funding {row['funding_rate']*100:.4f}%, Basis {row['basis_percentage']:.4f}%")
            else:
                logger.info("No extreme funding opportunities detected")
            
        # Clean up old files to save space on Replit
        # Keep only the latest 30 days of data
        def cleanup_old_files():
            try:
                logger.info("Cleaning up old data files...")
                for file_pattern in [f"{data_dir}/spot_prices_*.csv", f"{data_dir}/futures_prices_*.csv", 
                                   f"{data_dir}/funding_rates_*.csv", f"{data_dir}/arbitrage_opportunities_*.csv"]:
                    files = glob.glob(file_pattern)
                    if len(files) > 30:  # Keep 30 most recent files
                        files.sort(reverse=True)  # Most recent first
                        for old_file in files[30:]:
                            try:
                                os.remove(old_file)
                                logger.info(f"Removed old file: {old_file}")
                            except Exception as e:
                                logger.error(f"Error removing {old_file}: {e}")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                
        # Run cleanup
        cleanup_old_files()
            
        logger.info("Data collection completed successfully")
        return True
            
    except Exception as e:
        logger.error(f"Critical error in main processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()