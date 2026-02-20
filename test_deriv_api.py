"""
Test script for real Deriv data.
"""
from deriv_fetcher import DerivDataFetcher
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import os
import time  # ← ADD THIS IMPORT

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

def test_with_token():
    print("=" * 60)
    print("Testing Deriv API with Real Token")
    print("=" * 60)
    
    # Get token from environment
    token = os.getenv('DERIV_API_TOKEN')
    
    if not token:
        print("❌ No API token found in .env file")
        print("\nPlease add your token to .env:")
        print('DERIV_API_TOKEN=your_token_here')
        return False
    
    print(f"✅ Token found (length: {len(token)})")
    print(f"   Token starts with: {token[:5]}...")
    
    # Initialize fetcher with token
    fetcher = DerivDataFetcher(api_token=token, app_id="1089")
    
    # Wait for connection
    print("\n⏳ Waiting for WebSocket connection...")
    for i in range(10):
        time.sleep(1)
        status = fetcher.get_status()
        print(f"   Attempt {i+1}: Connected={status['connected']}, Authorized={status['authorized']}")
        if status['authorized']:
            break
    
    # Check status
    status = fetcher.get_status()
    print(f"\n📊 Final Connection Status:")
    print(f"   Connected: {status['connected']}")
    print(f"   Authorized: {status['authorized']}")
    print(f"   Subscriptions: {status['subscriptions']}")
    
    if not status['authorized']:
        print("\n❌ Authorization failed. Your token is invalid.")
        print("\n🔑 How to get a valid Deriv API token:")
        print("   1. Go to https://app.deriv.com")
        print("   2. Log in to your account")
        print("   3. Go to Settings → API Token")
        print("   4. Create a new token with 'Read' permission")
        print("   5. Copy the token (should be 32+ characters)")
        print("\n   Your current token length is", len(token), "- valid tokens are usually longer")
        return False
    
    # Test historical data
    print("\n📡 Testing historical data fetch...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    df = fetcher.fetch_historical_data(
        instrument="EUR/USD",
        start_date=start_date,
        end_date=end_date,
        timeframe="1h"
    )
    
    if df is not None and len(df) > 0:
        print(f"✅ SUCCESS! Got {len(df)} real candles")
        print(f"\nFirst few rows:")
        print(df.head())
        return True
    else:
        print("❌ Failed to get real data (using fallback)")
        return False

if __name__ == "__main__":
    test_with_token()