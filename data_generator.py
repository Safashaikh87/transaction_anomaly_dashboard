"""
Synthetic Data Generator - Creates realistic transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

def generate_transactions(
    n: int = 5000,
    anomaly_rate: float = 0.05,
    n_users: int = 100,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic patterns
    """
    if seed:
        np.random.seed(seed)
    
    # Generate user IDs
    user_ids = [f'USR{str(i).zfill(4)}' for i in range(n_users)]
    
    # Countries with realistic distributions
    countries = ['DE', 'GB', 'US', 'FR', 'IT', 'ES', 'NL', 'CH', 'AE', 'SG', 'CN', 'NG']
    countries_probs = [0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02]
    
    # Merchants
    merchants = ['Retail', 'Online', 'Wire Transfer', 'ATM', 'Travel', 
                 'Luxury Goods', 'Cryptocurrency', 'Gambling', 'Real Estate', 
                 'Professional Services']
    
    # Generate data
    data = []
    now = datetime.now()
    
    for i in range(n):
        # User assignment
        user_id = np.random.choice(user_ids)
        user_country = np.random.choice(countries, p=countries_probs)
        
        # Determine if anomaly
        is_anomaly = np.random.random() < anomaly_rate
        
        # Amount generation
        if is_anomaly:
            amount = np.random.exponential(10000) + 5000
        else:
            amount = np.random.gamma(2, 200) + 10
        
        # Time generation
        if is_anomaly and np.random.random() < 0.3:
            hour = np.random.choice([0, 1, 2, 3, 4, 5])
        else:
            hour = np.random.choice(range(6, 23), p=[0.04]*6 + [0.08]*12 + [0.03]*5)
        
        # Timestamp
        days_ago = np.random.randint(0, 7)
        timestamp = now - timedelta(
            days=days_ago, 
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        timestamp = timestamp.replace(hour=hour)
        
        # Countries
        sender_country = user_country
        if is_anomaly and np.random.random() < 0.4:
            receiver_country = np.random.choice([c for c in countries if c != sender_country])
        else:
            receiver_country = np.random.choice(countries, p=countries_probs)
        
        # Receiver account
        receiver_account = f'ACCT{str(np.random.randint(1000, 9999))}'
        
        # Merchant
        merchant = np.random.choice(merchants)
        
        data.append({
            'transaction_id': f'TXN{str(i).zfill(7)}',
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'sender_country': sender_country,
            'receiver_country': receiver_country,
            'receiver_account': receiver_account,
            'merchant_category': merchant,
            'is_anomaly': int(is_anomaly)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['user_id', 'timestamp'])
    return df