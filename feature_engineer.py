"""
Feature Engineering Module - Defines what signals matter for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering pipeline for transaction anomaly detection.
    Each feature captures a specific signal that could indicate fraud.
    """
    
    def __init__(self):
        self.feature_names = []
        self.feature_groups = {
            'amount_features': [],
            'temporal_features': [],
            'geo_features': [],
            'counterparty_features': [],
            'velocity_features': [],
            'behavioral_features': []
        }
        self.feature_descriptions = {}
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw transaction data
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ========================================
        # 1. AMOUNT-BASED FEATURES
        # ========================================
        
        df['amount_log'] = np.log1p(df['amount'])
        
        # User-based statistics
        if 'user_id' in df.columns:
            df['user_avg_amount'] = df.groupby('user_id')['amount'].transform('mean')
            df['user_std_amount'] = df.groupby('user_id')['amount'].transform('std').fillna(1)
            df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1)
            df['amount_zscore_user'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 0.001)
            df['amount_percentile_user'] = df.groupby('user_id')['amount'].rank(pct=True)
            df['is_amount_spike'] = (df['amount'] > 3 * df['user_avg_amount']).astype(int)
        else:
            df['user_avg_amount'] = df['amount'].mean()
            df['user_std_amount'] = df['amount'].std()
            df['amount_vs_user_avg'] = df['amount'] / (df['amount'].mean() + 1)
            df['amount_zscore_user'] = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 0.001)
            df['amount_percentile_user'] = df['amount'].rank(pct=True)
            df['is_amount_spike'] = (df['amount'] > 3 * df['amount'].mean()).astype(int)
        
        # Global statistics
        df['amount_vs_global_avg'] = df['amount'] / (df['amount'].mean() + 1)
        df['amount_zscore_global'] = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 0.001)
        
        # Round number detection
        df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        df['is_round_thousand'] = (df['amount'] % 1000 == 0).astype(int)
        
        amount_features = ['amount_log', 'amount_vs_user_avg', 'amount_zscore_user', 
                          'amount_percentile_user', 'is_amount_spike', 
                          'amount_vs_global_avg', 'amount_zscore_global',
                          'is_round_amount', 'is_round_thousand']
        self.feature_groups['amount_features'] = amount_features
        
        # ========================================
        # 2. TEMPORAL FEATURES
        # ========================================
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_off_hours'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
            
            if 'user_id' in df.columns:
                df['user_typical_hour'] = df.groupby('user_id')['hour'].transform('median')
                df['hour_deviation'] = abs(df['hour'] - df['user_typical_hour'])
            else:
                df['hour_deviation'] = abs(df['hour'] - 12)
        else:
            df['hour'] = np.random.randint(0, 24, len(df))
            df['day_of_week'] = np.random.randint(0, 7, len(df))
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_off_hours'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
            df['hour_deviation'] = abs(df['hour'] - 12)
        
        temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_off_hours', 
                            'is_business_hours', 'hour_deviation']
        self.feature_groups['temporal_features'] = temporal_features
        
        # ========================================
        # 3. GEOGRAPHIC FEATURES
        # ========================================
        
        if 'sender_country' in df.columns and 'receiver_country' in df.columns:
            df['country_mismatch'] = (df['sender_country'] != df['receiver_country']).astype(int)
            df['corridor'] = df['sender_country'] + '→' + df['receiver_country']
            
            # High-risk corridors
            high_risk_corridors = ['DE→NG', 'GB→CN', 'US→NG', 'GB→NG', 'DE→TR', 'NL→NG']
            df['is_high_risk_corridor'] = df['corridor'].isin(high_risk_corridors).astype(int)
            
            # User's typical country
            if 'user_id' in df.columns:
                user_countries = df.groupby('user_id')['sender_country'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'XX')
                df['user_typical_country'] = df['user_id'].map(user_countries)
                df['is_unusual_country'] = (df['sender_country'] != df['user_typical_country']).astype(int)
            else:
                df['is_unusual_country'] = 0
            
            # Continent mismatch
            continents = {
                'DE': 'EU', 'GB': 'EU', 'FR': 'EU', 'IT': 'EU', 'ES': 'EU', 'NL': 'EU',
                'US': 'NA', 'CA': 'NA', 'MX': 'NA',
                'CN': 'AS', 'JP': 'AS', 'SG': 'AS', 'IN': 'AS', 'AE': 'AS',
                'NG': 'AF', 'ZA': 'AF', 'EG': 'AF',
                'BR': 'SA', 'AR': 'SA'
            }
            df['sender_continent'] = df['sender_country'].map(continents).fillna('XX')
            df['receiver_continent'] = df['receiver_country'].map(continents).fillna('XX')
            df['continent_mismatch'] = (df['sender_continent'] != df['receiver_continent']).astype(int)
        else:
            df['country_mismatch'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
            df['is_unusual_country'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
            df['is_high_risk_corridor'] = np.random.choice([0, 1], len(df), p=[0.95, 0.05])
            df['continent_mismatch'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
        
        geo_features = ['country_mismatch', 'is_unusual_country', 'is_high_risk_corridor', 'continent_mismatch']
        self.feature_groups['geo_features'] = geo_features
        
        # ========================================
        # 4. COUNTERPARTY FEATURES
        # ========================================
        
        if 'receiver_account' in df.columns and 'user_id' in df.columns:
            df['first_txn_date'] = df.groupby(['user_id', 'receiver_account'])['timestamp'].transform('min')
            df['is_new_counterparty'] = (df['timestamp'] == df['first_txn_date']).astype(int)
            df['cp_txn_count'] = df.groupby(['user_id', 'receiver_account'])['transaction_id'].transform('count')
            df['cp_total_amount'] = df.groupby(['user_id', 'receiver_account'])['amount'].transform('sum')
            df['cp_avg_amount'] = df.groupby(['user_id', 'receiver_account'])['amount'].transform('mean')
        else:
            df['is_new_counterparty'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
            df['cp_txn_count'] = np.random.poisson(3, len(df)) + 1
            df['cp_total_amount'] = df['cp_txn_count'] * df['amount']
            df['cp_avg_amount'] = df['amount']
        
        counterparty_features = ['is_new_counterparty', 'cp_txn_count', 'cp_total_amount', 'cp_avg_amount']
        self.feature_groups['counterparty_features'] = counterparty_features
        
        # ========================================
        # 5. VELOCITY FEATURES
        # ========================================
        
        if 'timestamp' in df.columns and 'user_id' in df.columns:
            df = df.sort_values(['user_id', 'timestamp'])
            df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
            df['time_since_last_txn'] = df['time_since_last_txn'].fillna(24)
            
            df['txn_count_1h'] = df.groupby('user_id')['timestamp'].rolling('1H', closed='both').count().reset_index(0, drop=True)
            df['txn_count_1h'] = df['txn_count_1h'].fillna(1) - 1
            df['txn_count_24h'] = df.groupby('user_id')['timestamp'].rolling('24H', closed='both').count().reset_index(0, drop=True)
            df['txn_count_24h'] = df['txn_count_24h'].fillna(1) - 1
        else:
            df['time_since_last_txn'] = np.random.exponential(12, len(df))
            df['txn_count_1h'] = np.random.poisson(2, len(df))
            df['txn_count_24h'] = np.random.poisson(8, len(df)) + 5
        
        df['user_avg_txn_per_day'] = df['txn_count_24h'].mean()
        df['is_velocity_breach'] = (df['txn_count_1h'] > 3 * (df['user_avg_txn_per_day'] / 24 + 1)).astype(int)
        
        velocity_features = ['time_since_last_txn', 'txn_count_1h', 'txn_count_24h', 'is_velocity_breach']
        self.feature_groups['velocity_features'] = velocity_features
        
        # ========================================
        # 6. BEHAVIORAL FEATURES
        # ========================================
        
        if 'user_id' in df.columns:
            df['user_median_amount'] = df.groupby('user_id')['amount'].transform('median')
            if 'receiver_account' in df.columns:
                df['user_unique_cps'] = df.groupby('user_id')['receiver_account'].transform('nunique')
            else:
                df['user_unique_cps'] = np.random.poisson(5, len(df)) + 2
            df['user_cp_diversity'] = df['user_unique_cps'] / (df['txn_count_24h'] + 1)
            
            if 'is_anomaly' in df.columns:
                df['user_anomaly_rate'] = df.groupby('user_id')['is_anomaly'].transform('mean').fillna(0)
                df['is_high_risk_user'] = (df['user_anomaly_rate'] > 0.1).astype(int)
            else:
                df['is_high_risk_user'] = 0
        else:
            df['user_median_amount'] = df['amount'].median()
            df['user_cp_diversity'] = 0.5
            df['is_high_risk_user'] = 0
        
        behavioral_features = ['user_median_amount', 'user_cp_diversity', 'is_high_risk_user']
        self.feature_groups['behavioral_features'] = behavioral_features
        
        # ========================================
        # COMBINE ALL FEATURES
        # ========================================
        
        self.feature_names = (amount_features + temporal_features + geo_features + 
                             counterparty_features + velocity_features + behavioral_features)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return all feature names"""
        return self.feature_names
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return features grouped by category"""
        return self.feature_groups
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each feature signals"""
        return {
            'amount_log': 'Log-transformed amount for handling skew',
            'amount_vs_user_avg': 'How much this transaction deviates from user\'s average',
            'amount_zscore_user': 'Standard deviations from user\'s mean',
            'amount_percentile_user': 'Percentile rank within user\'s transaction history',
            'is_amount_spike': 'Binary flag: amount > 3x user average',
            'amount_vs_global_avg': 'How much this deviates from global average',
            'amount_zscore_global': 'Statistical outlier vs global transactions',
            'is_round_amount': 'Round number detection (structuring signal)',
            'is_round_thousand': 'Round thousand detection (structuring signal)',
            'hour': 'Time of day (off-hours = suspicious)',
            'day_of_week': 'Day of week (weekends may be suspicious)',
            'is_weekend': 'Weekend transaction flag',
            'is_off_hours': 'Off-hours transaction (12am-6am)',
            'is_business_hours': 'Business hours transaction (9am-5pm)',
            'hour_deviation': 'Deviation from user\'s typical transaction hour',
            'country_mismatch': 'Sender and receiver countries differ',
            'is_unusual_country': 'Sender country differs from user\'s typical country',
            'is_high_risk_corridor': 'High-risk country corridor',
            'continent_mismatch': 'Sender and receiver on different continents',
            'is_new_counterparty': 'First time user is sending to this account',
            'cp_txn_count': 'Number of transactions to this counterparty',
            'cp_total_amount': 'Total amount sent to this counterparty',
            'cp_avg_amount': 'Average amount sent to this counterparty',
            'time_since_last_txn': 'Hours since user\'s last transaction',
            'txn_count_1h': 'Number of transactions in last hour (velocity)',
            'txn_count_24h': 'Number of transactions in last 24 hours',
            'is_velocity_breach': 'Velocity > 3x user\'s normal rate',
            'user_median_amount': 'User\'s median transaction amount (baseline)',
            'user_cp_diversity': 'Diversity of user\'s counterparties',
            'is_high_risk_user': 'User has high historical anomaly rate'
        }