import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
import json
from pathlib import Path
from typing import Tuple, Dict

class TicketPriceAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions_history = []
        self.best_params = {}
        self.price_threshold = 0.2  # Consider lowest 20% prices as good deals
        
    def load_and_prepare_data(self, df):
        """
        Prepare the dataset for modeling
        """
        # First split the data
        train_df, val_df, test_df = self.split_data(df)
        
        # Engineer features for each set separately
        train_df = self.engineer_advanced_features(train_df)
        val_df = self.engineer_advanced_features(val_df)
        test_df = self.engineer_advanced_features(test_df)
        
        # Handle categorical variables
        train_df = self.encode_categorical_features(train_df)
        val_df = self.encode_categorical_features(val_df)
        test_df = self.encode_categorical_features(test_df)
        
        # Store the processed dataframes
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        return train_df, val_df, test_df
    
    def split_data(self, df, validation_frac=0.2, test_frac=0.2, random_state=42):
        """
        Split data into train, validation, and test sets by event, ensuring all listings
        for a given event stay in the same split. Uses the maximum (latest) listing_time of each
        event to determine the split.

        Default split: 60% training, 20% validation, 20% testing (by time, not by event count)

        Note: All listings for an event will be in the same split. However, some listings in the validation set
        may begin before the end of the training set. This is acceptable as long as all listings for an event
        are in the same split.

        Process:
        1. For each event, get the maximum (latest) listing_time
        2. Sort events by their maximum listing_time (chronological order)
        3. Compute time thresholds for 60/20/20 splits
        4. Assign events to splits based on which time range their max listing_time falls into
        5. Ensure all listings for an event stay in the same split
        """
        # Get the maximum (latest) listing_time for each event and sort in chronological order
        event_last_listings = df.groupby('event_name')['listing_time'].max().reset_index()
        event_last_listings = event_last_listings.sort_values('listing_time')

        # Compute time thresholds for splits
        min_time = event_last_listings['listing_time'].min()
        max_time = event_last_listings['listing_time'].max()
        total_seconds = (max_time - min_time).total_seconds()
        train_threshold = min_time + pd.Timedelta(seconds=total_seconds * 0.6)
        val_threshold = min_time + pd.Timedelta(seconds=total_seconds * 0.8)

        # Assign events to splits based on their max listing_time
        train_events = set(event_last_listings[event_last_listings['listing_time'] <= train_threshold]['event_name'])
        val_events = set(event_last_listings[(event_last_listings['listing_time'] > train_threshold) & (event_last_listings['listing_time'] <= val_threshold)]['event_name'])
        test_events = set(event_last_listings[event_last_listings['listing_time'] > val_threshold]['event_name'])

        # Split the data based on event names
        train_df = df[df['event_name'].isin(train_events)]
        val_df = df[df['event_name'].isin(val_events)]
        test_df = df[df['event_name'].isin(test_events)]

        # Get temporal boundaries for each split
        train_max_listing_time = train_df['listing_time'].max()
        val_min_listing_time = val_df['listing_time'].min() if len(val_df) > 0 else None
        val_max_listing_time = val_df['listing_time'].max() if len(val_df) > 0 else None
        test_min_listing_time = test_df['listing_time'].min() if len(test_df) > 0 else None

        print(f"\nData Split Summary:")
        print("==================")
        print(f"Training set: {len(train_events)} events, {len(train_df)} listings")
        print(f"Validation set: {len(val_events)} events, {len(val_df)} listings")
        print(f"Test set: {len(test_events)} events, {len(test_df)} listings")

        print("\nTemporal Boundaries (by Maximum Listing Time):")
        print("=============================================")
        print(f"Training set: {train_df['listing_time'].min()} to {train_max_listing_time}")
        print(f"Validation set: {val_min_listing_time} to {val_max_listing_time}")
        print(f"Test set: {test_min_listing_time} to {test_df['listing_time'].max()}")

        # Verify no event overlap between splits
        train_val_overlap = len(train_events & val_events)
        train_test_overlap = len(train_events & test_events)
        val_test_overlap = len(val_events & test_events)

        print(f"\nEvent Overlap Check:")
        print(f"Events in train & val: {train_val_overlap}")
        print(f"Events in train & test: {train_test_overlap}")
        print(f"Events in val & test: {val_test_overlap}")

        # Assert no overlap between splits
        assert train_val_overlap == 0, "Data leakage detected: Events appear in both training and validation sets"
        assert train_test_overlap == 0, "Data leakage detected: Events appear in both training and test sets"
        assert val_test_overlap == 0, "Data leakage detected: Events appear in both validation and test sets"

        self.train_df, self.val_df, self.test_df = train_df, val_df, test_df
        return train_df, val_df, test_df

    def engineer_advanced_features(self, df):
        """
        Create advanced features for price timing prediction
        """
        df = df.copy()
        
        # Ensure listing_time column exists and is properly formatted
        if 'listing_time' not in df.columns:
            raise ValueError("Required column 'listing_time' not found in DataFrame. This column must be present.")
            
        # Convert to UTC first to handle timezone-aware datetimes
        df['listing_time'] = pd.to_datetime(df['listing_time'], utc=True)
        # Convert to EST
        df['listing_time'] = df['listing_time'].dt.tz_convert('US/Eastern')
            
        # Ensure chronological order
        df = df.sort_values(by=['event_name', 'listing_time'])
        
        # Calculate price change over 48 hours using only historical data
        def calculate_price_change(group):
            # Calculate time difference between listings
            group['time_diff'] = group['listing_time'].diff()
            
            # Calculate price change for listings within 48 hours
            group['price_change_48h'] = np.nan
            
            for idx in range(1, len(group)):
                current_time = group.iloc[idx]['listing_time']
                current_price = group.iloc[idx]['price']
                
                # Look back at previous listings within 48 hours
                prev_listings = group.iloc[:idx]
                prev_listings = prev_listings[prev_listings['listing_time'] >= current_time - pd.Timedelta(hours=48)]
                
                if len(prev_listings) > 0:
                    # Use the most recent listing's price
                    prev_price = prev_listings.iloc[-1]['price']
                    group.loc[group.index[idx], 'price_change_48h'] = (current_price - prev_price) / prev_price
            
            return group
        
        # Apply per event group
        df = df.groupby('event_name', group_keys=False).apply(calculate_price_change)
        
        # Fill NaN values with 0 for first listings or when no previous listings within 48h
        df['price_change_48h'] = df['price_change_48h'].fillna(0)
        
        # Drop temporary column
        df = df.drop('time_diff', axis=1)
        
        # 1. Within-event relative features
        # Calculate statistics using only historical data for each event
        df['price_vs_event_min'] = 0.0  # Initialize as float
        df['price_vs_event_median'] = 0.0  # Initialize as float
        df['price_vs_event_max'] = 0.0
        
        # Calculate expanding window statistics
        def calculate_features(group):
            # Historical min/median/max (excludes current row)
            group['hist_min'] = group['price'].shift(1).expanding(min_periods=5).min()
            group['hist_median'] = group['price'].shift(1).expanding(min_periods=5).median()
            group['hist_max'] = group['price'].shift(1).expanding(min_periods=5).max()
            
            # Calculate ratios
            group['price_vs_event_min'] = group['price'] / group['hist_min']
            group['price_vs_event_median'] = group['price'] / group['hist_median']
            group['price_vs_event_max'] = group['price'] / group['hist_max']
            
            # Handle insufficient history - use 1.0 as default 
            mask = group['hist_min'].isna()
            group.loc[mask, 'price_vs_event_min'] = 1.0
            group.loc[mask, 'price_vs_event_median'] = 1.0
            group.loc[mask, 'price_vs_event_max'] = 1.0
            
            return group
        
        # Apply per event group
        df = df.groupby('event_name', group_keys=False).apply(calculate_features)
        
        # Clean up intermediate columns
        df = df.drop(['hist_min', 'hist_median', 'hist_max'], axis=1)
        
        # 2. Time-decay features
        decay_constant = 30
        df['demand_pressure'] = np.exp(-df['days_until_event'] / decay_constant)
        df['urgency_factor'] = 1 / (df['days_until_event'] + 1)
        
        max_days_per_event = df.groupby('event_name')['days_until_event'].transform('max')
        df['time_to_event_ratio'] = df['days_until_event'] / max_days_per_event
        
        # 3. Market movement features (using only past data)
        # Pre-calculate cumulative features
        df['cumulative_quantity'] = df.groupby('event_name')['quantity'].cumsum().shift(1).fillna(0)
        df['prev_cumulative'] = df.groupby('event_name')['cumulative_quantity'].shift(1).fillna(0)
        
        # Calculate rolling statistics using time-based 7-day window
        def rolling_time_features(group):
            group = group.sort_values('listing_time')
            group = group.set_index('listing_time')
            group['listing_volume_7d'] = group['quantity'].rolling('7D', min_periods=1).count()
            group['avg_quantity_7d'] = group['quantity'].rolling('7D', min_periods=1).mean()
            group['price_std_7d'] = group['price'].rolling('7D', min_periods=1).std()
            
            # Calculate price volatility (price_std / price_mean)
            price_mean = group['price'].rolling('7D', min_periods=1).mean().replace(0, 1e-6)
            group['price_volatility'] = group['price_std_7d'] / price_mean
            
            group = group.reset_index()
            return group
        df = df.groupby('event_name', group_keys=False).apply(rolling_time_features)
        df['listing_volume_7d'] = df['listing_volume_7d'].fillna(0)
        df['avg_quantity_7d'] = df['avg_quantity_7d'].fillna(0)
        df['price_std_7d'] = df['price_std_7d'].fillna(0)
        
        # 4. Event-specific features (using only historical data)
        df['event_listing_frequency'] = 0
        df['event_avg_price'] = 0.0
        df['event_price_std'] = 0.0
        
        for event_name in df['event_name'].unique():
            event_mask = df['event_name'] == event_name
            event_df = df[event_mask].copy()
            
            for idx in event_df.index:
                current_time = event_df.loc[idx, 'listing_time']
                
                # Get historical data for this event (only before current time)
                historical_data = event_df[event_df['listing_time'] < current_time]
                
                if len(historical_data) > 0:
                    df.loc[idx, 'event_listing_frequency'] = len(historical_data)
                    df.loc[idx, 'event_avg_price'] = historical_data['price'].mean()
                    df.loc[idx, 'event_price_std'] = historical_data['price'].std()
                else:
                    # Use default values for first listings
                    df.loc[idx, 'event_listing_frequency'] = 0
                    df.loc[idx, 'event_avg_price'] = df.loc[idx, 'price']
                    df.loc[idx, 'event_price_std'] = 0.0
        
        # 5. Timeline features
        df['prior_listings_count'] = df.groupby('event_name').cumcount()  # Number of PRIOR listings
        
        # Calculate days since first listing for each event
        df['days_since_first_listing'] = df.groupby('event_name')['listing_time'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600)
        )
        
        # 6. Event Progress features
        df['event_duration_so_far'] = (
            df.groupby('event_name')['listing_time'].transform(
                lambda x: (x.max() - x.min()).total_seconds() / (24 * 3600)
            )
        )
        df['listing_intensity'] = df['prior_listings_count'] / (df['event_duration_so_far'] + 1e-6)
        
        # 7. Phase Flags
        df['is_early_listing'] = (df['prior_listings_count'] < 5).astype(int)
        df['is_last_week'] = (df['days_until_event'] < 7).astype(int)
        
        # 8. Market metrics
        # Calculate expanding 20th percentile
        df['expanding_p20'] = df.groupby('event_name')['price'].transform(
            lambda x: x.expanding(min_periods=5).quantile(0.2).shift(1)
        )
        df['price_vs_expanding_p20'] = df['price'] / df['expanding_p20'].replace(0, np.nan)
        
        # Calculate historical volatility using a fixed window of 15 listings
        df['hist_volatility_30d'] = df.groupby('event_name')['price'].transform(
            lambda x: x.rolling(15, min_periods=5).std().shift(1)
        )
        
        # 9. Time window features
        df['time_window'] = df['listing_time'].dt.floor('7D')
        
        # Market stats from most recent period with data
        market_stats = df.groupby(['event_name', 'time_window']).agg(
            market_median=('price', 'median'),
            market_size=('price', 'count')
        ).reset_index()
        
        # For each event and time window, find the most recent market data
        def get_prev_market_stats(group):
            # Sort by time window
            group = group.sort_values('time_window')
            # Create a mapping of time windows to their stats
            window_stats = {}
            prev_median = None
            prev_size = None
            
            for _, row in group.iterrows():
                window = row['time_window']
                if prev_median is not None:  # If we have previous data
                    window_stats[window] = {
                        'prev_market_median': prev_median,
                        'prev_market_size': prev_size
                    }
                # Update previous stats for next iteration
                prev_median = row['market_median']
                prev_size = row['market_size']
            
            return pd.DataFrame.from_dict(window_stats, orient='index')
        
        # Apply the function to each event group
        prev_market = market_stats.groupby('event_name').apply(get_prev_market_stats).reset_index()
        prev_market = prev_market.rename(columns={'level_1': 'time_window'})
        
        # Merge with main dataframe
        df = df.merge(prev_market, on=['event_name', 'time_window'], how='left')
        
        # Calculate price vs previous market with error handling
        df['price_vs_prev_market'] = np.where(
            (df['prev_market_median'].notna()) & (df['prev_market_median'] > 0),
            df['price'] / df['prev_market_median'],
            1.0  # Default to 1.0 if no previous market data
        )
        
        # Fill missing market size with 0
        df['prev_market_size'] = df['prev_market_size'].fillna(0)
        
        # Fill NA for events with insufficient history
        new_event_mask = (df['prior_listings_count'] == 0)
        df.loc[new_event_mask, 'price_trend_7d'] = 0
        df.loc[new_event_mask, 'inventory_burn_rate'] = 0.5  # Default neutral value
        
        # Calculate price stability using only past data
        def calculate_price_stability(group):
            # Create a copy to avoid modifying the original
            group = group.copy()
            # Initialize the stability column
            group['price_stability_7d'] = np.nan
            
            # For each row, use only past data
            for i in range(len(group)):
                if i < 7:  # Skip first 7 rows as we need at least 7 days of data
                    continue
                    
                # Get the current timestamp
                current_time = group.iloc[i]['listing_time']
                # Get data from the past 7 days
                past_data = group.iloc[:i][group.iloc[:i]['listing_time'] >= current_time - pd.Timedelta(days=7)]
                
                if len(past_data) > 0:
                    # Calculate stability using only past data
                    std = past_data['price'].std()
                    mean = past_data['price'].mean()
                    if mean != 0:  # Avoid division by zero
                        group.iloc[i, group.columns.get_loc('price_stability_7d')] = std / mean
            
            return group
        
        # Apply the calculation for each event
        df = df.groupby('event_name', group_keys=False).apply(calculate_price_stability)
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical variables
        """
        df = df.copy()
        
        # Label encode categorical features
        categorical_cols = ['zone', 'section', 'event_name', 'day_of_week']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.scalers[f'{col}_encoder'] = le
        
        return df
    
    def prepare_features_target(self, df, min_feature_importance=0.0):
        """
        Prepare feature matrix and target variable for classification.
        Target is 1 if price is in lowest 20% of current market prices for that event-time window, 0 otherwise.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            min_feature_importance (float): Minimum feature importance threshold. Features with importance below this will be dropped.
        """
        print("\nPreparing features and target:")
        
        # Define feature columns
        feature_cols = [
            'avg_quantity_7d', 'days_since_first_listing', 'demand_pressure',
            'event_avg_price', 'event_duration_so_far', 'event_listing_frequency',
            'event_month', 'event_price_std', 'hist_volatility_30d',
            'is_weekend_event', 'listing_intensity', 'listing_volume_7d',
            'prev_market_size', 'price_change_48h', 'price_std_7d',
            'price_volatility', 'price_vs_event_max', 'price_vs_event_median',
            'price_vs_event_min', 'price_vs_expanding_p20', 'price_vs_prev_market',
            'prior_listings_count', 'time_to_event_ratio', 'price_stability_7d'
        ]
        
        # Remove any features that don't exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print("\nMissing features in DataFrame:")
            print(sorted(missing_features))
            print("\nAvailable columns in DataFrame:")
            print(sorted(df.columns.tolist()))
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Store feature columns as instance variable
        self.feature_cols = feature_cols
        
        # Calculate CURRENT market percentiles per event-time_window
        df['current_market_p20'] = df.groupby(['event_name', 'time_window'])['price'].transform(
            lambda x: x.quantile(0.2)
        )
        
        # Define target relative to CURRENT market
        df['is_good_deal'] = (df['price'] <= df['current_market_p20']).astype(int)
        
        # Handle missing values
        X = df[self.feature_cols].fillna(0)
        y = df['is_good_deal']
        
        # If min_feature_importance is specified, analyze and drop low-importance features
        if min_feature_importance > 0:
            X, self.feature_cols = self.drop_low_importance_features(X, y, min_feature_importance)
        
        return X, y, self.feature_cols

    def drop_low_importance_features(self, X, y, min_importance=0.01):
        """
        Analyze feature importance and drop features below the minimum importance threshold.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            min_importance (float): Minimum feature importance threshold
            
        Returns:
            tuple: (X with dropped features, list of remaining feature names)
        """
        print("\nAnalyzing feature importance...")
        
        # Train a quick XGBoost model to get feature importance
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Identify features to keep
        features_to_keep = importance_df[importance_df['importance'] >= min_importance]['feature'].tolist()
        
        # Print dropped features
        dropped_features = set(feature_names) - set(features_to_keep)
        if dropped_features:
            print("\nDropping low-importance features:")
            for feature in sorted(dropped_features):
                importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                print(f"{feature}: {importance:.4f}")
        
        # Return filtered X and feature names
        return X[features_to_keep], features_to_keep

    def train_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train all models and return their results."""
        results = {}
        predictions = {}
        feature_importance = {}
        metadata = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'feature_columns': self.feature_cols  # Store the feature columns used during training
        }
        
        # Store the feature names used during training
        self.training_feature_names = X_train.columns.tolist()
        
        # Store true labels
        predictions['true_labels'] = {
            'validation': y_val.tolist(),
            'test': y_test.tolist()
        }
        
        # First train all individual models
        for model_name, model in self.models.items():
            if model_name != 'ensemble':  # Skip ensemble for now
                print(f"\nTraining {model_name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Get predictions
                val_pred = model.predict(X_val)
                val_prob = model.predict_proba(X_val)[:, 1]
                test_pred = model.predict(X_test)
                test_prob = model.predict_proba(X_test)[:, 1]
                
                # Store predictions
                predictions[model_name] = {
                    'validation': {
                        'predictions': val_pred,
                        'probabilities': val_prob
                    },
                    'test': {
                        'predictions': test_pred,
                        'probabilities': test_prob
                    }
                }
                
                # Calculate metrics
                val_metrics = self.calculate_metrics(y_val, val_pred, val_prob)
                test_metrics = self.calculate_metrics(y_test, test_pred, test_prob)
                
                # Store results
                results[model_name] = {
                    'validation': val_metrics,
                    'test': test_metrics
                }
                
                # Calculate feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance[model_name] = {k: float(v) for k, v in zip(X_train.columns, model.feature_importances_)}
                
                # Print results
                print(f"\n{model_name} Results:")
                print("\nValidation Metrics:")
                for metric, value in val_metrics.items():
                    print(f"{metric}: {value:.4f}")
                print("\nTest Metrics:")
                for metric, value in test_metrics.items():
                    print(f"{metric}: {value:.4f}")
        
        # Now handle ensemble predictions using only Random Forest and XGBoost
        if 'ensemble' in self.models:
            print("\nGenerating ensemble predictions from Random Forest and XGBoost...")
            
            # Get predictions from Random Forest and XGBoost
            val_probs = []
            test_probs = []
            
            for model_name in ['random_forest', 'xgboost']:
                if model_name in predictions:
                    val_probs.append(predictions[model_name]['validation']['probabilities'])
                    test_probs.append(predictions[model_name]['test']['probabilities'])
            
            # Average the probabilities
            val_ensemble_prob = np.mean(val_probs, axis=0)
            test_ensemble_prob = np.mean(test_probs, axis=0)
            
            # Convert probabilities to predictions using 0.5 threshold
            val_ensemble_pred = (val_ensemble_prob >= 0.5).astype(int)
            test_ensemble_pred = (test_ensemble_prob >= 0.5).astype(int)
            
            # Store ensemble predictions
            predictions['ensemble'] = {
                'validation': {
                    'predictions': val_ensemble_pred,
                    'probabilities': val_ensemble_prob
                },
                'test': {
                    'predictions': test_ensemble_pred,
                    'probabilities': test_ensemble_prob
                }
            }
            
            # Calculate ensemble metrics
            val_metrics = self.calculate_metrics(y_val, val_ensemble_pred, val_ensemble_prob)
            test_metrics = self.calculate_metrics(y_test, test_ensemble_pred, test_ensemble_prob)
            
            # Store ensemble results
            results['ensemble'] = {
                'validation': val_metrics,
                'test': test_metrics
            }
            
            # Print ensemble results
            print("\nEnsemble Results (Random Forest + XGBoost):")
            print("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                print(f"{metric}: {value:.4f}")
            print("\nTest Metrics:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Save results for visualization
        self.save_results_for_streamlit(results, predictions, feature_importance, metadata)
        
        return results

    def save_results_for_streamlit(self, results: Dict, predictions: Dict, feature_importance: Dict, metadata: Dict):
        """Save model results, feature importance, and metadata for Streamlit visualization."""
        # Create streamlit_data directory if it doesn't exist
        data_dir = Path('streamlit_data')
        data_dir.mkdir(exist_ok=True)
        
        # Save model results
        with open(data_dir / 'model_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save feature importance
        with open(data_dir / 'feature_importance.json', 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        # Save metadata
        with open(data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save predictions to JSON
        with open(data_dir / 'model_predictions.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_predictions = {}
            for model_name, model_preds in predictions.items():
                if model_name == 'true_labels':
                    json_predictions[model_name] = model_preds
                else:
                    json_predictions[model_name] = {
                        'validation': {
                            'predictions': model_preds['validation']['predictions'].tolist(),
                            'probabilities': model_preds['validation']['probabilities'].tolist()
                        },
                        'test': {
                            'predictions': model_preds['test']['predictions'].tolist(),
                            'probabilities': model_preds['test']['probabilities'].tolist()
                        }
                    }
            json.dump(json_predictions, f, indent=4)
        
        # Save predictions for each model to CSV
        for model_name in self.models.keys():
            # Create a new DataFrame with only the columns used during training, in the correct order
            X = pd.DataFrame(index=self.test_df.index)
            for col in self.training_feature_names:  # Use the stored feature names
                X[col] = self.test_df.get(col, np.nan)
            X = X.fillna(0)
            
            # Use the original predictions instead of recalculating
            self.test_df[f'{model_name}_prediction'] = predictions[model_name]['test']['predictions']
            self.test_df[f'{model_name}_probability'] = predictions[model_name]['test']['probabilities']
            
            # Save predictions to CSV
            self.test_df.to_csv(data_dir / f'{model_name}_predictions.csv')

    def plot_feature_importance(self, model_name='xgboost', feature_names=None):
        """
        Plot feature importance 
        
        Args:
            model_name (str): Name of the model to plot importance for
            feature_names (list): List of feature names in the same order as used in training
        """
        if model_name not in self.feature_importance:
            print(f"No feature importance available for {model_name}")
            return
        
        importance = self.feature_importance[model_name]
        
        if feature_names is None:
            print("Warning: No feature names provided. Using generic feature names.")
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        elif len(feature_names) != len(importance):
            print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features ({len(importance)})")
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort features by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Feature Importance - {model_name}', pad=20)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        
        # Add value labels on the bars
        for i, v in enumerate(importance_df['importance']):
            plt.text(v, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed importance scores
        print("\nFeature Importance Scores:")
        print("========================")
        for idx, row in importance_df.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return importance_df

    def perform_hyperparameter_tuning(self, X, y, cv=5, n_iter=20):
        """
        Perform hyperparameter tuning using RandomizedSearchCV for XGBoost
        """
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_dist = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'n_estimators': [100, 200, 300, 400],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        # Initialize XGBoost model
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        
        # Store best parameters
        self.best_params = random_search.best_params_
        print("Best parameters found:", self.best_params)
        print("Best cross-validation score:", -random_search.best_score_)
        
        return random_search.best_estimator_

    def analyze_feature_importance(self, X, y, top_n=10):
        """
        Analyze feature importance using XGBoost
        """
        print("Analyzing feature importance...")
        
        # Train XGBoost model with best parameters if available
        if self.best_params:
            model = xgb.XGBRegressor(**self.best_params, random_state=42)
        else:
            model = xgb.XGBRegressor(random_state=42)
        
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Print detailed importance scores
        print("\nFeature Importance Scores:")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return importance_df

    def train_optimized_model(self, X_train, y_train, X_val, y_val):
        """
        Train model with optimized hyperparameters and cross-validation
        """
        # Perform hyperparameter tuning
        best_model = self.perform_hyperparameter_tuning(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = self.perform_cross_validation(X_train, y_train)
        
        # Train final model with best parameters
        best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        print("\nFinal Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")
        
        # Store the optimized model
        self.models['optimized_xgboost'] = best_model
        
        return best_model, {'mae': mae, 'rmse': rmse, 'r2': r2}

    def analyze_price_patterns(self, df):
        """
        Analyze price patterns by event type and time, focusing on top 20 most frequent events
        """
        # Get top 20 most frequent event names
        top_events = df['event_name'].value_counts().nlargest(20).index
        df_top = df[df['event_name'].isin(top_events)]
        
        # Create a complete month range for all events
        all_months = pd.DataFrame({
            'event_month': range(1, 13)
        })
        
        # Group by event name and month
        price_patterns = df_top.groupby(['event_name', 'event_month'])['price'].agg([
            'mean',
            'std',
            'min',
            'max',
            'count'
        ]).reset_index()
        
        # Create heatmap data
        heatmap_data = pd.crosstab(df_top['event_name'], df_top['event_month'])
        
        # Create figure with three subplots
        plt.figure(figsize=(15, 25))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        
        # Plot 1: Average Price by Event Name and Month
        for event_name in top_events:
            event_data = price_patterns[price_patterns['event_name'] == event_name]
            
            # Create complete month range for this event
            event_months = pd.merge(
                all_months,
                event_data,
                on='event_month',
                how='left'
            )
            
            # Plot the line
            line = ax1.plot(event_months['event_month'], event_months['mean'], 
                          label=event_name, marker='o', linewidth=2)
            
            # Add count annotations for non-zero months
            for idx, row in event_months.iterrows():
                if not pd.isna(row['mean']):
                    ax1.annotate(f"n={int(row['count'])}", 
                               (row['event_month'], row['mean']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', fontsize=8)
        
        # Set x-axis to show all months from 1 to 12
        ax1.set_xlim(0.5, 12.5)  # Add some padding
        ax1.set_xticks(range(1, 13))  # Show all months
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        ax1.set_title('Average Price by Event Name and Month (Top 20 Events)', fontsize=12, pad=20)
        ax1.set_xlabel('Month', fontsize=10)
        ax1.set_ylabel('Average Price ($)', fontsize=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap
        # Ensure all months are present in the heatmap data
        for month in range(1, 13):
            if month not in heatmap_data.columns:
                heatmap_data[month] = 0
        heatmap_data = heatmap_data.sort_index(axis=1)  # Sort columns by month
        
        sns.heatmap(heatmap_data, 
                   annot=True,  # Show numbers in cells
                   fmt='d',     # Format as integers
                   cmap='YlOrRd',  # Yellow to Orange to Red colormap
                   ax=ax2)
        
        ax2.set_title('Number of Listings by Event and Month (Top 20 Events)', fontsize=12, pad=20)
        ax2.set_xlabel('Month', fontsize=10)
        ax2.set_ylabel('Event Name', fontsize=10)
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax2.set_xticklabels(month_labels)
        
        # Rotate event names for better readability
        plt.setp(ax2.get_yticklabels(), rotation=0)
        
        # Plot 3: Average Prices
        event_summary = df_top.groupby('event_name')['price'].agg(['mean', 'std', 'count']).reset_index()
        event_summary = event_summary.sort_values('mean', ascending=False)
        
        bars = ax3.bar(event_summary['event_name'], event_summary['mean'], 
                      yerr=event_summary['std'], capsize=5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom')
        
        ax3.set_title('Average Price by Event Name (Top 20 Events)', fontsize=12, pad=20)
        ax3.set_xlabel('Event Name', fontsize=10)
        ax3.set_ylabel('Average Price ($)', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nPrice Pattern Summary (Top 20 Events):")
        print("====================================")
        for event_name in event_summary['event_name']:
            event_data = df_top[df_top['event_name'] == event_name]
            print(f"\n{event_name}:")
            print(f"Average Price: ${event_data['price'].mean():.2f}")
            print(f"Price Range: ${event_data['price'].min():.2f} - ${event_data['price'].max():.2f}")
            print(f"Number of Listings: {len(event_data)}")
            # Print month distribution
            month_counts = event_data['event_month'].value_counts().sort_index()
            print("Month Distribution:")
            for month, count in month_counts.items():
                print(f"  Month {month}: {count} listings")
        
        return price_patterns, heatmap_data

    def save_best_model(self, model_name='optimized_xgboost', filepath='best_model.pkl'):
        """
        Save the best performing model to a file for later use
        
        Args:
            model_name (str): Name of the model to save (default: 'optimized_xgboost')
            filepath (str): Path where to save the model file
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.models[model_name],
                'feature_importance': self.feature_importance.get(model_name),
                'best_params': self.best_params,
                'scalers': self.scalers
            }, f)
        
        print(f"Model {model_name} saved successfully to {filepath}")
        
    def calculate_metrics(self, y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }

    def calculate_days_until_event_probabilities(self, df, model_name='xgboost'):
        """
        Calculate average predicted probability of a good deal for each days_until_event value.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            model_name (str): Name of the model to use for predictions
            
        Returns:
            pd.DataFrame: DataFrame with days_until_event and average probability
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        # Get predictions for all data points
        if model_name == 'logistic_regression' and f'{model_name}_scaler' in self.scalers:
            X = self.scalers[f'{model_name}_scaler'].transform(df[self.feature_cols])
            probabilities = self.models[model_name].predict_proba(X)[:, 1]
        else:
            # For models that don't require scaling or don't have a scaler
            probabilities = self.models[model_name].predict_proba(df[self.feature_cols])[:, 1]
        
        # Create DataFrame with days_until_event and probabilities
        prob_df = pd.DataFrame({
            'days_until_event': df['days_until_event'],
            'probability': probabilities
        })
        
        # Group by days_until_event and calculate mean probability
        avg_probabilities = prob_df.groupby('days_until_event')['probability'].agg([
            'mean',
            'std',
            'count'
        ]).reset_index()
        
        # Sort by days_until_event
        avg_probabilities = avg_probabilities.sort_values('days_until_event')
        
        return avg_probabilities

# Example usage:
def main():
    # Load your data
    # df = pd.read_csv('your_ticket_data.csv')
    
    # Initialize analyzer
    analyzer = TicketPriceAnalyzer()
    
    # This is where you load and process your actual data
    print("Ticket Price Analyzer initialized!")
    print("Next steps:")
    print("1. Load your CSV data")
    print("2. Run: df_prepared = analyzer.load_and_prepare_data(df)")
    print("3. Run: X, y, feature_cols = analyzer.prepare_features_target(df_prepared)")
    print("4. Split data and train models")
    print("5. Generate recommendations")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()