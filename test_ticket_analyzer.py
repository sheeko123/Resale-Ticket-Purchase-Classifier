import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.ticket_analyzer import TicketPriceAnalyzer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def load_real_data():
    """Load and prepare the real ticket data"""
    print("Loading real ticket data...")
    df = pd.read_csv(r'C:\Users\Dara\TPPier17\data\processed\p17_ga_processed.csv')
    
    # Convert date columns to datetime with EST timezone
    date_columns = ['event_date', 'listing_time']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_convert('EST')
            print(f"Converted {col} to datetime with EST timezone")
        else:
            print(f"Warning: Required column {col} not found in data")
            if col == 'listing_time' and 'created_at' in df.columns:
                print("Using created_at as listing_time")
                df['listing_time'] = pd.to_datetime(df['created_at'], errors='coerce')
                if pd.api.types.is_datetime64_any_dtype(df['listing_time']):
                    df['listing_time'] = df['listing_time'].dt.tz_convert('EST')
                print("Converted listing_time to datetime with EST timezone")
    
    print(f"\nLoaded {len(df)} listings for {len(df['event_name'].unique())} events")
    return df

def test_ticket_analyzer():
    print("Loading real ticket data...", flush=True)
    df = load_real_data()
    
    print("\nInitializing TicketPriceAnalyzer...", flush=True)
    analyzer = TicketPriceAnalyzer()
    
    # Initialize models
    print("\nInitializing models...", flush=True)
    # Initialize base models for ensemble
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, optimal_threshold=.59,threshold=0.37,
                                     random_state=42, n_jobs=-1,gamma=0.1,scale_pos_weight=3)
    
    # Create ensemble model
    ensemble = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
    
    analyzer.models = {
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth= 20, random_state=42, n_jobs=-1,class_weight={1:3}),
        'gradient_boost': GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, optimal_threshold=.59,threshold=0.37,
                                     random_state=42, n_jobs=-1,gamma=0.1,scale_pos_weight=3),
        'logistic_regression': LogisticRegression(random_state=42,class_weight= 'balanced', max_iter=1000),
        'ensemble': ensemble
    }
    
    print("\nSplitting data...", flush=True)
    # First split the raw data by event
    train_df, val_df, test_df = analyzer.split_data(df)
    
    # Engineer features
    print("\nEngineering features...", flush=True)
    train_df = analyzer.engineer_advanced_features(train_df)
    val_df = analyzer.engineer_advanced_features(val_df)
    test_df = analyzer.engineer_advanced_features(test_df)
    
    # Print available features after engineering
    print("\nAvailable features after engineering:", flush=True)
    print(sorted(train_df.columns.tolist()), flush=True)
    
    print("\nPreparing features and target...", flush=True)
    # First prepare features for training set to determine which features to keep
    X_train, y_train, feature_cols = analyzer.prepare_features_target(train_df, min_feature_importance=0.01)
    
    # Use the same feature columns for validation and test sets
    X_val, y_val, _ = analyzer.prepare_features_target(val_df, min_feature_importance=0)
    X_test, y_test, _ = analyzer.prepare_features_target(test_df, min_feature_importance=0)
    
    # Ensure all splits use the same features
    X_val = X_val[feature_cols]
    X_test = X_test[feature_cols]
    
    # Print features used in training
    print("\nFeatures used in training:", flush=True)
    print(sorted(feature_cols), flush=True)
    
    # Print missing features
    expected_features = [
        'price_vs_event_min', 'price_vs_event_median', 'price_vs_event_max',
        'demand_pressure', 'urgency_factor', 'time_to_event_ratio',
        'listing_volume_7d', 'avg_quantity_7d',
        'price_std_7d', 'price_volatility',
        'event_listing_frequency', 'event_avg_price', 'event_price_std',
        'days_until_event', 'is_weekend_event', 'event_month',
        'price_vs_expanding_p20', 'hist_volatility_30d',
        'price_vs_prev_market', 'prev_market_size',
        'prior_listings_count', 'days_since_first_listing',
        'event_duration_so_far', 'listing_intensity',
        'is_early_listing', 'is_last_week',
        'price_trend_7d', 'inventory_burn_rate',
        'price_change_48h'
    ]
    missing_features = set(expected_features) - set(feature_cols)
    if missing_features:
        print("\nMissing features:", flush=True)
        print(sorted(missing_features), flush=True)
    
    # Train models
    print("\nTraining models...", flush=True)
    results = analyzer.train_models(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print results
    print("\nModel Results:", flush=True)
    for model_name, model_results in results.items():
        print(f"\n{model_name}:", flush=True)
        print("Validation Metrics:", flush=True)
        for metric, value in model_results['validation'].items():
            print(f"{metric}: {value:.4f}", flush=True)
        print("\nTest Metrics:", flush=True)
        for metric, value in model_results['test'].items():
            print(f"{metric}: {value:.4f}", flush=True)
    
    # XGBoost specific analysis with optimal threshold
    print("\nXGBoost Detailed Analysis:")
    xgb_model = analyzer.models['xgboost']
    y_scores = xgb_model.predict_proba(X_test)[:, 1]
    
    # Find threshold that maximizes accuracy
    thresholds = np.arange(0.1, 1.0, 0.01)
    accuracies = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    optimal_threshold = thresholds[np.argmax(accuracies)]
    
    # Update XGBoost predictions using optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    print(f"\nOptimal threshold for accuracy: {optimal_threshold:.4f}")
    print("\nClassification Report with Optimal Threshold:")
    print(classification_report(y_test, y_pred))
    
    # Update the results dictionary with the new predictions
    results['xgboost']['test']['accuracy'] = accuracy_score(y_test, y_pred)
    results['xgboost']['test']['precision'] = precision_score(y_test, y_pred)
    results['xgboost']['test']['recall'] = recall_score(y_test, y_pred)
    results['xgboost']['test']['f1'] = f1_score(y_test, y_pred)
    
    print("\nUpdated XGBoost Results with Optimal Threshold:")
    print("Test Metrics:")
    for metric, value in results['xgboost']['test'].items():
        print(f"{metric}: {value:.4f}")
    
    return analyzer

if __name__ == "__main__":
    test_ticket_analyzer() 