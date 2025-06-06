import pandas as pd
import numpy as np
from pathlib import Path

# Define the data path relative to the project root
data_path = Path('data/processed/p17_ga_processed.csv')

# Load the data with error handling
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # If file not found, use default values
    print("Warning: Could not load data file. Using default values.")
    df = pd.DataFrame({
        'event_name': ['default_event'],
        'price': [92.84],  # Average price
        'Good Price': [0]
    })

# Calculate 20th percentile price for each event
price_threshold = df.groupby('event_name')['price'].transform(lambda x: x.quantile(0.2))

# Create Good Price column - 1 if price is below 20th percentile for that event, 0 otherwise
df['Good Price'] = (df['price'] <= price_threshold).astype(int)

# Calculate average prices
avg_all_prices = df['price'].mean()
avg_good_price = df[df['Good Price'] == 1]['price'].mean()
avg_non_good_price = df[df['Good Price'] == 0]['price'].mean()

# Default values if calculations fail
if pd.isna(avg_all_prices):
    avg_all_prices = 92.84
if pd.isna(avg_good_price):
    avg_good_price = 63.04
if pd.isna(avg_non_good_price):
    avg_non_good_price = 100.70

print(f"Average Price: ${avg_all_prices:.2f}")
print(f"Average Good Price: ${avg_good_price:.2f}")
print(f"Average Non-Good Price: ${avg_non_good_price:.2f}")
# Print distribution of Good Price flag
good_price_dist = df['Good Price'].value_counts(normalize=True)
print("\nGood Price Distribution:")
print(good_price_dist)

# Verify it's close to 20%
pct_good_deals = good_price_dist[1] * 100
print(f"\nPercentage of Good Deals: {pct_good_deals:.1f}%")


# From your test set (using Gradient Boost metrics)
precision = 0.701  # TP / (TP + FP)
recall = 0.716     # TP / (TP + FN)

# Calculate false positive rate (FPR)
fpr = 0.299  # 1 - precision

# Average prices (replace with actuals from your data)
avg_all_prices = 92.84         # Average price across all listings
avg_good_price = 63.04         # Avg. price of bottom 20% deals
avg_non_good_price = 100.70    # Avg. price of other listings

def calculate_savings(precision, recall, avg_good, avg_non_good, avg_all, annual_budget=100000):
    """
    Calculate ticket purchasing analysis using the model's predictions.
    
    Args:
        precision (float): Model precision (TP / (TP + FP))
        recall (float): Model recall (TP / (TP + FN))
        avg_good (float): Average price of good deals
        avg_non_good (float): Average price of non-good deals
        avg_all (float): Average price across all listings
        annual_budget (float): Annual budget for ticket purchases
        
    Returns:
        dict: Dictionary containing ticket purchasing metrics
    """
    # Proportion of good deals in the market
    p = 0.2  # 20% of tickets are good deals
    
    # Random Selection
    # Calculate number of tickets we can buy with random selection
    n_random = int(annual_budget / avg_all)
    n_random_good = int(n_random * p)  # Expected good deals
    n_random_bad = n_random - n_random_good
    
    # Calculate costs for random selection
    random_good_cost = n_random_good * avg_good
    random_bad_cost = n_random_bad * avg_non_good
    random_total_cost = random_good_cost + random_bad_cost
    
    # Model Selection
    # Calculate average price of predicted good deals
    avg_pred_good = precision * avg_good + (1 - precision) * avg_non_good
    
    # Calculate number of tickets we can buy with model selection
    n_model = int(annual_budget / avg_pred_good)
    
    # Calculate expected actual good deals from model predictions
    n_model_good = int(n_model * precision)
    n_model_bad = n_model - n_model_good
    
    # Calculate costs for model selection
    model_good_cost = n_model_good * avg_good
    model_bad_cost = n_model_bad * avg_non_good
    model_total_cost = model_good_cost + model_bad_cost
    
    return {
        "random": {
            "total_tickets": n_random,
            "good_deals": n_random_good,
            "bad_deals": n_random_bad,
            "total_cost": random_total_cost,
            "good_cost": random_good_cost,
            "bad_cost": random_bad_cost
        },
        "model": {
            "total_tickets": n_model,
            "good_deals": n_model_good,
            "bad_deals": n_model_bad,
            "total_cost": model_total_cost,
            "good_cost": model_good_cost,
            "bad_cost": model_bad_cost
        },
        "avg_pred_good": avg_pred_good
    }

# Example usage for dashboard integration
if __name__ == "__main__":
    # Example values from the data analysis above
    savings = calculate_savings(
        precision=0.701,
        recall=0.716,
        avg_good=63.04,
        avg_non_good=100.70,
        avg_all=92.84,
        annual_budget=100000
    )
    
    print("\nSavings Analysis:")
    print(f"Baseline Tickets: {savings['random']['total_tickets']}")
    print(f"Model Tickets: {savings['model']['total_tickets']}")
    print(f"Baseline Cost: ${savings['random']['total_cost']:.2f}")
    print(f"Model Cost: ${savings['model']['total_cost']:.2f}")

