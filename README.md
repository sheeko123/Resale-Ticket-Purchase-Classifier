# Ticket Price Analyzer Dashboard

A machine learning-powered dashboard for analyzing and predicting good deals in the secondary ticket market.

## Overview

This dashboard provides insights into ticket pricing trends and helps identify optimal purchase windows for events at Pier 17, NYC. It uses machine learning to classify "good deals" based on historical pricing data and market dynamics.

## Features

- Interactive price analysis visualization
- Model performance metrics and comparisons
- Feature importance analysis
- Savings calculator
- Real-time deal probability predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ticket-price-analyzer.git
cd ticket-price-analyzer
```

2. Install the required packages:
```bash
pip install -r streamlit_requirements.txt
```

3. Run the dashboard:
```bash
streamlit run src/streamlit_app.py
```

## Project Structure

```
ticket-price-analyzer/
├── src/
│   └── streamlit_app.py
├── streamlit_data/
│   ├── model_metrics.json
│   ├── feature_importance.json
│   ├── model_predictions.json
│   ├── metadata.json
│   └── days_until_event_probabilities.json
├── streamlit_requirements.txt
└── README.md
```

## Data Sources

- Venue: Pier 17, NYC
- Data Source: SeatGeek.io
- Time Period: 2022-2024
- Dataset Size: 27,000 total listings across 170 unique events

## Key Metrics

- Average Savings: $23.46 per ticket
- Optimal Purchase Window: 10 days before event
- Model Performance:
  - Precision: 62.3%
  - Recall: 77.6%
  - AUC: 0.88 