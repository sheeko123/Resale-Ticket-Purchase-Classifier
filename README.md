# Ticket Price Analyzer Dashboard

A machine learning-powered dashboard for analyzing and predicting good deals in the secondary ticket market.

## Overview

This project uses machine learning to identify below-market resale tickets (“good deals”) on the secondary ticket market. The interactive dashboard helps users:

Spot pricing trends across 170+ events

Understand optimal purchase windows

The model and app are trained and deployed on real resale(StubHub) ticket data from SeatGeek, with a focus on Pier 17 (NYC) events between 2022 and 2024.

## Features

Interactive Price Analysis
Visualize price trends over time and across events

Model Performance Metrics
Evaluate precision, recall, and AUC across different models

Savings Calculator
Estimate potential $ savings per ticket based on model predictions

Feature Importance Visuals
Understand which factors influence good deals (e.g. days to event)

## Dashboard
https://resale-ticket-purchase-classifier.streamlit.app/

##Model Performance

| Metric                  | Score                      |
| ----------------------- | -------------------------- |
| Precision               | 62.3%                      |
| Recall                  | 77.6%                      |
| AUC                     | 0.88                       |
| Avg Savings per Deal    | **\$23.46**                |
| Optimal Purchase Window | **\~10 days before event** |


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ticket-price-analyzer.git
cd ticket-price-analyzer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run src/streamlit_app.py
```
N.B. If wanting to play with the model.
```
Model requirements is in src/model
Along with the ticket_analyzer_model
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
├── requirements.txt
└── README.md
```

## Data Sources

- Venue: Pier 17, NYC
- Data Source: SeatGeek.io
- Time Period: 2022-2024
- Dataset Size: 27,000 total listings across 170 unique events


## Why This Matters
Secondary ticketing is a $1B+ market where consumers often overpay.
This project shows how data science can drive real savings, inform buyer behavior, and uncover pricing inefficiencies—valuable for roles in:

Data analytics (price trend insights)

Data engineering (pipeline + dashboard design)

Data science (classification, feature importance, prediction)


## Future Improvements
Deploy and integrate with live scraped ticket data.
Add new venues.
Improve model generalizability to different event types.


## Contact
Darasheehan101@gmail.com

## Linked in 
https://www.linkedin.com/in/dara-sheehan-942532208/

