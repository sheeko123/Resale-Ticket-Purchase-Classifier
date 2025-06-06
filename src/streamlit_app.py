import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from calculate_savings import calculate_savings
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Ticket Price Analyzer Dashboard",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #FF8C00;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --border-color: #E0E0E0;
        --success-color: #28A745;
        --warning-color: #FFC107;
        --danger-color: #DC3545;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 2rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-top: 0;
        text-align: center;
        color: var(--primary-color);
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 2rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
    }
    
    /* Cards and Containers */
    .stCard {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease;
    }
    
    .stCard:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    .stMetric {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.8;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: var(--background-color);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: var(--text-color);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Charts and Graphs */
    .stPlotlyChart {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    
    /* Data Tables */
    .stDataFrame {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    
    /* Form Elements */
    .stTextInput, .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    .stTextInput > div, .stNumberInput > div, .stSelectbox > div {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #1976D2;
        transform: translateY(-1px);
    }
    
    /* Info Boxes */
    .stInfo {
        background-color: #E3F2FD;
        border-left: 4px solid var(--primary-color);
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: #E8F5E9;
        border-left: 4px solid var(--success-color);
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Warning Messages */
    .stWarning {
        background-color: #FFF8E1;
        border-left: 4px solid var(--warning-color);
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Error Messages */
    .stError {
        background-color: #FFEBEE;
        border-left: 4px solid var(--danger-color);
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Code Blocks */
    code {
        background-color: #F5F5F5;
        color: var(--text-color);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Fira Code', monospace;
        font-size: 0.9em;
    }
    
    /* Horizontal Rules */
    hr {
        border-color: var(--border-color);
        opacity: 0.5;
        margin: 2rem 0;
    }
    
    /* Grid System */
    .row-widget.stHorizontal {
        gap: 1.5rem;
        margin: 0 -0.75rem;
    }
    
    .row-widget.stHorizontal > div {
        padding: 0 0.75rem;
    }
    
    /* Section spacing */
    .element-container {
        margin-bottom: 1.5rem;
    }
    
    /* Remove Streamlit's default blue highlight */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Remove blue highlight from selected elements */
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
    }
    
    /* Remove blue highlight from hover states */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E3F2FD !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #BDBDBD;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Load all the saved JSON files from the streamlit_data directory"""
    data_dir = Path('streamlit_data')
    
    try:
        with open(data_dir / 'model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        with open(data_dir / 'feature_importance.json', 'r') as f:
            feature_importance = json.load(f)
        
        with open(data_dir / 'model_predictions.json', 'r') as f:
            predictions = json.load(f)
        
        with open(data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        with open(data_dir / 'days_until_event_probabilities.json', 'r') as f:
            days_until_event_probs = json.load(f)
        
        return metrics, feature_importance, predictions, metadata, days_until_event_probs
    except FileNotFoundError as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def create_metrics_comparison_chart(metrics):
    """Create a grouped bar chart comparing model metrics"""
    # Define the metrics to plot
    metrics_to_plot = ['f1', 'precision', 'recall', 'auc']
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for each model
    colors = {
        'random_forest': '#1f77b4',  # blue
        'gradient_boost': '#2ca02c',  # green
        'xgboost': '#ff7f0e',  # orange
        'logistic_regression': '#d62728',  # red
        'ensemble': '#9467bd'  # purple
    }
    
    # Add traces for each model
    for model_name in metrics.keys():
        # Add first metric with showlegend=True, rest with showlegend=False
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(go.Bar(
                name=model_name.replace('_', ' ').title(),
                x=[metric],
                y=[metrics[model_name]['test'][metric]],
                marker_color=colors[model_name],
                showlegend=i == 0  # Only show legend for first metric of each model
            ))
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        showlegend=True
    )
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create a confusion matrix plot for a model"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with a clean style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(3, 2.4))  # Doubled size
    
    # Create heatmap with minimal styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'],
                cbar=False)
    
    # Minimal customization
    plt.title(model_name.replace('_', ' ').title(), pad=1, fontsize=10)
    ax.set_xlabel('Pred', fontsize=8, labelpad=0)
    ax.set_ylabel('True', fontsize=8, labelpad=0)
    
    # Remove all padding and extra space
    plt.tight_layout(pad=0.1)
    
    return fig

def create_feature_importance_plot(feature_importance, selected_model=None):
    """Create a feature importance plot using Plotly."""
    # Create figure
    fig = go.Figure()
    
    # Define colors for each model
    colors = {
        'random_forest': '#1f77b4',  # blue
        'gradient_boost': '#2ca02c',  # green
        'xgboost': '#ff7f0e',  # orange
        'ensemble': '#9467bd'  # purple
    }
    
    # Create DataFrame for feature importance
    importance_data = []
    for model_name, model_importance in feature_importance.items():
        for feature, importance in model_importance.items():
            importance_data.append({
                'Model': model_name,
                'Feature': feature,
                'Importance': importance
            })
    
    importance_df = pd.DataFrame(importance_data)
    
    if selected_model and selected_model in feature_importance:
        # Show only selected model
        model_data = importance_df[importance_df['Model'] == selected_model]
        model_data = model_data.sort_values('Importance', ascending=False)
        
        fig.add_trace(go.Bar(
            name=selected_model.replace('_', ' ').title(),
            x=model_data['Feature'],
            y=model_data['Importance'],
            marker_color=colors.get(selected_model, '#1f77b4')
        ))
        
        title = f'Feature Importance - {selected_model.replace("_", " ").title()}'
    else:
        # Show all models
        # Sort features by average importance across all models
        avg_importance = importance_df.groupby('Feature')['Importance'].mean()
        sorted_features = avg_importance.sort_values(ascending=False).index
        
        # Add traces for each model
        for model_name in feature_importance.keys():
            model_data = importance_df[importance_df['Model'] == model_name]
            model_data = model_data.set_index('Feature').reindex(sorted_features)
            
            fig.add_trace(go.Bar(
                name=model_name.replace('_', ' ').title(),
                x=model_data.index,
                y=model_data['Importance'],
                marker_color=colors.get(model_name, '#1f77b4')
            ))
        
        title = 'Feature Importance Comparison'
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Features',
        yaxis_title='Importance',
        barmode='group' if not selected_model else 'stack',
        showlegend=True,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        height=600,
        margin=dict(b=150)  # Increase bottom margin for rotated labels
    )
    
    return fig, importance_df

def create_probability_vs_days_chart(days_until_event_probs, model_name='xgboost'):
    """Create a line chart showing average predicted probability vs days until event"""
    if model_name not in days_until_event_probs:
        st.error(f"Model {model_name} not found in the data")
        return None
        
    # Get data for the specified model
    model_data = days_until_event_probs[model_name]
    
    # Create figure
    fig = go.Figure()
    
    # Add main line for mean probability
    fig.add_trace(go.Scatter(
        x=model_data['days_until_event'],
        y=model_data['mean_probability'],
        mode='lines+markers',
        name='Mean Probability',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8)
    ))
    
    # Add confidence interval
    upper_bound = np.array(model_data['mean_probability']) + np.array(model_data['std_probability'])
    lower_bound = np.array(model_data['mean_probability']) - np.array(model_data['std_probability'])
    
    fig.add_trace(go.Scatter(
        x=model_data['days_until_event'] + model_data['days_until_event'][::-1],
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Average Predicted Probability of Good Deal vs Days Until Event<br><span style="font-size:0.8em">{model_name.replace("_", " ").title()} Model</span>',
        xaxis_title='Days Until Event',
        yaxis_title='Probability of Good Deal',
        xaxis=dict(
            range=[-2, 62],  # Set range from -2 to 62 to show 0-60 with padding
            dtick=10,  # Show tick marks every 10 days
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            range=[0, 1],
            tickformat='.2f',
            gridcolor='lightgray',
            gridwidth=1
        ),
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def load_visualization_data():
    """Load visualization data from JSON file"""
    try:
        with open('streamlit_data/visualization_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Visualization data not found. Please run the model training first.")
        return None

def create_average_price_chart(viz_data, days_range=(0, 60)):
    """Create a line chart showing average price vs days until event for all events"""
    # Get all events data
    event_data = viz_data.get('event_stats', {})
    if not event_data:
        return None
        
    # Create figure
    fig = go.Figure()
    
    # Add a line for each event
    for event_name, event_stats in event_data.items():
        days = list(event_stats['prices'].keys())
        prices = list(event_stats['prices'].values())
        counts = list(event_stats['counts'].values())
        
        # Only include points with sufficient data
        valid_indices = [i for i, count in enumerate(counts) if count >= 5]
        if valid_indices:
            days = [days[i] for i in valid_indices]
            prices = [prices[i] for i in valid_indices]
            counts = [counts[i] for i in valid_indices]
            
            fig.add_trace(go.Scatter(
                x=days,
                y=prices,
                mode='lines+markers',
                name=event_name,
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate=(
                    "Event: %{fullData.name}<br>"
                    "Days Until Event: %{x}<br>"
                    "Average Price: $%{y:.2f}<br>"
                    "Sample Count: %{customdata}<extra></extra>"
                ),
                customdata=counts
            ))
    
    # Update layout
    fig.update_layout(
        title='Average Ticket Price vs Days Until Event<br><span style="font-size:0.8em">All Events</span>',
        xaxis_title='Days Until Event',
        yaxis_title='Average Price ($)',
        xaxis=dict(
            range=days_range,
            dtick=10,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1
        ),
        height=500,
        margin=dict(t=80, b=50, l=50, r=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        hovermode='x unified'
    )
    
    return fig

def create_days_until_event_tab():
    """Create tab for days until event vs good deal probability visualization"""
    # Load visualization data
    viz_data = load_visualization_data()
    if viz_data is None:
        st.warning("No visualization data available.")
        return

    # Dashboard Title and Description
    st.title("Price Analysis Dashboard")
    st.markdown("""
    This dashboard visualizes ticket pricing trends and deal probabilities based on the days until an event. 
    Select a model to explore trends and key statistics.
    """)

    # Model selection - exclude event_stats from the options
    model_names = [key for key in viz_data.keys() if key != 'event_stats']
    # Find the index of 'ensemble' in the model names
    ensemble_index = model_names.index('ensemble') if 'ensemble' in model_names else 0
    selected_model = st.selectbox(
        "Select Model",
        model_names,
        index=ensemble_index
    )

    if selected_model:
        model_data = viz_data[selected_model]
        
        # Create DataFrame for line chart
        line_df = pd.DataFrame({
            'Days Until Event': model_data['line_chart']['days_until_event'],
            'Probability': model_data['line_chart']['mean_probability'],
            'Std Dev': model_data['line_chart']['std_probability'],
            'Sample Count': model_data['line_chart']['count']
        })
        
        # Check if data is available
        if line_df.empty:
            st.warning("No data available for this model.")
            return
        
        # Create line chart
        fig = go.Figure()
        
        # Add main line with professional color
        fig.add_trace(go.Scatter(
            x=line_df['Days Until Event'],
            y=line_df['Probability'],
            mode='lines+markers',
            name='Mean Probability',
            line=dict(color='#0066CC', width=2),
            marker=dict(size=8)
        ))
        
        # Add 30% threshold line
        fig.add_hline(
            y=0.3,
            line_dash="dash",
            line_color="red",
            annotation_text="30% Threshold",
            annotation_position="bottom right"
        )
        
        # Update layout with professional template and font
        fig.update_layout(
            title='Probability of Good Deal by Days Until Event',
            xaxis_title='Days Until Event',
            yaxis_title='Probability of Good Deal',
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template='plotly_white',
            font_family="Arial"
        )
        
        # Add hover template for better readability with upper/lower bounds
        fig.update_traces(
            hovertemplate=(
                "Days Until Event: %{x}<br>"
                "Probability: %{y:.2%}<br>"
                "Upper Bound: %{customdata[0]:.2%}<br>"
                "Lower Bound: %{customdata[1]:.2%}<br>"
                "Sample Count: %{customdata[2]:,}<extra></extra>"
            ),
            customdata=np.column_stack((
                line_df['Probability'] + line_df['Std Dev'],
                line_df['Probability'] - line_df['Std Dev'],
                line_df['Sample Count']
            ))
        )
        
        # Display the probability plot
        st.subheader("Deal Probability Trend")
        # Add slider to adjust days_until_event axis range for probability chart
        max_days = 200  # Set a reasonable maximum
        days_range = st.slider("Days Until Event Range", 0, max_days, (0, 60), step=10, key="probability_slider")
        fig.update_layout(xaxis=dict(range=days_range))
        st.plotly_chart(fig, use_container_width=True)
        
        # Add average price chart
        st.subheader("Average Price Trends")
        # Add slider to adjust days_until_event axis range
        max_days = 200  # Set a reasonable maximum
        days_range = st.slider("Days Until Event Range", 0, max_days, (0, 60), step=10, key="price_slider")
        price_fig = create_average_price_chart(viz_data, days_range=days_range)
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
        else:
            st.warning("Price trend data not available.")
        
        # Add statistics section with key metrics
        st.subheader("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Filter for days with 5 or more samples
            reliable_data = line_df[line_df['Sample Count'] >= 5]
            if not reliable_data.empty:
                best_day = reliable_data.loc[reliable_data['Probability'].idxmax(), 'Days Until Event']
                best_prob = reliable_data['Probability'].max()
                st.metric(
                    "Best Time to Buy",
                    f"{best_day} days",
                    f"Probability: {best_prob:.1%}"
                )
                st.caption("(Of days with 5 or more samples)")
            else:
                st.metric(
                    "Best Time to Buy",
                    "No reliable data",
                    "Need more samples"
                )
                st.caption("(Of days with 5 or more samples)")
        
        with col2:
            # Filter for days within 30 days with 5 or more samples
            reliable_data_30 = line_df[(line_df['Sample Count'] >= 5) & (line_df['Days Until Event'] <= 30)]
            if not reliable_data_30.empty:
                best_day_30 = reliable_data_30.loc[reliable_data_30['Probability'].idxmax(), 'Days Until Event']
                best_prob_30 = reliable_data_30['Probability'].max()
                st.metric(
                    "Best Time to Buy (<30 days)",
                    f"{best_day_30} days",
                    f"Probability: {best_prob_30:.1%}"
                )
                st.caption("(Within 30 days)")
            else:
                st.metric(
                    "Best Time to Buy (<30 days)",
                    "No reliable data",
                    "Need more samples"
                )
                st.caption("(Within 30 days)")
        
        with col3:
            avg_prob = line_df['Probability'].mean()
            std_prob = line_df['Probability'].std()
            st.metric(
                "Average Probability",
                f"{avg_prob:.1%}",
                f"Std Dev: {std_prob:.1%}"
            )
        
        with col4:
            total_samples = line_df['Sample Count'].sum()
            unique_days = len(line_df)
            st.metric(
                "Total Samples",
                f"{total_samples:,}",
                f"Unique Days: {unique_days}"
            )
        
        # Add analysis text with improved formatting
        st.markdown("""
        ### Analysis of Deal Probability Trends
        
        #### Early Booking Window (170-200 Days)
        The analysis reveals that the highest probability of securing a good deal occurs in the early booking window, approximately 170-200 days before the event. However, this observation comes with an important caveat: the data points in this range are limited, with only 8 instances recorded at the 187-day mark. This small sample size suggests that while the probability appears high, the finding should be interpreted with caution.
        
        #### Mid-Term Trend (50-170 Days)
        Following the initial peak, the probability of finding a good deal decreases. There is a lot of volatility in the probability of finding a good deal in the mid-term window untill it reaches
        around the 74 day mark where it rises to 50%. This then begins to level out around 30% as the days progress.
        
        #### Short-Term Window (50-10 Days)
        The probability exhibits a stabilization after the 44-day mark .This stabilization suggests a more predictable pricing environment the several weeks before the event.
                    With 18 days till the event there is a the final peak of 38% before dropping at 14 days to 26% there is another rebound 8-12 days where the final peak of finding a good deal occurs.
                   
        #### Final Days (10-0 Days)
        The final 10 days before the event show significant changes in the market dynamics:
        - Sample count increases substantially, reflecting higher trading activity
        - Upper and lower bounds show increased variability (ranging from 4% to 70%)
        - This period exhibits the highest uncertainty in deal probability
        - The model suggests purchasing a ticket 8-10 days before the event or waiting until the last minute to purchase a ticket to have the highest probability of finding a good deal.
        
        
        This analysis underscores the importance of considering both the probability of finding a good deal and the associated uncertainty when making purchasing decisions at different time points before an event.
        """)

def create_roc_curves(metrics, predictions):
    """Create ROC curves for all models"""
    # Get true labels
    y_true = predictions['true_labels']['test']
    
    # Create figure
    fig = go.Figure()
    
    # Define colors for each model
    colors = {
        'random_forest': '#1f77b4',  # blue
        'gradient_boost': '#2ca02c',  # green
        'xgboost': '#ff7f0e',  # orange
        'logistic_regression': '#d62728',  # red
        'ensemble': '#9467bd'  # purple
    }
    
    # Add ROC curve for each model
    for model_name in metrics.keys():
        try:
            # Get probabilities for the model
            y_prob = predictions[model_name]['test']['probabilities']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = metrics[model_name]['test']['auc']
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})",
                line=dict(color=colors.get(model_name, '#1f77b4'))
            ))
        except Exception as e:
            st.error(f"Could not create ROC curve for {model_name}: {str(e)}")
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def create_savings_calculator_tab():
    """Create tab for savings calculator"""
    st.header("Savings Calculator")
    
    # Load data
    metrics, feature_importance, predictions, metadata, days_until_event_probs = load_data()
    if metrics is None:
        st.warning("No data available for calculations.")
        return
    
    # Get ensemble model metrics
    ensemble_metrics = metrics.get('ensemble', {}).get('test', {})
    if not ensemble_metrics:
        st.warning("Ensemble model metrics not available.")
        return
    
    # Calculate key metrics
    default_precision = ensemble_metrics.get('precision', 0.701)
    default_recall = ensemble_metrics.get('recall', 0.716)
    
    # Get average prices from metadata with default values
    default_avg_good = metadata.get('avg_good_price', 63.04)
    default_avg_bad = metadata.get('avg_bad_price', 100.70)
    default_avg_all = metadata.get('avg_all_price', 92.84)
    
    if default_avg_good == 0 or default_avg_bad == 0:
        st.warning("Price data not available. Using default values for demonstration.")
        default_avg_good = 63.04
        default_avg_bad = 100.70
        default_avg_all = 92.84
    
    # Create calculator interface
    st.markdown("""
    This calculator helps you estimate potential savings when using our model to identify good deals.
    Adjust the parameters below to see how different scenarios affect your ticket purchasing power.
    """)
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        precision = st.slider(
            "Model Precision",
            min_value=0.0,
            max_value=1.0,
            value=default_precision,
            step=0.01,
            help="The model's precision (TP / (TP + FP))"
        )
        
        recall = st.slider(
            "Model Recall",
            min_value=0.0,
            max_value=1.0,
            value=default_recall,
            step=0.01,
            help="The model's recall (TP / (TP + FN))"
        )
    
    with col2:
        st.subheader("Price Parameters")
        avg_good_price = st.number_input(
            "Average Good Deal Price ($)",
            min_value=0.0,
            value=default_avg_good,
            step=1.0,
            help="Average price of good deals"
        )
        
        avg_bad_price = st.number_input(
            "Average Bad Deal Price ($)",
            min_value=0.0,
            value=default_avg_bad,
            step=1.0,
            help="Average price of bad deals"
        )
        
        avg_all_price = st.number_input(
            "Average Overall Price ($)",
            min_value=0.0,
            value=default_avg_all,
            step=1.0,
            help="Average price across all listings"
        )
    
    # Annual budget input
    budget = st.number_input(
        "Annual Budget for Tickets ($)",
        min_value=1000,
        value=10000,
        step=1000,
        help="Your annual budget for ticket purchases"
    )
    
    # Calculate results
    def calculate_savings(budget, precision, recall, avg_good_price, avg_bad_price):
        if avg_bad_price == 0:
            return {
                'random': {'total_tickets': 0, 'good_deals': 0, 'total_cost': 0, 'cost_per_ticket': 0},
                'model': {'total_tickets': 0, 'good_deals': 0, 'total_cost': 0, 'cost_per_ticket': 0}
            }
            
        # Calculate how many tickets you can buy with random selection
        random_tickets = int(budget / avg_bad_price)
        random_good_deals = int(random_tickets * 0.2)  # Assuming 20% are good deals
        
        # Calculate how many tickets you can buy with model
        model_tickets = int(budget / (avg_good_price * precision + avg_bad_price * (1 - precision)))
        model_good_deals = int(model_tickets * precision)
        
        # Calculate costs
        random_cost = random_tickets * avg_bad_price
        model_cost = model_tickets * (avg_good_price * precision + avg_bad_price * (1 - precision))
        
        # Calculate cost per ticket
        random_cost_per_ticket = random_cost / random_tickets if random_tickets > 0 else 0
        model_cost_per_ticket = model_cost / model_tickets if model_tickets > 0 else 0
        
        return {
            'random': {
                'total_tickets': random_tickets,
                'good_deals': random_good_deals,
                'total_cost': random_cost,
                'cost_per_ticket': random_cost_per_ticket
            },
            'model': {
                'total_tickets': model_tickets,
                'good_deals': model_good_deals,
                'total_cost': model_cost,
                'cost_per_ticket': model_cost_per_ticket
            }
        }
    
    # Calculate and display results
    analysis = calculate_savings(budget, precision, recall, avg_good_price, avg_bad_price)
    
    # Calculate differences
    total_tickets_diff = analysis['model']['total_tickets'] - analysis['random']['total_tickets']
    good_deals_diff = analysis['model']['good_deals'] - analysis['random']['good_deals']
    cost_diff = analysis['model']['total_cost'] - analysis['random']['total_cost']
    cost_per_ticket_diff = analysis['model']['cost_per_ticket'] - analysis['random']['cost_per_ticket']
    
    # Display results in a nice format
    st.subheader("Annual Ticket Analysis")
    
    # Create metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Tickets Without Model",
            f"{analysis['random']['total_tickets']:,} tickets",
            f"${analysis['random']['cost_per_ticket']:.2f} per ticket",
            help="Number of tickets and cost per ticket with random selection"
        )
    
    with col2:
        st.metric(
            "Tickets With Model",
            f"{analysis['model']['total_tickets']:,} tickets",
            f"${analysis['model']['cost_per_ticket']:.2f} per ticket",
            help="Number of tickets and cost per ticket using the model"
        )
    
    with col3:
        st.metric(
            "Savings per Ticket",
            f"${abs(cost_per_ticket_diff):.2f}",
            f"{'saved' if cost_per_ticket_diff < 0 else 'more'} per ticket",
            help="Average savings per ticket using the model"
        )
    
    # Display detailed breakdown
    st.subheader("Detailed Breakdown")
    
    # Create a DataFrame for the detailed breakdown
    breakdown_data = {
        "Category": [
            "Random Selection",
            "Model Selection",
            "Difference"
        ],
        "Total Tickets": [
            f"{analysis['random']['total_tickets']:,}",
            f"{analysis['model']['total_tickets']:,}",
            f"{total_tickets_diff:+,}"
        ],
        "Good Deals": [
            f"{analysis['random']['good_deals']:,}",
            f"{analysis['model']['good_deals']:,}",
            f"{good_deals_diff:+,}"
        ],
        "Total Cost": [
            f"${analysis['random']['total_cost']:,.2f}",
            f"${analysis['model']['total_cost']:,.2f}",
            f"${cost_diff:+,.2f}"
        ],
        "Cost per Ticket": [
            f"${analysis['random']['cost_per_ticket']:.2f}",
            f"${analysis['model']['cost_per_ticket']:.2f}",
            f"${cost_per_ticket_diff:+,.2f}"
        ]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True)
    
    # Add analysis text
    st.markdown(f"""
    ### Analysis
    
    This calculator shows how many tickets you can buy with your budget, comparing random selection to using our model to identify good deals:
    
    #### Random Selection
    - You can buy {analysis['random']['total_tickets']:,} tickets with your budget
    - On average, {analysis['random']['good_deals']:,} of these will be good deals (20% of total)
    - Total cost: ${analysis['random']['total_cost']:,.2f}
    - Average cost per ticket: ${analysis['random']['cost_per_ticket']:.2f}
    
    #### Model Selection
    - The model helps you identify good deals with {precision:.1%} precision
    - You can buy {analysis['model']['total_tickets']:,} tickets using the model
    - {analysis['model']['good_deals']:,} of these will be good deals
    - Total cost: ${analysis['model']['total_cost']:,.2f}
    - Average cost per ticket: ${analysis['model']['cost_per_ticket']:.2f}
    
    #### Impact of Using the Model
    - Additional tickets: {total_tickets_diff:+,}
    - Additional good deals: {good_deals_diff:+,}
    - Total cost difference: ${cost_diff:+,.2f}
    - Savings per ticket: ${abs(cost_per_ticket_diff):.2f} {'saved' if cost_per_ticket_diff < 0 else 'more'}
    
    The model's performance is measured by:
    - **Precision**: How many of the predicted good deals are actually good deals ({precision:.1%})
    - **Recall**: How many of the actual good deals are correctly identified ({recall:.1%})
    
    Higher precision means fewer false positives (bad deals), while higher recall means more true positives (good deals).
    """)

def create_homepage():
    """Create the homepage with project overview and key metrics"""
    # Project Overview Section
    st.header("Project Overview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Objective
        Build an ML classifier to identify "good deal" tickets in the secondary market, 
        helping consumers make informed purchasing decisions.
        
        ### Business Value
        """)
        
        # Key Performance Metrics
        st.markdown("""
        <div style="background-color: #1E88E5; padding: 20px; border-radius: 10px; color: white;">
            <h4>Key Performance Metrics</h4>
            <p><strong>Average Savings:</strong> $23.46 per ticket</p>
            <p><strong>Optimal Purchase Window:</strong> 10 days before event</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Benefits
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 15px;">
            <h4>Key Benefits</h4>
            <ul>
                <li>23% average savings on ticket purchases</li>
                <li>Data-driven purchase decisions</li>
                <li>Risk-adjusted return on investment</li>
                <li>Practical implementation of ML insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        ### Core Innovation
        
        Our proprietary approach combines:
        
        **1. Time-Aware Analysis**
        - Dynamic market price tracking
        - Real-time deal identification
        - Adaptive value thresholds
        
        **2. Advanced ML Architecture**
        - Temporal feature engineering
        - Event-specific modeling
        - Ensemble prediction system
        """)
    
    # Data Profile Section
    st.header("Data Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset Overview:**
        - 27,000 total listings
        - 170 unique events
        - 3 concert seasons (2022-2024)
        """)
    
    with col2:
        st.markdown("""
        **Source Information:**
        - Venue: Pier 17, NYC
        - Data Source: SeatGeek.io
        - Time Period: 2022-2024
        """)
    
    # Technical Details Section
    st.header("Technical Innovation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Dynamic Target Variable
        ```python
        is_good_deal = (price ≤ 20th_percentile(CurrentMarketPrices))
        ```
        
        **Why it matters:**
        - Realistic "value" definition            
        - Reflects market volatility
        - Based on 7-day time periods
        - Adapts to changing market conditions
        """)
    
    with col2:
        st.markdown("""
        ### Leakage Prevention
        **Strict temporal feature engineering:**
        - Only past-available data used
        - No future information leakage
        
        **Event-wise time-series splitting:**
        - No event overlap across sets
        - Maintains temporal integrity
        - Preserves real-world prediction scenario
        """)
    
   

def main():
    """Main function to run the Streamlit app"""
    # Display title at the very top
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Resale Ticket Value Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    metrics, feature_importance, predictions, metadata, days_until_event_probs = load_data()
    
    if metrics is None:
        st.error("Failed to load data. Please check if all required data files are present.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Home",
        "Price Analysis",
        "Model Performance",
        "Savings Calculator"
    ])
    
    with tab1:
        create_homepage()
    
    with tab2:
        create_days_until_event_tab()
    
    with tab3:
        # Model Performance Tab
        st.header("Model Performance Analysis")
        
        # Model Metrics Comparison
        st.subheader("Model Metrics Comparison")
        metrics_fig = create_metrics_comparison_chart(metrics)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Model Selection Summary
        st.markdown("""
        ### Model Selection Summary
        #### Results are modest however there is still value to using the model to identify good deals compared to random selection.
        The ensemble model(XGBoost + Random Forest) has been selected for deployment due to its superior overall performance on the test set. It achieved the highest AUC (0.88), indicating strong overall predictive power, and the highest F1-score (0.69),
                     demonstrating the best balance between precision and recall. Critically, it captures 77.6% of positive cases (recall) and correctly identifies 62.3% of good deals(Precision),
                     significantly outperforming other models in identifying true positives, While accuracy (80.7%) is competitive, 
                    the ensemble's robust recall and AUC make it the optimal choice to minimize missed deals. As missing deals can be the most costly outcome as prices can rise significantly.

        **Key Advantage:**  
        This model provides the most reliable identification of positive cases while maintaining a better precision-recall balance than alternatives like XGBoost (higher recall but much lower precision).
        """)
        
        # Feature Importance
        st.subheader("Feature Importance Analysis")
        
        # Find the index of 'xgboost' in the model names
        model_names = list(feature_importance.keys())
        xgboost_index = model_names.index('xgboost') if 'xgboost' in model_names else 0
        
        # Create model selector for feature importance
        selected_model_fi = st.selectbox(
            "Select Model for Feature Importance",
            options=model_names,
            index=xgboost_index,
            key="feature_importance_model"
        )
        
        # Create feature importance plot
        fig, importance_df = create_feature_importance_plot(feature_importance, selected_model_fi)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add feature importance summary
        st.markdown("""
        ### Feature Importance Summary
        
        **Ensemble Feature Importance (XGBoost + Random Forest)**
        **Top Shared Drivers**
        - `price_change_48h`: Strongest overall (XGB: 19.3%, RF: 11.6%)
        - `price_vs_prev_market`: Critical for market relativity (XGB: 17.0%, RF: 13.6%)
        - `price_vs_expanding_p20`: Higher in RF (RF: 9.0% vs XGB: 4.8%)

        **Model-Specific Focus**
        - **XGBoost**: Recent activity signals
          - `listing_volume_7d`: 4.7%
          - `prev_market_size`: 3.1%
        - **Random Forest**: Event pricing anchors
          - `price_vs_event_median`: 8.5%
          - `price_vs_event_min`: 6.1%
          - `price_vs_event_max`: 4.8%

        **Lower Impact**
        - Longer-term factors: `days_since_first_listing` (RF: 3.3%)
        - Volatility metrics: All <3.5% in both models

        Combines XGBoost's real-time sensitivity with RF's event pricing context.
        """)
        
        # ROC Curves
        st.subheader("ROC Curves")
        roc_fig = create_roc_curves(metrics, predictions)
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # Confusion Matrix Section
        st.subheader("Confusion Matrix")
        
        # Create two columns for the matrix and summary
        col1, col2 = st.columns([1, 1])
        
        with col1:  # Left column for the matrix
            # Model selector
            model_names = list(predictions.keys())
            model_names.remove('true_labels')
            ensemble_index = model_names.index('ensemble') if 'ensemble' in model_names else 0
            selected_model_cm = st.selectbox(
                "Select Model",
                options=model_names,
                index=ensemble_index,
                key="confusion_matrix_model"
            )
            
            # Get predictions
            y_true = predictions['true_labels']['test']
            y_pred = predictions[selected_model_cm]['test']['predictions']
            
            # Create and display confusion matrix
            cm_fig = create_confusion_matrix_plot(y_true, y_pred, selected_model_cm)
            st.pyplot(cm_fig, use_container_width=False)
        
        with col2:  # Right column for the summary
            st.markdown("""
            ### Confusion Matrix Summary

            **True Negatives (TN = 5,073)**: Correctly rejected mediocre deals (avoided overpaying)

            **False Positives (FP = 1,097)**: Purchased mediocre deals misclassified as good (overpayment risk)

            **False Negatives (FN = 536)**: Missed profitable opportunities (good deals incorrectly rejected)

            **True Positives (TP = 1,813)**: Correctly captured high-value deals (optimal purchases)

            ### Key Business Implications

            **Mediocre Deals Purchased (FP = 1,097)**  
            → 1,097 overpayments due to false "good deal" labels = direct revenue erosion

            **Missed Opportunities (FN = 536)**  
            → 536 high-potential deals overlooked = lost profit from undervalued assets

            ### Performance Trade-off Analysis
            The model emphasizes risk avoidance (strong TN performance prevents 5,073 bad purchases) but compromises through:

            - Missed 23% of good deals (FN = 536 / [TP 1,813 + FN 536])
            - 38% false approval rate (FP = 1,097 / [TP 1,813 + FP 1,097])
            """)
    
    with tab4:
        create_savings_calculator_tab()

if __name__ == "__main__":
    main() 