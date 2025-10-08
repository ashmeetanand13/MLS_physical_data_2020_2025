import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MLS Physical Performance Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    # Load from GitHub (CORRECTED URL)
    try:
        github_url = "https://raw.githubusercontent.com/ashmeetanand13/MLS_physical_data_2020_2025/main/mls_physical.csv"
        df = pd.read_csv(github_url)
    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None, None
    
    # Convert date columns
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    
    # Ensure Match column is string type and handle NaN values
    df['Match'] = df['Match'].fillna('').astype(str)
    
    # Extract home/away teams from Match column
    # Try different separators that might be used in MLS data
    separators = [' - ', ' vs ', ' vs. ', ' v ', ' @ ']
    match_separated = False
    
    for separator in separators:
        if df['Match'].str.contains(separator, regex=False, na=False).any():
            try:
                match_split = df['Match'].str.split(separator, expand=True)
                if match_split.shape[1] >= 2:
                    df['Home Team'] = match_split[0].str.strip()
                    df['Away Team'] = match_split[1].str.strip()
                    match_separated = True
                    break
            except:
                continue
    
    # If no separator worked, check if team names are in a different pattern
    if not match_separated:
        # Alternative: Check if the Match column contains the team name
        # and determine home/away based on some other logic
        df['Home Team'] = df['Team'].copy()
        df['Away Team'] = df['Team'].copy()
        
        # Try to infer from Match string - often home team is listed first
        for idx, row in df.iterrows():
            match_str = str(row['Match'])
            team_name = str(row['Team'])
            # Check if team name appears at the beginning (likely home) or end (likely away)
            if match_str.startswith(team_name):
                df.at[idx, 'Is Home'] = True
            elif team_name in match_str and not match_str.startswith(team_name):
                df.at[idx, 'Is Home'] = False
            else:
                # Default assumption
                df.at[idx, 'Is Home'] = True
    else:
        df['Is Home'] = df['Team'] == df['Home Team']
    
    # Debug info - print sample to console
    if 'Is Home' in df.columns:
        home_count = df['Is Home'].sum()
        away_count = (~df['Is Home']).sum()
        print(f"Debug: Home games: {home_count}, Away games: {away_count}")
        print(f"Debug: Sample Match values: {df['Match'].head(3).tolist()}")
    
    # Create position benchmarks
    position_benchmarks = calculate_position_benchmarks(df)
    
    return df, position_benchmarks

# Calculate position-specific benchmarks
def calculate_position_benchmarks(df):
    # Key physical metrics for benchmarking
    key_metrics = [
        'Distance P90', 'HSR Distance P90', 'Sprint Distance P90',
        'HI Count P90', 'High Acceleration Count P90', 'High Deceleration Count P90'
    ]
    
    # Calculate total minutes per player first
    player_minutes = df.groupby(['Player', 'Position'])['Minutes'].sum().reset_index()
    player_minutes = player_minutes[player_minutes['Minutes'] >= 450]  # At least 450 total minutes
    
    # Get average metrics for qualified players
    qualified_players = player_minutes['Player'].unique()
    df_filtered = df[df['Player'].isin(qualified_players)].copy()
    
    benchmarks = df_filtered.groupby('Position')[key_metrics].agg(['mean', 'std', 'median'])
    return benchmarks

# Team aggregation functions
def aggregate_team_metrics(df, method='weighted'):
    """Aggregate player metrics to team level"""
    
    physical_cols = [col for col in df.columns if 'P90' in col or 
                    any(x in col for x in ['Distance', 'Count', 'HSR', 'Sprint', 'Acceleration', 'Deceleration'])]
    physical_cols = [col for col in physical_cols if 'P90' not in col and col not in ['Distance 1', 'Distance 2']]
    
    if method == 'weighted':
        # Weight by minutes played
        team_data = []
        for (match_id, team), group in df.groupby(['Match ID', 'Team']):
            weighted_metrics = {}
            total_minutes = group['Minutes'].sum()
            
            for col in physical_cols:
                if col in group.columns:
                    weighted_metrics[col] = (group[col] * group['Minutes']).sum() / total_minutes
            
            weighted_metrics['Match ID'] = match_id
            weighted_metrics['Team'] = team
            weighted_metrics['Total Minutes'] = total_minutes
            team_data.append(weighted_metrics)
        
        return pd.DataFrame(team_data)
    else:
        # Simple average
        return df.groupby(['Match ID', 'Team'])[physical_cols].mean().reset_index()

# Load the data from GitHub
# Load the data from GitHub
with st.spinner("Loading data from GitHub..."):
    df, position_benchmarks = load_data()

# CHECK IF LOADING FAILED - STOP THE APP
if df is None:
    st.error("⚠️ Failed to load data from GitHub. Please check:")
    st.write("1. The GitHub URL is correct")
    st.write("2. The CSV file exists at that location")
    st.write("3. The repository is public")
    st.stop()

# Title and description
st.title("⚽ MLS Physical Performance Analytics Dashboard")
st.markdown("### Comprehensive physical performance analysis across teams, players, and seasons")

# Sidebar filters
st.sidebar.header("🎯 Global Filters")

selected_seasons = st.sidebar.multiselect(
    "Select Seasons",
    options=sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique())[-2:]  # Last 2 years by default
)

selected_teams = st.sidebar.multiselect(
    "Select Teams",
    options=sorted(df['Team'].unique()),
    default=[]
)

min_minutes = st.sidebar.slider(
    "Minimum Total Minutes Played",
    min_value=0,
    max_value=900,
    value=450,
    step=45,
    help="Filter players by total minutes played across all selected games"
)

# Apply filters
df_filtered = df[df['Year'].isin(selected_seasons)]
if selected_teams:
    df_filtered = df_filtered[df_filtered['Team'].isin(selected_teams)]

# Calculate total minutes per player for filtering
player_total_minutes = df_filtered.groupby('Player')['Minutes'].sum()
qualified_players = player_total_minutes[player_total_minutes >= min_minutes].index
df_filtered = df_filtered[df_filtered['Player'].isin(qualified_players)]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Playing Style DNA", 
    "📈 Fatigue & Season Dynamics", 
    "🎯 Performance Intelligence",
    "👤 Player Comparison Tool",
    "🔬 Advanced Performance Intelligence",
    "📈 Physical Style Evolution (5-Year Trends)",
    "🎯 Peak Performance Windows"
])

# Tab 1: Playing Style DNA & Clustering
with tab1:
    st.header("Playing Style DNA & Team Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Team aggregation method selection
        agg_method = st.selectbox(
            "Team Aggregation Method",
            ["weighted", "average"],
            help="Weighted: Accounts for minutes played | Average: Simple mean"
        )
    
    # Aggregate team data
    team_df = aggregate_team_metrics(df_filtered, method=agg_method)
    
    # Merge with original data to get additional info
    team_df = team_df.merge(
        df_filtered[['Match ID', 'Date', 'Season', 'Match']].drop_duplicates(),
        on='Match ID',
        how='left'
    )
    
    # PCA Analysis for Playing Style
    st.subheader("🧬 Team Physical Style Clustering (PCA)")
    
    # Select features for PCA
    pca_features = [
        'Distance', 'HSR Distance', 'Sprint Distance',
        'HI Count', 'High Acceleration Count', 'High Deceleration Count'
    ]
    
    # Prepare data for PCA
    team_season_avg = team_df.groupby(['Team', 'Season'])[pca_features].mean().reset_index()
    
    if len(team_season_avg) > 0:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(team_season_avg[pca_features])
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            X_pca,
            columns=['PC1', 'PC2']
        )
        pca_df['Team'] = team_season_avg['Team']
        pca_df['Season'] = team_season_avg['Season']
        
        # Create interactive PCA plot
        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Team',
            symbol='Season',
            title=f'Team Playing Style Clusters (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
            hover_data=['Team', 'Season'],
            height=500
        )
        
        fig_pca.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Feature importance
        st.subheader("📊 Physical Metrics Contribution to Playing Style")
        
        # Get feature loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=pca_features
        )
        
        # Create loading plot
        fig_loadings = go.Figure()
        
        for i, feature in enumerate(pca_features):
            fig_loadings.add_trace(go.Scatterpolar(
                r=[abs(loadings.loc[feature, 'PC1']), abs(loadings.loc[feature, 'PC2'])],
                theta=['PC1', 'PC2'],
                name=feature,
                fill='toself'
            ))
        
        fig_loadings.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Feature Contributions to Principal Components"
        )
        
        st.plotly_chart(fig_loadings, use_container_width=True)
    
    # Home vs Away Analysis
    st.subheader("🏠 Home vs Away Physical Output")
    
    # Get unique matches first
    unique_matches = df_filtered[['Match ID', 'Match', 'Date', 'Team', 'Is Home']].drop_duplicates()
    
    # Count unique matches (divide by 2 since each match appears twice - once for each team)
    total_unique_matches = len(unique_matches['Match ID'].unique())
    
    # For home/away analysis, we need to aggregate at team-match level first
    match_team_agg = df_filtered.groupby(['Match ID', 'Date', 'Team', 'Is Home']).agg({
        'Distance': 'sum',
        'HSR Distance': 'sum', 
        'Sprint Distance': 'sum',
        'HI Count': 'sum',
        'High Acceleration Count': 'sum',
        'High Deceleration Count': 'sum',
        'Minutes': 'sum'  # Total minutes played by team
    }).reset_index()
    
    # Now group by team and home/away status
    home_away_summary = match_team_agg.groupby(['Team', 'Is Home']).agg({
        'Distance': 'mean',
        'HSR Distance': 'mean',
        'Sprint Distance': 'mean',
        'Match ID': 'count'  # Count of matches
    }).reset_index()
    home_away_summary.rename(columns={'Match ID': 'Match Count'}, inplace=True)
    
    # Display correct diagnostic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Matches", total_unique_matches)
    with col2:
        home_match_count = match_team_agg[match_team_agg['Is Home'] == True]['Match ID'].nunique()
        st.metric("Teams Playing at Home", home_match_count)
    with col3:
        away_match_count = match_team_agg[match_team_agg['Is Home'] == False]['Match ID'].nunique()
        st.metric("Teams Playing Away", away_match_count)
    
    # Show sample Match values for debugging
    with st.expander("Debug: Sample Match Values & Season Info"):
        st.write("First 5 unique Match values:")
        st.write(df_filtered['Match'].unique()[:5])
        
        # Show matches per team per season
        matches_per_season = df_filtered.groupby(['Team', 'Season'])['Match ID'].nunique().reset_index()
        matches_per_season.columns = ['Team', 'Season', 'Matches Played']
        st.write("\nMatches per team per season (sample):")
        st.dataframe(matches_per_season.head(10))
    
    # Only proceed if we have both home and away data
    teams_with_both = []
    for team in home_away_summary['Team'].unique():
        team_data = home_away_summary[home_away_summary['Team'] == team]
        if len(team_data) == 2:  # Has both home and away records
            teams_with_both.append(team)
    
    if teams_with_both:
        # Pivot for comparison
        home_away_pivot = home_away_summary.pivot(index='Team', columns='Is Home', values='Distance').reset_index()
        
        # Fix column names
        home_away_pivot.columns.name = None
        home_away_pivot = home_away_pivot.rename(columns={False: 'Away', True: 'Home'})
        
        # Filter to only teams with both home and away data
        home_away_pivot = home_away_pivot.dropna()
        
        if len(home_away_pivot) > 0:
            # Calculate advantage as percentage difference
            home_away_pivot['Home Advantage %'] = ((home_away_pivot['Home'] - home_away_pivot['Away']) / home_away_pivot['Away'] * 100)
            
            # Also get match counts for context
            match_counts = home_away_summary.pivot(index='Team', columns='Is Home', values='Match Count').reset_index()
            match_counts.columns.name = None
            match_counts = match_counts.rename(columns={False: 'Away Matches', True: 'Home Matches'})
            
            # Merge with the distance data
            home_away_pivot = home_away_pivot.merge(match_counts[['Team', 'Home Matches', 'Away Matches']], on='Team')
            
            # Create the bar chart
            fig_home_away = px.bar(
                home_away_pivot.sort_values('Home Advantage %'),
                x='Home Advantage %',
                y='Team',
                orientation='h',
                title='Home vs Away Distance Coverage Advantage (Team Total per Match)',
                color='Home Advantage %',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                hover_data=['Home Matches', 'Away Matches', 'Home', 'Away']
            )
            
            fig_home_away.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_home_away, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Teams with Home Advantage", 
                         len(home_away_pivot[home_away_pivot['Home Advantage %'] > 0]),
                         f"{len(home_away_pivot[home_away_pivot['Home Advantage %'] > 0])/len(home_away_pivot)*100:.1f}%")
            with col2:
                avg_advantage = home_away_pivot['Home Advantage %'].mean()
                st.metric("Average Home Advantage", 
                         f"{avg_advantage:.2f}%",
                         "Higher at home" if avg_advantage > 0 else "Higher away")
        else:
            st.warning("No teams have both home and away data after filtering.")
    else:
        st.warning("Unable to find teams with both home and away games. Check the Match column format.")
        st.info("Expected format: 'Home Team - Away Team' or similar")

with tab7: 
    # Section H: Peak Performance Windows
        st.subheader("Peak Performance Windows & Seasonal Timing")
        st.markdown("Identify when teams and players hit their physical peak during the season")
        
        # Add month name mapping
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        
        # Ensure Month column exists
        if 'Month' not in df_filtered.columns:
            df_filtered['Month'] = df_filtered['Date'].dt.month
        
        df_filtered['Month Name'] = df_filtered['Month'].map(month_names)
        
        # Check if we have enough temporal data
        unique_months = df_filtered['Month'].nunique()
        
        if unique_months < 3:
            st.warning("Need at least 3 months of data for meaningful peak performance analysis.")
        else:
            peak_metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
                          'High Acceleration Count P90']
            
            # H1: Team Peak Performance Heatmap
            st.markdown("### H1: Team Performance Calendar Heatmap")
            
            heatmap_metric_h = st.selectbox(
                "Select Metric for Heatmap",
                peak_metrics,
                key="peak_heatmap_metric"
            )
            
            # Calculate monthly team averages
            team_month_stats = df_filtered.groupby(['Team', 'Month', 'Month Name'])[heatmap_metric_h].mean().reset_index()
            
            # Pivot for heatmap
            heatmap_peak = team_month_stats.pivot(
                index='Team',
                columns='Month Name',
                values=heatmap_metric_h
            )
            
            # Reorder columns by month number
            month_order = [month_names[m] for m in sorted(df_filtered['Month'].unique())]
            heatmap_peak = heatmap_peak[month_order]
            
            # Create heatmap
            fig_peak_heatmap = px.imshow(
                heatmap_peak,
                labels=dict(x="Month", y="Team", color=heatmap_metric_h),
                x=heatmap_peak.columns,
                y=heatmap_peak.index,
                aspect="auto",
                color_continuous_scale='YlOrRd',
                title=f'{heatmap_metric_h} - Monthly Performance Heatmap (All Seasons Combined)'
            )
            
            fig_peak_heatmap.update_layout(
                height=700,
                xaxis_title="Month",
                yaxis_title="Team"
            )
            
            st.plotly_chart(fig_peak_heatmap, use_container_width=True)
            
            # Identify peak months for each team
            st.markdown("#### Team Peak Months")
            
            team_peak_months = []
            for team in team_month_stats['Team'].unique():
                team_data = team_month_stats[team_month_stats['Team'] == team]
                peak_month_row = team_data.loc[team_data[heatmap_metric_h].idxmax()]
                
                team_peak_months.append({
                    'Team': team,
                    'Peak Month': peak_month_row['Month Name'],
                    'Peak Value': peak_month_row[heatmap_metric_h],
                    'Average Value': team_data[heatmap_metric_h].mean(),
                    'Peak Advantage %': ((peak_month_row[heatmap_metric_h] - team_data[heatmap_metric_h].mean()) / team_data[heatmap_metric_h].mean() * 100)
                })
            
            peak_months_df = pd.DataFrame(team_peak_months).sort_values('Peak Advantage %', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Teams - Highest Peak Advantage**")
                st.dataframe(
                    peak_months_df.head(5)[['Team', 'Peak Month', 'Peak Advantage %']].style.format({
                        'Peak Advantage %': '{:+.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                # Count which months are most common peaks
                peak_month_counts = peak_months_df['Peak Month'].value_counts().reset_index()
                peak_month_counts.columns = ['Month', 'Teams Peaking']
                
                st.markdown("**Most Common Peak Months**")
                st.dataframe(peak_month_counts.head(5), use_container_width=True, hide_index=True)
            
            # H2: Player Peak Timing Distribution
            st.markdown("---")
            st.markdown("### H2: Player Peak Timing Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                player_peak_metric = st.selectbox(
                    "Select Metric for Player Analysis",
                    peak_metrics,
                    key="player_peak_metric"
                )
            
            with col2:
                position_filter_peak = st.selectbox(
                    "Filter by Position Group",
                    ["All Positions"] + sorted(df_filtered['Position Group'].unique().tolist()),
                    key="peak_position_filter"
                )
            
            # Filter data
            df_peak_analysis = df_filtered.copy()
            if position_filter_peak != "All Positions":
                df_peak_analysis = df_peak_analysis[df_peak_analysis['Position Group'] == position_filter_peak]
            
            # Calculate player monthly averages (need minimum games per month)
            player_month_stats = df_peak_analysis.groupby(['Player', 'Month', 'Month Name']).agg({
                player_peak_metric: 'mean',
                'Match ID': 'nunique'
            }).reset_index()
            
            # Only consider months where player had at least 2 games
            player_month_stats = player_month_stats[player_month_stats['Match ID'] >= 2]
            
            # Find each player's peak month
            player_peaks = []
            for player in player_month_stats['Player'].unique():
                player_data = player_month_stats[player_month_stats['Player'] == player]
                
                if len(player_data) >= 3:  # Need at least 3 months of data
                    peak_month_row = player_data.loc[player_data[player_peak_metric].idxmax()]
                    
                    player_peaks.append({
                        'Player': player,
                        'Peak Month': peak_month_row['Month Name'],
                        'Peak Month Num': peak_month_row['Month'],
                        'Peak Value': peak_month_row[player_peak_metric]
                    })
            
            if player_peaks:
                player_peaks_df = pd.DataFrame(player_peaks)
                
                # Create histogram of peak months
                peak_distribution = player_peaks_df['Peak Month'].value_counts().reset_index()
                peak_distribution.columns = ['Month', 'Number of Players']
                
                # Order by month number
                month_to_num = {v: k for k, v in month_names.items()}
                peak_distribution['Month Num'] = peak_distribution['Month'].map(month_to_num)
                peak_distribution = peak_distribution.sort_values('Month Num')
                
                fig_peak_dist = px.bar(
                    peak_distribution,
                    x='Month',
                    y='Number of Players',
                    title=f'Distribution of Player Peak Months - {player_peak_metric}',
                    color='Number of Players',
                    color_continuous_scale='Viridis',
                    height=500
                )
                
                st.plotly_chart(fig_peak_dist, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    most_common_peak = peak_distribution.iloc[0]
                    st.metric(
                        "Most Common Peak Month",
                        most_common_peak['Month'],
                        f"{most_common_peak['Number of Players']} players"
                    )
                
                with col2:
                    early_season_peaks = len(player_peaks_df[player_peaks_df['Peak Month Num'] <= 5])
                    st.metric(
                        "Early Season Peakers (Jan-May)",
                        f"{early_season_peaks}/{len(player_peaks_df)}",
                        f"{early_season_peaks/len(player_peaks_df)*100:.1f}%"
                    )
            else:
                st.warning("Not enough data to analyze player peak timing. Players need at least 3 months with 2+ games.")
            
            # H3: Early Season vs Late Season Performance
            st.markdown("---")
            st.markdown("### H3: Early Season vs Late Season Comparison")
            
            early_late_metric = st.selectbox(
                "Select Metric for Early/Late Comparison",
                peak_metrics,
                key="early_late_metric"
            )
            
            # Define early (March-May) and late (Aug-Oct) for MLS season
            # User can adjust based on their data
            early_months = [3, 4, 5]  # March, April, May
            late_months = [8, 9, 10]  # August, September, October
            
            st.info(f"Early Season: {', '.join([month_names[m] for m in early_months])} | Late Season: {', '.join([month_names[m] for m in late_months])}")
            
            # Calculate team early/late averages
            team_early_late = []
            
            for team in df_filtered['Team'].unique():
                team_data = df_filtered[df_filtered['Team'] == team]
                
                early_data = team_data[team_data['Month'].isin(early_months)]
                late_data = team_data[team_data['Month'].isin(late_months)]
                
                if len(early_data) > 0 and len(late_data) > 0:
                    early_avg = early_data[early_late_metric].mean()
                    late_avg = late_data[early_late_metric].mean()
                    
                    team_early_late.append({
                        'Team': team,
                        'Early Season': early_avg,
                        'Late Season': late_avg,
                        'Difference': late_avg - early_avg,
                        'Trend': 'Late Season Grinder' if late_avg > early_avg else 'Early Season Sprinter'
                    })
            
            if team_early_late:
                early_late_df = pd.DataFrame(team_early_late)
                
                # Create scatter plot
                fig_early_late = px.scatter(
                    early_late_df,
                    x='Early Season',
                    y='Late Season',
                    text='Team',
                    color='Trend',
                    title=f'Early Season vs Late Season Performance - {early_late_metric}',
                    labels={'Early Season': f'Early Season Avg {early_late_metric}', 
                           'Late Season': f'Late Season Avg {early_late_metric}'},
                    color_discrete_map={
                        'Late Season Grinder': '#2ecc71',
                        'Early Season Sprinter': '#e74c3c'
                    },
                    height=600
                )
                
                # Add diagonal line (y=x) for reference
                min_val = min(early_late_df['Early Season'].min(), early_late_df['Late Season'].min())
                max_val = max(early_late_df['Early Season'].max(), early_late_df['Late Season'].max())
                
                fig_early_late.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Equal Performance',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig_early_late.update_traces(textposition='top center', selector=dict(mode='markers+text'))
                
                st.plotly_chart(fig_early_late, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    grinders = len(early_late_df[early_late_df['Trend'] == 'Late Season Grinder'])
                    st.metric(
                        "Late Season Grinders",
                        f"{grinders}/{len(early_late_df)}",
                        f"{grinders/len(early_late_df)*100:.1f}%"
                    )
                
                with col2:
                    biggest_grinder = early_late_df.loc[early_late_df['Difference'].idxmax()]
                    st.metric(
                        "Biggest Late Season Improver",
                        biggest_grinder['Team'],
                        f"+{biggest_grinder['Difference']:.1f}"
                    )
                
                with col3:
                    biggest_sprinter = early_late_df.loc[early_late_df['Difference'].idxmin()]
                    st.metric(
                        "Biggest Early Season Performer",
                        biggest_sprinter['Team'],
                        f"{biggest_sprinter['Difference']:.1f}"
                    )
            else:
                st.warning("Not enough data in early and late season months for comparison.")
            
            # H4: Individual Team Season Arc
            st.markdown("---")
            st.markdown("### H4: Team Season Arc Analysis")
            
            selected_team_arc = st.selectbox(
                "Select Team for Detailed Season Arc",
                options=sorted(df_filtered['Team'].unique()),
                key="team_arc_select"
            )
            
            team_arc_data = df_filtered[df_filtered['Team'] == selected_team_arc]
            
            # Calculate monthly averages for this team
            team_monthly = team_arc_data.groupby(['Month', 'Month Name'])[peak_metrics].mean().reset_index()
            team_monthly = team_monthly.sort_values('Month')
            
            if len(team_monthly) >= 3:
                # Create multi-line chart
                fig_arc = go.Figure()
                
                for metric in peak_metrics:
                    fig_arc.add_trace(go.Scatter(
                        x=team_monthly['Month Name'],
                        y=team_monthly[metric],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig_arc.update_layout(
                    title=f'{selected_team_arc} - Monthly Performance Arc',
                    xaxis_title='Month',
                    yaxis_title='Metric Value',
                    hovermode='x unified',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_arc, use_container_width=True)
                
                # Identify dips and peaks
                st.markdown(f"#### {selected_team_arc} - Monthly Performance Summary")
                
                summary_data = []
                for metric in peak_metrics:
                    peak_month = team_monthly.loc[team_monthly[metric].idxmax()]
                    low_month = team_monthly.loc[team_monthly[metric].idxmin()]
                    
                    summary_data.append({
                        'Metric': metric,
                        'Peak Month': peak_month['Month Name'],
                        'Peak Value': peak_month[metric],
                        'Lowest Month': low_month['Month Name'],
                        'Lowest Value': low_month[metric],
                        'Range': peak_month[metric] - low_month[metric],
                        'Variability %': ((peak_month[metric] - low_month[metric]) / team_monthly[metric].mean() * 100)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                st.dataframe(
                    summary_df.style.format({
                        'Peak Value': '{:.1f}',
                        'Lowest Value': '{:.1f}',
                        'Range': '{:.1f}',
                        'Variability %': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"Need at least 3 months of data for {selected_team_arc}")

with tab6:
    # Section F: Physical Style Evolution (5-Year Trends)
        st.subheader("Physical Style Evolution Over Time")
        st.markdown("Track how teams have evolved their physical approach across multiple seasons")
        
        # Check if we have multi-year data
        available_years = sorted(df_filtered['Year'].unique())
        
        if len(available_years) < 2:
            st.warning("Need at least 2 years of data for evolution analysis. Please adjust your season filters.")
        else:
            # Calculate team season averages for evolution metrics
            evolution_metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
                               'High Acceleration Count P90', 'High Deceleration Count P90']
            
            team_year_stats = df_filtered.groupby(['Team', 'Year'])[evolution_metrics].mean().reset_index()
            
            # F1: League-Wide Evolution Summary
            st.markdown("### F1: League-Wide Evolution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                league_evolution_metric = st.selectbox(
                    "Select Metric for League Evolution",
                    evolution_metrics,
                    key="league_evolution_metric"
                )
            
            # Calculate league averages by year
            league_year_avg = df_filtered.groupby('Year')[evolution_metrics].mean().reset_index()
            
            # Calculate year-over-year change
            league_year_avg_sorted = league_year_avg.sort_values('Year')
            
            fig_league_evolution = go.Figure()
            
            fig_league_evolution.add_trace(go.Scatter(
                x=league_year_avg_sorted['Year'],
                y=league_year_avg_sorted[league_evolution_metric],
                mode='lines+markers',
                name='League Average',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=10)
            ))
            
            fig_league_evolution.update_layout(
                title=f'MLS League-Wide {league_evolution_metric} Evolution',
                xaxis_title='Year',
                yaxis_title=league_evolution_metric,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_league_evolution, use_container_width=True)
            
            # Calculate overall trend
            if len(league_year_avg_sorted) >= 2:
                first_year_val = league_year_avg_sorted[league_evolution_metric].iloc[0]
                last_year_val = league_year_avg_sorted[league_evolution_metric].iloc[-1]
                total_change_pct = ((last_year_val - first_year_val) / first_year_val * 100)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"{available_years[0]} Average",
                        f"{first_year_val:.1f}",
                        ""
                    )
                
                with col2:
                    st.metric(
                        f"{available_years[-1]} Average",
                        f"{last_year_val:.1f}",
                        f"{total_change_pct:+.1f}% vs {available_years[0]}"
                    )
                
                with col3:
                    trend_direction = "Increasing" if total_change_pct > 2 else "Decreasing" if total_change_pct < -2 else "Stable"
                    st.metric(
                        "League Trend",
                        trend_direction,
                        f"{total_change_pct:+.1f}%"
                    )
            
            # F2: Team-Specific Evolution Tracking
            st.markdown("---")
            st.markdown("### F2: Team Evolution Tracking")
            
            col1, col2 = st.columns(2)
            
            with col1:
                evolution_metric_focus = st.selectbox(
                    "Select Metric to Track",
                    evolution_metrics,
                    key="team_evolution_metric"
                )
            
            with col2:
                # Team selector for highlighting specific teams
                teams_to_highlight = st.multiselect(
                    "Highlight Specific Teams (leave empty to show all)",
                    options=sorted(team_year_stats['Team'].unique()),
                    default=[],
                    key="teams_highlight"
                )
            
            # Create evolution line chart
            fig_team_evolution = go.Figure()
            
            # Add league average as reference
            fig_team_evolution.add_trace(go.Scatter(
                x=league_year_avg_sorted['Year'],
                y=league_year_avg_sorted[evolution_metric_focus],
                mode='lines',
                name='League Average',
                line=dict(color='black', width=3, dash='dash'),
                opacity=0.7
            ))
            
            # If specific teams selected, show only those; otherwise show all
            if teams_to_highlight:
                teams_to_plot = teams_to_highlight
            else:
                teams_to_plot = team_year_stats['Team'].unique()
            
            for team in teams_to_plot:
                team_data = team_year_stats[team_year_stats['Team'] == team].sort_values('Year')
                
                # Determine if this is a highlighted team (thicker line)
                is_highlighted = team in teams_to_highlight if teams_to_highlight else False
                
                fig_team_evolution.add_trace(go.Scatter(
                    x=team_data['Year'],
                    y=team_data[evolution_metric_focus],
                    mode='lines+markers',
                    name=team,
                    line=dict(width=3 if is_highlighted else 1),
                    opacity=1.0 if is_highlighted else 0.4,
                    marker=dict(size=8 if is_highlighted else 4)
                ))
            
            fig_team_evolution.update_layout(
                title=f'Team {evolution_metric_focus} Evolution Over Time',
                xaxis_title='Year',
                yaxis_title=evolution_metric_focus,
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            st.plotly_chart(fig_team_evolution, use_container_width=True)
            
            # F3: Heatmap - Team Intensity Evolution
            st.markdown("---")
            st.markdown("### F3: Team Physical Intensity Heatmap")
            
            heatmap_metric = st.selectbox(
                "Select Metric for Heatmap",
                evolution_metrics,
                key="heatmap_metric"
            )
            
            # Pivot data for heatmap
            heatmap_data = team_year_stats.pivot(
                index='Team',
                columns='Year',
                values=heatmap_metric
            )
            
            # Create heatmap
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="Team", color=heatmap_metric),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale='RdYlGn',
                title=f'{heatmap_metric} - Team Evolution Heatmap'
            )
            
            fig_heatmap.update_layout(
                height=800,
                xaxis_title="Year",
                yaxis_title="Team"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # F4: Biggest Movers Analysis
            st.markdown("---")
            st.markdown("### F4: Biggest Movers - Year 1 vs Year 5")
            
            # Calculate change from first to last year in dataset
            first_year = available_years[0]
            last_year = available_years[-1]
            
            mover_metric = st.selectbox(
                "Select Metric for Biggest Movers",
                evolution_metrics,
                key="mover_metric"
            )
            
            # Get first and last year data
            first_year_data = team_year_stats[team_year_stats['Year'] == first_year][['Team', mover_metric]]
            last_year_data = team_year_stats[team_year_stats['Year'] == last_year][['Team', mover_metric]]
            
            # Merge to calculate change
            movers_df = first_year_data.merge(
                last_year_data,
                on='Team',
                suffixes=(f'_{first_year}', f'_{last_year}')
            )
            
            movers_df['Absolute Change'] = movers_df[f'{mover_metric}_{last_year}'] - movers_df[f'{mover_metric}_{first_year}']
            movers_df['Percent Change'] = (movers_df['Absolute Change'] / movers_df[f'{mover_metric}_{first_year}'] * 100)
            
            movers_df = movers_df.sort_values('Percent Change', ascending=True)
            
            # Create diverging bar chart
            fig_movers = px.bar(
                movers_df,
                x='Percent Change',
                y='Team',
                orientation='h',
                title=f'Biggest Movers: {mover_metric} ({first_year} → {last_year})',
                color='Percent Change',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                labels={'Percent Change': f'% Change in {mover_metric}'},
                hover_data=[f'{mover_metric}_{first_year}', f'{mover_metric}_{last_year}', 'Absolute Change'],
                height=700
            )
            
            fig_movers.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_movers, use_container_width=True)
            
            # Summary stats for movers
            col1, col2, col3 = st.columns(3)
            
            with col1:
                biggest_improver = movers_df.iloc[-1]
                st.metric(
                    "Biggest Improver",
                    biggest_improver['Team'],
                    f"+{biggest_improver['Percent Change']:.1f}%"
                )
            
            with col2:
                biggest_decliner = movers_df.iloc[0]
                st.metric(
                    "Biggest Decliner",
                    biggest_decliner['Team'],
                    f"{biggest_decliner['Percent Change']:.1f}%"
                )
            
            with col3:
                teams_improved = len(movers_df[movers_df['Percent Change'] > 0])
                st.metric(
                    "Teams Improved",
                    f"{teams_improved}/{len(movers_df)}",
                    f"{teams_improved/len(movers_df)*100:.1f}%"
                )
            
            # F5: Multi-Metric Team Evolution Comparison
            st.markdown("---")
            st.markdown("### F5: Complete Team Evolution Profile")
            
            selected_team_evolution = st.selectbox(
                "Select Team to View Complete Evolution",
                options=sorted(team_year_stats['Team'].unique()),
                key="team_complete_evolution"
            )
            
            team_evolution_data = team_year_stats[team_year_stats['Team'] == selected_team_evolution].sort_values('Year')
            
            # Create multi-line chart showing all metrics
            fig_multi_evolution = go.Figure()
            
            for metric in evolution_metrics:
                fig_multi_evolution.add_trace(go.Scatter(
                    x=team_evolution_data['Year'],
                    y=team_evolution_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
            
            fig_multi_evolution.update_layout(
                title=f'{selected_team_evolution} - Complete Physical Evolution',
                xaxis_title='Year',
                yaxis_title='Metric Value',
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_multi_evolution, use_container_width=True)
            
            # Show year-over-year changes table
            st.markdown(f"#### {selected_team_evolution} - Year-over-Year Changes")
            
            if len(team_evolution_data) >= 2:
                yoy_changes = []
                
                for i in range(1, len(team_evolution_data)):
                    prev_row = team_evolution_data.iloc[i-1]
                    curr_row = team_evolution_data.iloc[i]
                    
                    change_row = {
                        'Period': f"{int(prev_row['Year'])} → {int(curr_row['Year'])}"
                    }
                    
                    for metric in evolution_metrics:
                        prev_val = prev_row[metric]
                        curr_val = curr_row[metric]
                        pct_change = ((curr_val - prev_val) / prev_val * 100) if prev_val > 0 else 0
                        change_row[metric] = pct_change
                    
                    yoy_changes.append(change_row)
                
                yoy_df = pd.DataFrame(yoy_changes)
                
                # Style the dataframe
                def color_changes(val):
                    if isinstance(val, (int, float)):
                        if val > 2:
                            return 'background-color: #d4edda; color: #155724'
                        elif val < -2:
                            return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_yoy = yoy_df.style.applymap(
                    color_changes,
                    subset=[m for m in evolution_metrics]
                ).format({metric: '{:+.1f}%' for metric in evolution_metrics})
                
                st.dataframe(styled_yoy, use_container_width=True)
            else:
                st.info(f"Need at least 2 years of data for {selected_team_evolution}")

# Tab 2: Fatigue & Season Dynamics
with tab2:
    st.header("Fatigue Analysis & Season Dynamics")
    
    # Season progression analysis
    st.subheader("📅 Physical Output Across Season")
    
    # Add month column
    df_filtered['Month'] = df_filtered['Date'].dt.month
    
    # Calculate monthly averages
    monthly_metrics = df_filtered.groupby(['Month', 'Team'])[
        ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90']
    ].mean().reset_index()
    
    # Create season progression plot
    metric_choice = st.selectbox(
        "Select Metric for Season Progression",
        ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90']
    )
    
    fig_season = px.line(
        monthly_metrics,
        x='Month',
        y=metric_choice,
        color='Team',
        title=f'{metric_choice} Throughout the Season',
        markers=True
    )
    
    fig_season.update_xaxes(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    
    # First Half vs Second Half Analysis
    st.subheader("⏱️ Within-Game Fatigue: 1st Half vs 2nd Half")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_team_fatigue = st.selectbox(
            "Select Team for Fatigue Analysis",
            options=sorted(df_filtered['Team'].unique())
        )
    
    team_fatigue_df = df_filtered[df_filtered['Team'] == selected_team_fatigue]
    
    # Calculate fatigue metrics
    fatigue_metrics = []
    for _, row in team_fatigue_df.iterrows():
        if pd.notna(row['Distance 1']) and pd.notna(row['Distance 2']):
            fatigue_metrics.append({
                'Player': row['Short Name'],
                'Match': row['Match'][:20] + '...',
                'Distance Drop %': ((row['Distance 1'] - row['Distance 2']) / row['Distance 1'] * 100),
                'HSR Drop %': ((row['HSR Distance 1'] - row['HSR Distance 2']) / row['HSR Distance 1'] * 100) if row['HSR Distance 1'] > 0 else 0,
                'Sprint Drop %': ((row['Sprint Distance 1'] - row['Sprint Distance 2']) / row['Sprint Distance 1'] * 100) if row['Sprint Distance 1'] > 0 else 0
            })
    
    if fatigue_metrics:
        fatigue_df = pd.DataFrame(fatigue_metrics)
        
        # Average fatigue by player
        player_fatigue = fatigue_df.groupby('Player')[['Distance Drop %', 'HSR Drop %', 'Sprint Drop %']].mean().reset_index()
        player_fatigue = player_fatigue.sort_values('Distance Drop %', ascending=False).head(15)
        
        fig_fatigue = px.bar(
            player_fatigue,
            x='Player',
            y=['Distance Drop %', 'HSR Drop %', 'Sprint Drop %'],
            title=f'Top 15 Players - Physical Drop-off (1st Half vs 2nd Half) - {selected_team_fatigue}',
            barmode='group'
        )
        
        fig_fatigue.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Drop-off %",
            legend_title="Metric"
        )
        
        st.plotly_chart(fig_fatigue, use_container_width=True)
    
    # Cumulative Load
    st.subheader("📊 Cumulative Season Load")
    
    # Get top players by TOTAL minutes played
    player_minutes_total = df_filtered.groupby('Player').agg({
        'Minutes': 'sum',
        'Match ID': 'nunique'
    }).reset_index()
    player_minutes_total.columns = ['Player', 'Total Minutes', 'Games Played']
    
    # Get top 20 players by total minutes
    top_players = player_minutes_total.nlargest(20, 'Total Minutes')['Player'].tolist()
    
    # Display top players' total minutes
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Most Used Player", 
                  top_players[0] if top_players else "N/A",
                  f"{player_minutes_total[player_minutes_total['Player'] == top_players[0]]['Total Minutes'].values[0]:.0f} mins" if top_players else "")
    with col2:
        avg_minutes = player_minutes_total['Total Minutes'].mean()
        st.metric("Avg Minutes per Player",
                  f"{avg_minutes:.0f} mins",
                  f"Across {len(player_minutes_total)} players")
    
    # Calculate cumulative load for top players
    cumulative_df = df_filtered[df_filtered['Player'].isin(top_players)].sort_values('Date')
    
    cumulative_load = []
    for player in top_players:
        player_df = cumulative_df[cumulative_df['Player'] == player].sort_values('Date')
        player_df['Cumulative Distance'] = player_df['Distance'].cumsum()
        player_df['Cumulative Minutes'] = player_df['Minutes'].cumsum()
        player_df['Cumulative HI Distance'] = player_df['HI Distance'].cumsum()
        cumulative_load.append(player_df[['Player', 'Date', 'Cumulative Distance', 'Cumulative Minutes', 'Cumulative HI Distance']])
    
    if cumulative_load:
        cumulative_final = pd.concat(cumulative_load)
        
        # Let user choose what to display
        load_metric = st.selectbox(
            "Select Cumulative Metric",
            ["Cumulative Distance", "Cumulative Minutes", "Cumulative HI Distance"]
        )
        
        fig_cumulative = px.line(
            cumulative_final,
            x='Date',
            y=load_metric,
            color='Player',
            title=f'{load_metric} - Top 20 Players by Total Minutes Played',
            height=500,
            hover_data=['Player']
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        # Show summary table
        st.markdown("#### Season Totals - Top Players")
        summary_df = df_filtered[df_filtered['Player'].isin(top_players)].groupby('Player').agg({
            'Minutes': 'sum',
            'Distance': 'sum',
            'HSR Distance': 'sum',
            'Sprint Distance': 'sum',
            'Match ID': 'nunique'
        }).round(0).reset_index()
        summary_df.columns = ['Player', 'Total Minutes', 'Total Distance', 'Total HSR', 'Total Sprint', 'Games']
        summary_df = summary_df.sort_values('Total Minutes', ascending=False)
        
        st.dataframe(summary_df.style.format({
            'Total Minutes': '{:.0f}',
            'Total Distance': '{:.0f}',
            'Total HSR': '{:.0f}',
            'Total Sprint': '{:.0f}',
            'Games': '{:.0f}'
        }), use_container_width=True)

# Tab 3: Performance Intelligence
with tab3:
    st.header("Performance Intelligence & Success Correlation")
    
    # Physical Dominance Matrix
    st.subheader("💪 Physical Dominance Analysis")
    
    # Calculate match-level physical dominance
    match_comparison = []
    
    for match_id in df_filtered['Match ID'].unique():
        match_data = df_filtered[df_filtered['Match ID'] == match_id]
        teams = match_data['Team'].unique()
        
        if len(teams) == 2:
            team1_data = match_data[match_data['Team'] == teams[0]]
            team2_data = match_data[match_data['Team'] == teams[1]]
            
            # Calculate team totals
            metrics_to_compare = ['Distance', 'HSR Distance', 'Sprint Distance', 'HI Count']
            
            comparison = {
                'Match': match_data['Match'].iloc[0],
                'Date': match_data['Date'].iloc[0],
                'Team 1': teams[0],
                'Team 2': teams[1]
            }
            
            for metric in metrics_to_compare:
                team1_total = team1_data[metric].sum()
                team2_total = team2_data[metric].sum()
                comparison[f'{metric} Diff'] = team1_total - team2_total
                comparison[f'{metric} Dom %'] = (team1_total / (team1_total + team2_total) * 100) if (team1_total + team2_total) > 0 else 50
            
            match_comparison.append(comparison)
    
    if match_comparison:
        dominance_df = pd.DataFrame(match_comparison)
        
        # Team dominance summary
        team_dominance = []
        for team in df_filtered['Team'].unique():
            team_matches = dominance_df[(dominance_df['Team 1'] == team) | (dominance_df['Team 2'] == team)]
            
            dom_count = 0
            for _, match in team_matches.iterrows():
                if match['Team 1'] == team:
                    dom_count += sum([match[f'{m} Dom %'] > 50 for m in ['Distance', 'HSR Distance', 'Sprint Distance', 'HI Count']])
                else:
                    dom_count += sum([match[f'{m} Dom %'] < 50 for m in ['Distance', 'HSR Distance', 'Sprint Distance', 'HI Count']])
            
            team_dominance.append({
                'Team': team,
                'Matches': len(team_matches),
                'Physical Dominance Score': dom_count / (len(team_matches) * 4) * 100 if len(team_matches) > 0 else 0
            })
        
        team_dom_df = pd.DataFrame(team_dominance).sort_values('Physical Dominance Score', ascending=False)
        
        fig_dominance = px.bar(
            team_dom_df.head(15),
            x='Physical Dominance Score',
            y='Team',
            orientation='h',
            title='Team Physical Dominance Score (% of metrics where team outperformed opponents)',
            color='Physical Dominance Score',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig_dominance, use_container_width=True)
    
    # Key Physical Differentiators
    st.subheader("🔑 Key Physical Differentiators")
    
    # Calculate team season averages
    team_season_stats = df_filtered.groupby('Team')[
        ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
         'High Acceleration Count P90', 'High Deceleration Count P90']
    ].mean().reset_index()
    
    # Create correlation matrix
    corr_matrix = team_season_stats[
        ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
         'High Acceleration Count P90', 'High Deceleration Count P90']
    ].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Physical Metrics Correlation Matrix',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Position-specific excellence
    st.subheader("🎯 Position-Specific Excellence")
    
    position_group = st.selectbox(
        "Select Position Group",
        options=sorted(df_filtered['Position Group'].unique())
    )
    
    position_df = df_filtered[df_filtered['Position Group'] == position_group]
    
    # Top performers by position
    top_position_players = position_df.groupby('Player')[
        ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90']
    ].mean().reset_index()
    
    top_position_players['Overall Score'] = (
        top_position_players['Distance P90'].rank(pct=True) * 0.3 +
        top_position_players['HSR Distance P90'].rank(pct=True) * 0.35 +
        top_position_players['Sprint Distance P90'].rank(pct=True) * 0.35
    )
    
    top_position_players = top_position_players.nlargest(10, 'Overall Score')
    
    fig_position = px.scatter(
        top_position_players,
        x='HSR Distance P90',
        y='Sprint Distance P90',
        size='Distance P90',
        color='Overall Score',
        hover_data=['Player'],
        title=f'Top 10 {position_group} - Physical Performance',
        color_continuous_scale='Viridis'
    )
    
    # Add player names to the plot
    for _, row in top_position_players.iterrows():
        fig_position.add_annotation(
            x=row['HSR Distance P90'],
            y=row['Sprint Distance P90'],
            text=row['Player'][:15],
            showarrow=False,
            font=dict(size=8)
        )
    
    st.plotly_chart(fig_position, use_container_width=True)

# Tab 5: Advanced Performance Intelligenc
with tab5:
    st.header("Advanced Performance Intelligence")
    st.markdown("Deep-dive analytics on consistency, evolution, and peak performance windows")

        
        # Minimum games filter
    min_games_consistency = 5
    
    # Key metrics for consistency analysis
    consistency_metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 'High Acceleration Count P90']
    
    # E1: Player Consistency Analysis
    st.markdown("### E1: Player Consistency Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consistency_time_view = st.radio(
            "Time Period View",
            ["All Seasons Combined", "Per-Season Analysis"],
            help="Analyze consistency across all data or year-by-year"
        )
    
    with col2:
        position_filter_consistency = st.selectbox(
            "Filter by Position Group",
            ["All Positions"] + sorted(df_filtered['Position Group'].unique().tolist()),
            key="consistency_position"
        )
    
    with col3:
        consistency_metric_focus = st.selectbox(
            "Primary Metric for Analysis",
            consistency_metrics,
            key="consistency_metric"
        )
    
    # Filter by position if selected
    df_consistency = df_filtered.copy()
    if position_filter_consistency != "All Positions":
        df_consistency = df_consistency[df_consistency['Position Group'] == position_filter_consistency]
    
    # Calculate player consistency
    if consistency_time_view == "All Seasons Combined":
        # Group by player across all seasons
        player_games_count = df_consistency.groupby('Player')['Match ID'].nunique().reset_index()
        player_games_count.columns = ['Player', 'Games Played']
        
        # Filter players with minimum games
        qualified_players_consistency = player_games_count[player_games_count['Games Played'] >= min_games_consistency]['Player'].tolist()
        df_qualified = df_consistency[df_consistency['Player'].isin(qualified_players_consistency)]
        
        # Calculate mean, std, and CV for each player
        player_consistency_stats = []
        
        for player in qualified_players_consistency:
            player_data = df_qualified[df_qualified['Player'] == player]
            
            stats = {
                'Player': player,
                'Position': player_data['Position'].mode()[0] if len(player_data) > 0 else 'Unknown',
                'Position Group': player_data['Position Group'].mode()[0] if len(player_data) > 0 else 'Unknown',
                'Games Played': len(player_data),
                'Team': player_data['Team'].mode()[0] if len(player_data) > 0 else 'Unknown'
            }
            
            for metric in consistency_metrics:
                mean_val = player_data[metric].mean()
                std_val = player_data[metric].std()
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                
                stats[f'{metric} Mean'] = mean_val
                stats[f'{metric} Std'] = std_val
                stats[f'{metric} CV'] = cv
                stats[f'{metric} Consistency Score'] = mean_val / std_val if std_val > 0 else 0
            
            player_consistency_stats.append(stats)
        
        player_consistency_df = pd.DataFrame(player_consistency_stats)
        
        if len(player_consistency_df) > 0:
            # Scatter Plot: Mean vs CV
            st.markdown(f"#### Performance vs Consistency: {consistency_metric_focus}")
            
            # Create quadrants for interpretation
            mean_col = f'{consistency_metric_focus} Mean'
            cv_col = f'{consistency_metric_focus} CV'
            
            median_mean = player_consistency_df[mean_col].median()
            median_cv = player_consistency_df[cv_col].median()
            
            # Add quadrant labels
            player_consistency_df['Quadrant'] = player_consistency_df.apply(
                lambda row: 
                    'Elite Reliable' if row[mean_col] >= median_mean and row[cv_col] <= median_cv
                    else 'High Output Volatile' if row[mean_col] >= median_mean and row[cv_col] > median_cv
                    else 'Low Output Consistent' if row[mean_col] < median_mean and row[cv_col] <= median_cv
                    else 'Inconsistent Low Output',
                axis=1
            )
            
            fig_scatter = px.scatter(
                player_consistency_df,
                x=mean_col,
                y=cv_col,
                color='Quadrant',
                size='Games Played',
                hover_data=['Player', 'Position', 'Team', 'Games Played'],
                title=f'{consistency_metric_focus}: Average Performance vs Coefficient of Variation',
                labels={mean_col: f'Average {consistency_metric_focus}', cv_col: 'Coefficient of Variation (%)'},
                color_discrete_map={
                    'Elite Reliable': '#2ecc71',
                    'High Output Volatile': '#f39c12',
                    'Low Output Consistent': '#3498db',
                    'Inconsistent Low Output': '#e74c3c'
                },
                height=600
            )
            
            # Add median lines
            fig_scatter.add_hline(y=median_cv, line_dash="dash", line_color="gray", opacity=0.5)
            fig_scatter.add_vline(x=median_mean, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add annotations for quadrants
            fig_scatter.add_annotation(x=player_consistency_df[mean_col].max() * 0.95, y=player_consistency_df[cv_col].min() * 1.1,
                                        text="Elite Reliable", showarrow=False, font=dict(size=10, color="gray"))
            fig_scatter.add_annotation(x=player_consistency_df[mean_col].max() * 0.95, y=player_consistency_df[cv_col].max() * 0.95,
                                        text="High Output Volatile", showarrow=False, font=dict(size=10, color="gray"))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Top 20 Most Consistent Players
            st.markdown(f"#### Top 20 Most Consistent Players: {consistency_metric_focus}")
            
            consistency_score_col = f'{consistency_metric_focus} Consistency Score'
            top_consistent = player_consistency_df.nlargest(20, consistency_score_col)
            
            fig_top_consistent = px.bar(
                top_consistent,
                y='Player',
                x=consistency_score_col,
                orientation='h',
                color=mean_col,
                title=f'Top 20 Most Consistent Players (Min {min_games_consistency} games)',
                labels={consistency_score_col: 'Consistency Score', mean_col: f'Avg {consistency_metric_focus}'},
                hover_data=['Position', 'Team', 'Games Played', cv_col],
                color_continuous_scale='Viridis',
                height=600
            )
            
            st.plotly_chart(fig_top_consistent, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                elite_count = len(player_consistency_df[player_consistency_df['Quadrant'] == 'Elite Reliable'])
                st.metric("Elite Reliable Players", elite_count, f"{elite_count/len(player_consistency_df)*100:.1f}%")
            
            with col2:
                volatile_count = len(player_consistency_df[player_consistency_df['Quadrant'] == 'High Output Volatile'])
                st.metric("High Output Volatile", volatile_count, f"{volatile_count/len(player_consistency_df)*100:.1f}%")
            
            with col3:
                avg_cv = player_consistency_df[cv_col].mean()
                st.metric("Average CV%", f"{avg_cv:.1f}%")
            
            with col4:
                most_consistent_player = top_consistent.iloc[0]
                st.metric("Most Consistent", most_consistent_player['Player'][:15], f"CV: {most_consistent_player[cv_col]:.1f}%")
            
        else:
            st.warning(f"No players found with minimum {min_games_consistency} games in the selected filters.")
    
    else:  # Per-Season Analysis
        st.markdown("#### Year-over-Year Consistency Comparison")
        
        # Calculate per-season consistency
        season_consistency_stats = []
        
        for season in df_consistency['Season'].unique():
            season_data = df_consistency[df_consistency['Season'] == season]
            
            # Get players with minimum games in this season
            player_games = season_data.groupby('Player')['Match ID'].nunique().reset_index()
            player_games.columns = ['Player', 'Games']
            qualified = player_games[player_games['Games'] >= min_games_consistency]['Player'].tolist()
            
            season_qualified = season_data[season_data['Player'].isin(qualified)]
            
            for player in qualified:
                player_data = season_qualified[season_qualified['Player'] == player]
                
                stats = {
                    'Player': player,
                    'Season': season,
                    'Position': player_data['Position'].mode()[0] if len(player_data) > 0 else 'Unknown',
                    'Position Group': player_data['Position Group'].mode()[0] if len(player_data) > 0 else 'Unknown',
                    'Games': len(player_data),
                    'Team': player_data['Team'].mode()[0] if len(player_data) > 0 else 'Unknown'
                }
                
                for metric in consistency_metrics:
                    mean_val = player_data[metric].mean()
                    std_val = player_data[metric].std()
                    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                    
                    stats[f'{metric} Mean'] = mean_val
                    stats[f'{metric} CV'] = cv
                
                season_consistency_stats.append(stats)
        
        season_consistency_df = pd.DataFrame(season_consistency_stats)
        
        if len(season_consistency_df) > 0:
            # Select a player to view their consistency evolution
            players_multi_season = season_consistency_df.groupby('Player')['Season'].nunique()
            players_multi_season = players_multi_season[players_multi_season >= 2].index.tolist()
            
            if players_multi_season:
                selected_consistency_player = st.selectbox(
                    "Select Player to View Consistency Evolution",
                    sorted(players_multi_season),
                    key="player_consistency_evolution"
                )
                
                player_evolution = season_consistency_df[season_consistency_df['Player'] == selected_consistency_player]
                
                # Create line chart showing CV across seasons
                fig_evolution = go.Figure()
                
                for metric in consistency_metrics:
                    fig_evolution.add_trace(go.Scatter(
                        x=player_evolution['Season'],
                        y=player_evolution[f'{metric} CV'],
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig_evolution.update_layout(
                    title=f'{selected_consistency_player} - Consistency Evolution (CV%)',
                    xaxis_title='Season',
                    yaxis_title='Coefficient of Variation (%)',
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_evolution, use_container_width=True)
                
                # Season comparison table
                st.markdown(f"#### {selected_consistency_player} - Season-by-Season Stats")
                
                display_cols = ['Season', 'Games', 'Team'] + [f'{m} Mean' for m in consistency_metrics] + [f'{m} CV' for m in consistency_metrics]
                display_df = player_evolution[display_cols].sort_values('Season')
                
                st.dataframe(
                    display_df.style.format({
                        **{f'{m} Mean': '{:.1f}' for m in consistency_metrics},
                        **{f'{m} CV': '{:.1f}%' for m in consistency_metrics},
                        'Games': '{:.0f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No players with data in multiple seasons found.")
        else:
            st.warning(f"No players found with minimum {min_games_consistency} games per season.")
    
    # E2: Team Consistency Analysis
    st.markdown("---")
    st.markdown("### E2: Team Consistency Analysis")
    st.markdown("Game-to-game variance in team physical output")
    
    # Calculate team-level consistency (Option A: aggregate team output per game, then measure variance)
    team_game_aggregates = []
    
    for (match_id, team), group in df_filtered.groupby(['Match ID', 'Team']):
        agg_data = {
            'Match ID': match_id,
            'Team': team,
            'Date': group['Date'].iloc[0],
            'Season': group['Season'].iloc[0]
        }
        
        # Aggregate team physical output for this game
        for metric in consistency_metrics:
            # Sum of all players' P90 metrics weighted by their minutes
            total_minutes = group['Minutes'].sum()
            if total_minutes > 0:
                # Convert P90 back to total, sum, then re-normalize
                metric_base = metric.replace(' P90', '')
                if metric_base in group.columns:
                    total_output = (group[metric_base]).sum()
                    agg_data[metric] = total_output / total_minutes * 90
                else:
                    agg_data[metric] = group[metric].mean()  # Fallback to mean if base metric not found
            else:
                agg_data[metric] = 0
        
        team_game_aggregates.append(agg_data)
    
    team_game_df = pd.DataFrame(team_game_aggregates)
    
    # Calculate team consistency stats
    team_consistency_stats = []
    
    for team in team_game_df['Team'].unique():
        team_data = team_game_df[team_game_df['Team'] == team]
        
        if len(team_data) >= min_games_consistency:
            stats = {
                'Team': team,
                'Games': len(team_data),
                'Seasons': team_data['Season'].nunique()
            }
            
            for metric in consistency_metrics:
                mean_val = team_data[metric].mean()
                std_val = team_data[metric].std()
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                
                stats[f'{metric} Mean'] = mean_val
                stats[f'{metric} Std'] = std_val
                stats[f'{metric} CV'] = cv
                stats[f'{metric} Consistency Score'] = mean_val / std_val if std_val > 0 else 0
            
            team_consistency_stats.append(stats)
    
    team_consistency_df = pd.DataFrame(team_consistency_stats)
    
    if len(team_consistency_df) > 0:
        # Team consistency ranking
        st.markdown(f"#### Team Consistency Rankings: {consistency_metric_focus}")
        
        cv_col_team = f'{consistency_metric_focus} CV'
        mean_col_team = f'{consistency_metric_focus} Mean'
        
        team_consistency_sorted = team_consistency_df.sort_values(cv_col_team)
        
        fig_team_consistency = px.bar(
            team_consistency_sorted,
            x=cv_col_team,
            y='Team',
            orientation='h',
            color=mean_col_team,
            title=f'Team Consistency: {consistency_metric_focus} (Lower CV% = More Consistent)',
            labels={cv_col_team: 'Coefficient of Variation (%)', mean_col_team: f'Average {consistency_metric_focus}'},
            hover_data=['Games', 'Seasons'],
            color_continuous_scale='RdYlGn_r',
            height=700
        )
        
        st.plotly_chart(fig_team_consistency, use_container_width=True)
        
        # Most vs Least Consistent Teams
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Most Consistent Teams")
            most_consistent = team_consistency_sorted.head(5)[['Team', cv_col_team, mean_col_team, 'Games']]
            st.dataframe(
                most_consistent.style.format({
                    cv_col_team: '{:.1f}%',
                    mean_col_team: '{:.1f}',
                    'Games': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("##### Most Volatile Teams")
            least_consistent = team_consistency_sorted.tail(5)[['Team', cv_col_team, mean_col_team, 'Games']]
            st.dataframe(
                least_consistent.style.format({
                    cv_col_team: '{:.1f}%',
                    mean_col_team: '{:.1f}',
                    'Games': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # Comparison across all metrics
        st.markdown("#### Team Consistency Across All Metrics")
        
        team_cv_comparison = team_consistency_df[['Team'] + [f'{m} CV' for m in consistency_metrics]]
        team_cv_melted = team_cv_comparison.melt(id_vars=['Team'], var_name='Metric', value_name='CV%')
        team_cv_melted['Metric'] = team_cv_melted['Metric'].str.replace(' CV', '')
        
        fig_team_multi = px.box(
            team_cv_melted,
            x='Metric',
            y='CV%',
            title='Distribution of Team Consistency Across Metrics',
            labels={'CV%': 'Coefficient of Variation (%)'},
            height=500
        )
        
        st.plotly_chart(fig_team_multi, use_container_width=True)
        
    else:
        st.warning(f"No teams found with minimum {min_games_consistency} games.")

# Tab 4: Player Comparison Tool
with tab4:
    st.header("Player Comparison & Benchmarking Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Player selection
        selected_player = st.selectbox(
            "Select Player",
            options=sorted(df_filtered['Player'].unique())
        )
    
    with col2:
        comparison_type = st.radio(
            "Comparison Type",
            ["Season vs Season", "Player vs Position Average", "Player vs League Top Performers"]
        )
    
    player_df = df_filtered[df_filtered['Player'] == selected_player]
    
    if comparison_type == "Season vs Season":
        st.subheader(f"📊 {selected_player} - Season Comparison")
        
        # Get player's seasons
        player_seasons = player_df['Season'].unique()
        
        if len(player_seasons) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                season1 = st.selectbox("Season 1", options=player_seasons, index=0)
            with col2:
                season2 = st.selectbox("Season 2", options=player_seasons, index=1 if len(player_seasons) > 1 else 0)
            
            # Calculate season averages
            metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
                      'High Acceleration Count P90', 'High Deceleration Count P90']
            
            season1_data = player_df[player_df['Season'] == season1][metrics].mean()
            season2_data = player_df[player_df['Season'] == season2][metrics].mean()
            
            # Create comparison radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=season1_data.values,
                theta=metrics,
                fill='toself',
                name=season1
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=season2_data.values,
                theta=metrics,
                fill='toself',
                name=season2
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(season1_data.max(), season2_data.max()) * 1.1]
                    )
                ),
                showlegend=True,
                title=f"{selected_player} - Season Comparison"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Performance change table
            st.subheader("Performance Change Analysis")
            
            change_data = pd.DataFrame({
                'Metric': metrics,
                season1: season1_data.values,
                season2: season2_data.values,
                'Change %': ((season2_data.values - season1_data.values) / season1_data.values * 100)
            })
            
            # Style the dataframe
            def style_change(val):
                if val > 0:
                    return 'color: green'
                elif val < 0:
                    return 'color: red'
                return ''
            
            styled_df = change_data.style.applymap(style_change, subset=['Change %'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning(f"Not enough seasons of data for {selected_player}")
    
    elif comparison_type == "Player vs Position Average":
        st.subheader(f"📊 {selected_player} vs Position Average")
        
        # Get player position
        player_position = player_df['Position'].mode()[0] if len(player_df) > 0 else None
        
        if player_position:
            # Get position average
            position_avg_df = df_filtered[df_filtered['Position'] == player_position]
            
            metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 
                      'High Acceleration Count P90', 'High Deceleration Count P90']
            
            player_avg = player_df[metrics].mean()
            position_avg = position_avg_df[metrics].mean()
            
            # Normalize for radar chart
            max_vals = position_avg_df[metrics].max()
            player_norm = (player_avg / max_vals * 100).fillna(0)
            position_norm = (position_avg / max_vals * 100).fillna(0)
            
            # Create radar chart
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=player_norm.values,
                theta=metrics,
                fill='toself',
                name=selected_player,
                line_color='blue'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=position_norm.values,
                theta=metrics,
                fill='toself',
                name=f'{player_position} Average',
                line_color='gray',
                opacity=0.5
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=f"{selected_player} vs {player_position} Position Average (Normalized to 100)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Percentile rankings
            st.subheader("Percentile Rankings")
            
            percentiles = []
            for metric in metrics:
                percentile = (position_avg_df[metric] <= player_avg[metric]).mean() * 100
                percentiles.append({
                    'Metric': metric,
                    'Player Value': player_avg[metric],
                    'Position Average': position_avg[metric],
                    'Percentile': percentile
                })
            
            percentile_df = pd.DataFrame(percentiles)
            
            # Create bar chart for percentiles
            fig_percentile = px.bar(
                percentile_df,
                x='Percentile',
                y='Metric',
                orientation='h',
                title=f'{selected_player} - Percentile Rankings within {player_position} Position',
                color='Percentile',
                color_continuous_scale='Viridis',
                text='Percentile'
            )
            
            fig_percentile.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_percentile.update_xaxes(range=[0, 100])
            
            st.plotly_chart(fig_percentile, use_container_width=True)
    
    else:  # Player vs League Top Performers
        st.subheader(f"📊 {selected_player} vs League Top Performers")
        
        # Get player position and season for fair comparison
        player_position = player_df['Position'].mode()[0] if len(player_df) > 0 else None
        player_seasons = player_df['Season'].unique()
        
        if player_position and len(player_seasons) > 0:
            # Let user choose comparison scope
            col1, col2 = st.columns(2)
            with col1:
                comparison_scope = st.radio(
                    "Comparison Scope",
                    ["Current Season", "All Seasons", "Career Average"],
                    help="Choose how to compare the player"
                )
            with col2:
                if comparison_scope == "Current Season":
                    selected_comparison_season = st.selectbox(
                        "Select Season",
                        options=sorted(player_seasons, reverse=True),
                        index=0
                    )
            
            # Filter comparison data based on scope
            if comparison_scope == "Current Season":
                position_players = df_filtered[
                    (df_filtered['Position'] == player_position) & 
                    (df_filtered['Season'] == selected_comparison_season)
                ]
                player_stats = player_df[player_df['Season'] == selected_comparison_season]
                scope_label = f" ({selected_comparison_season} Season)"
            elif comparison_scope == "All Seasons":
                position_players = df_filtered[df_filtered['Position'] == player_position]
                player_stats = player_df
                scope_label = " (All Seasons in Filter)"
            else:  # Career Average
                position_players = df_filtered[df_filtered['Position'] == player_position]
                player_stats = player_df
                scope_label = " (Career Average)"
            
            # Calculate metrics with minimum game threshold
            min_games_threshold = 5  # Minimum games to be included in rankings
            min_minutes_threshold = 450  # Minimum total minutes
            metrics = ['Distance P90', 'HSR Distance P90', 'Sprint Distance P90']
            
            # Group by player and calculate averages AND totals
            player_rankings = position_players.groupby('Player').agg({
                'Distance P90': 'mean',
                'HSR Distance P90': 'mean',
                'Sprint Distance P90': 'mean',
                'Minutes': 'sum',  # Total minutes played
                'Match ID': 'nunique'  # Count unique matches
            }).reset_index()
            
            # Filter for minimum games AND minutes
            player_rankings = player_rankings[
                (player_rankings['Match ID'] >= min_games_threshold) & 
                (player_rankings['Minutes'] >= min_minutes_threshold)
            ]
            player_rankings.rename(columns={'Match ID': 'Games Played', 'Minutes': 'Total Minutes'}, inplace=True)
            
            # Calculate overall score
            player_rankings['Overall Score'] = (
                player_rankings['Distance P90'].rank(pct=True) * 0.3 +
                player_rankings['HSR Distance P90'].rank(pct=True) * 0.35 +
                player_rankings['Sprint Distance P90'].rank(pct=True) * 0.35
            )
            
            # Get top 5 performers plus selected player
            top_players = player_rankings.nlargest(5, 'Overall Score')['Player'].tolist()
            if selected_player not in top_players:
                top_players.append(selected_player)
            
            comparison_df = player_rankings[player_rankings['Player'].isin(top_players)]
            
            # Create grouped bar chart
            comparison_df_melted = comparison_df.melt(
                id_vars=['Player', 'Games Played', 'Total Minutes'], 
                value_vars=metrics
            )
            
            fig_comparison = px.bar(
                comparison_df_melted,
                x='variable',
                y='value',
                color='Player',
                barmode='group',
                title=f'{selected_player} vs Top 5 {player_position} Players{scope_label}',
                labels={'variable': 'Metric', 'value': 'Value'},
                hover_data=['Games Played', 'Total Minutes']
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Ranking table
            st.subheader(f"League Rankings{scope_label}")
            
            player_rank = player_rankings[player_rankings['Player'] == selected_player]
            total_players = len(player_rankings)
            
            if not player_rank.empty:
                # Calculate ranks for each metric
                for metric in metrics:
                    player_rankings[f'{metric}_rank'] = player_rankings[metric].rank(ascending=False, method='min')
                
                player_rank_num = player_rankings[player_rankings['Player'] == selected_player]['Overall Score'].rank(ascending=False, method='min').values[0]
                
                # Display overall rank
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Overall Physical Rank",
                        f"{int(player_rank_num)} / {total_players}",
                        f"Top {player_rank_num/total_players*100:.1f}%"
                    )
                with col2:
                    games_played = player_rank['Games Played'].values[0]
                    total_mins = player_rank['Total Minutes'].values[0]
                    st.metric(
                        "Games / Minutes",
                        f"{int(games_played)} / {int(total_mins)}",
                        f"Min. {min_games_threshold} games & {min_minutes_threshold} mins"
                    )
                with col3:
                    st.metric(
                        "Position",
                        player_position,
                        f"{total_players} qualified players"
                    )
                
                # Detailed metrics comparison
                st.markdown("#### Detailed Metric Rankings")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    player_dist = player_rank['Distance P90'].values[0]
                    dist_rank = int(player_rankings[player_rankings['Player'] == selected_player]['Distance P90_rank'].values[0])
                    dist_percentile = (total_players - dist_rank + 1) / total_players * 100
                    st.metric(
                        "Distance P90",
                        f"{player_dist:.1f}",
                        f"Rank {dist_rank}/{total_players} (Top {dist_percentile:.1f}%)"
                    )
                
                with col2:
                    player_hsr = player_rank['HSR Distance P90'].values[0]
                    hsr_rank = int(player_rankings[player_rankings['Player'] == selected_player]['HSR Distance P90_rank'].values[0])
                    hsr_percentile = (total_players - hsr_rank + 1) / total_players * 100
                    st.metric(
                        "HSR Distance P90",
                        f"{player_hsr:.1f}",
                        f"Rank {hsr_rank}/{total_players} (Top {hsr_percentile:.1f}%)"
                    )
                
                with col3:
                    player_sprint = player_rank['Sprint Distance P90'].values[0]
                    sprint_rank = int(player_rankings[player_rankings['Player'] == selected_player]['Sprint Distance P90_rank'].values[0])
                    sprint_percentile = (total_players - sprint_rank + 1) / total_players * 100
                    st.metric(
                        "Sprint Distance P90",
                        f"{player_sprint:.1f}",
                        f"Rank {sprint_rank}/{total_players} (Top {sprint_percentile:.1f}%)"
                    )
                
                # Show top 10 players in position
                st.markdown(f"#### Top 10 {player_position} Players{scope_label}")
                top_10_display = player_rankings.nlargest(10, 'Overall Score')[
                    ['Player', 'Games Played', 'Total Minutes', 'Distance P90', 'HSR Distance P90', 'Sprint Distance P90', 'Overall Score']
                ].reset_index(drop=True)
                top_10_display.index += 1  # Start ranking from 1
                
                # Highlight selected player if in top 10
                def highlight_player(row):
                    if row['Player'] == selected_player:
                        return ['background-color: yellow'] * len(row)
                    return [''] * len(row)
                
                styled_df = top_10_display.style.apply(highlight_player, axis=1).format({
                    'Distance P90': '{:.1f}',
                    'HSR Distance P90': '{:.1f}',
                    'Sprint Distance P90': '{:.1f}',
                    'Overall Score': '{:.3f}',
                    'Games Played': '{:.0f}',
                    'Total Minutes': '{:.0f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
            else:
                st.warning(f"Player doesn't have enough games ({min_games_threshold} min) or minutes ({min_minutes_threshold} min) for ranking{scope_label}")

# Footer with additional information
st.markdown("---")
st.markdown("""
### 📌 About This Dashboard
This dashboard provides comprehensive physical performance analytics for MLS players and teams:
- **Playing Style DNA**: Identifies team physical profiles using PCA clustering
- **Fatigue Analysis**: Tracks performance decline across seasons and within games
- **Performance Intelligence**: Correlates physical metrics with team success
- **Player Comparison**: Benchmarks individual players against peers and position averages

**Data Notes**: 
- P90 metrics are normalized per 90 minutes of play
- Home teams are extracted from match strings (first team listed)
- Position-specific benchmarks are calculated from players with 45+ minutes
""")

# Add download functionality for filtered data
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Data")

if st.sidebar.button("Download Filtered Data as CSV"):
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"mls_physical_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
