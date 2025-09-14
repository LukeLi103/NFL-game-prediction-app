import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
import requests
from bs4 import BeautifulSoup
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore")

st.title("Luke's NFL Prediction App")

# Scrape the NFL schedule for the current season
url = "https://www.pro-football-reference.com/years/2025/games.htm"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.pro-football-reference.com/",
}
time.sleep(1)

response = requests.get(url, headers=headers)
# response.raise_for_status()
html = response.text
print(html[:500])

with open("games2025.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# soup = BeautifulSoup(response.text, "html.parser")

# Find the schedule table
table = soup.find("table", id="games")

# Extract headers
headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

# Extract rows
rows = []
for row in table.find("tbody").find_all("tr"):
    if "class" in row.attrs and "thead" in row["class"]:
        continue  # skip mid-table headers
    cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
    rows.append(cells)

# Create DataFrame
sked_df = pd.DataFrame(rows, columns=headers)
sked_df.columns = ['Week', 'Day', 'Date', 'Time', 'Visitor', 'H/A', 'Home', 'Boxscore', 'PtsW', 'PtsL', 'YdsW', 'TOW', 'YdsL', 'TOL']
sked_df = sked_df[['Week', 'Day', 'Date', 'Time', 'Visitor', 'H/A', 'Home']]
sked_df['Matchup String'] = sked_df['Visitor'] + " " + sked_df['H/A'].apply(lambda x: "vs" if x == "" else "@") + " " + sked_df['Home']

# Let the user select a week to filter down the schedule
week = st.sidebar.selectbox('Select a week number', sked_df['Week'].unique())
filtered_sked_df = sked_df[sked_df['Week'] == week]

# Let the user select a specific matchup from the filtered list of matchups
matchup = st.sidebar.selectbox('Select a matchup', filtered_sked_df['Matchup String'].unique())
st.write("The user selected the following matchup:", matchup)

mapping = {
    "kan": "Kansas City Chiefs",
    "phi": "Philadelphia Eagles",
    "atl": "Atlanta Falcons",
    "buf": "Buffalo Bills",
    "nor": "New Orleans Saints",
    "chi": "Chicago Bears",
    "cin": "Cincinnati Bengals",
    "clt": "Indianapolis Colts",
    "mia": "Miami Dolphins",
    "nyg": "New York Giants",
    "sea": "Seattle Seahawks",
    "sdg": "Los Angeles Chargers",
    "cle": "Cleveland Browns",
    "tam": "Tampa Bay Buccaneers",
    "det": "Detroit Lions",
    "sfo": "San Francisco 49ers",
    "car": "Carolina Panthers",
    "jax": "Jacksonville Jaguars",
    "gnb": "Green Bay Packers",
    "dal": "Dallas Cowboys",
    "min": "Minnesota Vikings",
    "nwe": "New England Patriots",
    "was": "Washington Commanders",
    "oti": "Tennessee Titans",
    "rav": "Baltimore Ravens",
    "crd": "Arizona Cardinals",
    "den": "Denver Broncos",
    "htx": "Houston Texans",
    "nyj": "New York Jets",
    "pit": "Pittsburgh Steelers",
    "rai": "Las Vegas Raiders",
    "ram": "Los Angeles Rams"
}

# home and away were flipped on the original dataset
column_names = ['id', 'winner', 'loser', 'winner_score', 'loser_score', 'home_team_first_downs', 'away_team_first_downs', 'home_team_rushes', 'home_team_rushing_yds', 'home_team_rushing_tds', 'away_team_rushes', 'away_team_rushing_yds', 'away_team_rushing_tds', 'home_team_completions', 'home_team_pass_attempts', 'home_team_pass_yds', 'home_team_pass_tds', 'home_team_ints', 'away_team_completions', 'away_team_pass_attempts', 'away_team_pass_yds', 'away_team_pass_tds', 'away_team_ints', 'home_team_sacks', 'home_team_sack_yds', 'away_team_sacks', 'away_team_sack_yds', 'home_team_net_pass_yds', 'away_team_net_pass_yds', 'home_team_total_yds', 'away_team_total_yds', 'home_team_fumbles', 'home_team_fumbles_lost', 'away_team_fumbles', 'away_team_fumbles_lost', 'home_team_turnovers', 'away_team_turnovers', 'home_team_penalties', 'home_team_penalty_yds', 'away_team_penalties', 'away_team_penalty_yds', 'home_team_third_down_conversions', 'home_team_third_down_attempts', 'away_team_third_down_conversions', 'away_team_third_down_attempts', 'home_team_fourth_down_conversions', 'home_team_fourth_down_attempts', 'away_team_fourth_down_conversions', 'away_team_fourth_down_attempts', 'home_team_possession_time', 'away_team_possession_time']
table = pd.read_csv("boxscore_data.csv", header = None, names = column_names)
table = table.drop(index=0)

# converting score columns to numeric
table['winner_score'] = pd.to_numeric(table['winner_score'], errors='coerce').fillna(0)
table['loser_score'] = pd.to_numeric(table['loser_score'], errors='coerce').fillna(0)
for col in table.columns:
  table[col] = table[col].fillna(0)

# adding/engineering new variables
table.insert(loc=4, column='won_by', value=table['winner_score'] - table['loser_score'])
table['home_team_won'] = table.apply(lambda x: mapping[x['id'][-3:]] == x['winner'], axis = 1).astype(int)
# table.insert(loc=4, column="home_team_won", )
st.dataframe(table)

# removing highly correlating variables
a = table.drop(columns=['id', 'loser', 'winner_score', 'loser_score', 'away_team_possession_time', 'home_team_possession_time', 'won_by'], axis=1)
b = a.drop(columns=['winner'], axis=1)

correlation = b.corr().abs()
upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]
table_to_use = a.drop(to_drop, axis=1)
table_to_use.head()

d = table_to_use.drop(columns=['winner', 'home_team_won'], axis=1)
feature_cols = d.columns.tolist()

x = table_to_use[feature_cols]
y = table_to_use.home_team_won

for col in x.columns:
  x[col] = x[col].astype(float)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


logreg = LogisticRegression(random_state=16)
logreg.fit(x_train, y_train)

if "@" in matchup:
    away_team, home_team = matchup.split(" @ ")
elif "vs" in matchup:
    home_team, away_team = matchup.split(" vs ")
else:
    st.error(f"Could not parse matchup string: {matchup}")
    home_team, away_team = None, None

if home_team and away_team:
    home_stats = table[(table['winner'] == home_team) | (table['loser'] == home_team)]
    away_stats = table[(table['winner'] == away_team) | (table['loser'] == away_team)]

    home_avg = home_stats.mean(numeric_only=True)
    away_avg = away_stats.mean(numeric_only=True)

    row = {}
    for col in feature_cols:
        if col.startswith("home_team"):
            row[col] = home_avg.get(col, 0)
        elif col.startswith("away_team"):
            row[col] = away_avg.get(col, 0)
        else:
            row[col] = 0

    matchup_features = pd.DataFrame([row], columns=feature_cols)

    prediction = logreg.predict(matchup_features)
    prediction_prob = logreg.predict_proba(matchup_features)

    st.write("Prediction (1 = home team wins, 0 = away team wins):", int(prediction[0]))
    st.write("Prediction Probability (home win):", prediction_prob[0][1])

y_predlr = logreg.predict(x_test)

cm = confusion_matrix(y_test, y_predlr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

coefficients = logreg.coef_[0]
feature_importance = pd.DataFrame({'Feature': x.columns, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
st.write("Feature Importance based on Logistic Regression Coefficients: ")
st.write(feature_importance)

st.write("Confusion Matrix Array:")
st.write(cm)

st.write("Accuracy:", metrics.accuracy_score(y_test, y_predlr))

results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_predlr,
        'correct': y_test == y_predlr
})

# Add features
results = pd.concat([results, pd.DataFrame(x_test, columns=feature_cols)], axis=1)

# Get misclassified games
misclassified = results[~results['correct']]

st.write(f"Total misclassified games: {len(misclassified)}")
st.write(f"Misclassification rate: {len(misclassified)/len(results):.2%}")

# First, let's enhance the analysis with prediction probabilities
y_pred_proba = logreg.predict_proba(x_test)[:, 1]

# Update results with probabilities
results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_predlr,
    'prediction_prob': y_pred_proba,
    'correct': y_test == y_predlr
})

# Add features
results = pd.concat([results, pd.DataFrame(x_test, columns=feature_cols)], axis=1)
results = results.reset_index(drop=True)

# Get misclassified games with detailed analysis
misclassified = results[~results['correct']]
st.write(f"Total misclassified games: {len(misclassified)}")

# Analyze each misclassified game
for i, (idx, game) in enumerate(misclassified.iterrows(), 1):
    st.write(f"\n=== Misclassified Game {i} ===")
    st.write(f"Actual: {game['actual']}, Predicted: {game['predicted']}")
    st.write(f"Prediction confidence: {game['prediction_prob']:.3f}")

    # Get top 3 features that contributed most to this prediction
    feature_contributions = []
    for feature in feature_cols:
        contribution = game[feature] * feature_importance[feature_importance['Feature'] == feature]['Coefficient'].values[0]
        feature_contributions.append((feature, contribution))

    # Sort by absolute contribution
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    st.write("Top contributing features:")
    for feature, contrib in feature_contributions[:3]:
        st.write(f"  {feature}: {contrib:.4f} (value: {game[feature]:.3f})")

    # Check if key features had extreme values
    extreme_features = []
    for feature in feature_cols:
        feature_mean = x_train[feature].mean()
        feature_std = x_train[feature].std()
        if abs(game[feature] - feature_mean) > 2 * feature_std:
            extreme_features.append((feature, game[feature], feature_mean))

    if extreme_features:
        st.write("Extreme feature values:")
        for feature, value, mean in extreme_features[:2]:
            st.write(f"  {feature}: {value:.3f} (mean: {mean:.3f})")

# Analyze patterns across all misclassifications
st.write("\n=== OVERALL PATTERNS ===")
st.write(f"Average prediction confidence: {misclassified['prediction_prob'].mean():.3f}")
st.write(f"Games where model was very confident but wrong (prob > 0.7): {len(misclassified[misclassified['prediction_prob'] > 0.7])}")
st.write(f"Games where model was uncertain but wrong (prob between 0.4-0.6): {len(misclassified[(misclassified['prediction_prob'] >= 0.4) & (misclassified['prediction_prob'] <= 0.6)])}")



