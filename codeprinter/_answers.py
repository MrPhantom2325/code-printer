"""Central answer store for the codeprinter package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

Answer = Union[str, int, float, dict[str, Any], list[Any], Callable[[], Any]]

ANSWERS: dict[int, Answer] = {
    1: ('''Question 1:
1.Identify missing values using isnull() and notnull() functions and create a strategy for handling them with methods like fillna and dropna. What impact does filling missing values in Occupation and Satisfaction_Level have on analysis results?
2.Apply custom functions to create a transformed dataset—for instance, converting the Satisfaction_Level to a binary variable where values above 0.7 are labeled as "High" and others as "Low."
3.Use map() to convert the Purchase_History categorical data into a numerical format, where "High" is mapped to 2, "Medium" to 1, and "Low" to 0.
4.Apply statistical techniques like the Z-score or IQR method to identify outliers in Income.
5.Assess the impact of missing values in Years_Employed on data reliability. Apply the fillna() method using mean or median imputation.

Answer 1:
import pandas as pd
df=pd.read_csv('lab1.csv')
df.head()

df.describe()
df.info()

missing_occupation=df['Occupation'].isnull().sum()
missing_satisfaction=df["Satisfaction_Level"].isnull().sum()
print(missing_occupation)
print(missing_satisfaction)

df['Occupation'].fillna(df['Occupation'].mode()[0],inplace=True)
df['Satisfaction_Level'].fillna(df['Satisfaction_Level'].mean(),inplace=True)
df

df["Satisfaction_Binary"]=df["Satisfaction_Level"].apply(lambda x: "High" if x>0.7 else "Low")
df


mapping_values={'Low':0,'Medium':1,'High':2}
df['Purchase_History']=df['Purchase_History'].map(mapping_values)
df

q1=df['Income'].quantile(0.25)
q3=df['Income'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
outliers= df[(df['Income']<lower_bound)| (df['Income']>upper_bound)]
print("outliers in INCOME column:")
print(f'{outliers}',end='')

df['Years_Employed'].fillna(df['Years_Employed'].median(),inplace=True)
'''),
    2: ('''Question 2:
1.Apply a function that fills missing values in the Age column with the mean age and in the City column with the value "Unknown."
2.Apply a function that removes duplicates based on all columns in the DataFrame.
3.Write a function that replaces inconsistent values in the Gender column (e.g., replace "M" with "Male" and "F" with "Female").
4.Write a function that groups values in the Age column into ranges, such as "18-30," "30-40," and "40-50."
5.Write a function that converts the City column into dummy variables.

Answer 2:
import pandas as pd
data=pd.read_csv("Lab2.csv")
df=pd.DataFrame(data)
df
df.describe()
df.info()

df['Age'].fillna(df['Age'].mean(),inplace=True)
df['City'].fillna('Unknown',inplace=True)
df

df.drop_duplicates(inplace=True)
df

values={'Male':'M','Female':'F'}
df['Gender']=df['Gender'].replace(values)
df

bins=[0,10,20,30,40,50]
labels=['0-10','10-20','20-30','30-40','40-50']
df['Age_Group']=pd.cut(df['Age'],bins=bins,labels=labels)
df

x=pd.get_dummies(df,columns=['City'])
x
'''),
    3:('''
Question 3:
1.Use the Product and Month columns in the Sales Data DataFrame to create a hierarchical index, and describe the benefits of this structure, such as more efficient subsetting and grouping operations.
2.Merge the Sales Data and Customer Feedback DataFrames on OrderID using various joins. Explain when each join type (e.g., inner for shared data, outer for all data, etc.) is particularly beneficial.
3.Concatenate a list of Sales DataFrames (e.g., data from different quarters) both vertically (stacking rows) and horizontally (adding new columns). Discuss the implications of each approach in combining data.
4.Write a function that combines the Sales Data and Customer Feedback DataFrames, filling any missing Feedback_Score values in one with those from the other. This is useful in cases where multiple sources have partial data for the same records
5.Write a function that pivots the Sales Data based on Month, with Product as rows and Sales as values, so that each month’s sales become separate columns. This transformation can help in comparing monthly sales trends across products.

Answer 3:

import pandas as pd
import numpy as np
sales_df = pd.read_csv('sales_data.csv')
feedback_df = pd.read_csv('customer_feedback.csv')

sales_df.head()
feedback_df.head()

def create_hierarchical_index(df):
  
    if 'Product' in df.columns and 'Month' in df.columns:
        df_hierarchical = df.set_index(['Product', 'Month'])
        return df_hierarchical
    else:
        print("Required columns 'Product' and/or 'Month' not found")
        return df

sales_hierarchical = create_hierarchical_index(sales_df)
print(f"\n{sales_hierarchical}")


print("\nHierarchical Index Example - Subsetting:")
user_product = input("Enter a product name to view its sales: ")
product_sales = sales_hierarchical.xs(user_product, level='Product')
print(f"Sales for {user_product}:\n{product_sales.head()}")


def demonstrate_merges(sales_df, feedback_df):
    print("\nMerge Operations:\n")

    inner_merged = pd.merge(sales_df, feedback_df, on='OrderID', how='inner')
    print(f"\n{inner_merged}\n")

    left_merged = pd.merge(sales_df, feedback_df, on='OrderID', how='left')
    print(f"\n{left_merged}\n")

    right_merged = pd.merge(sales_df, feedback_df, on='OrderID', how='right')
    print(f"\n{right_merged}\n")

    outer_merged = pd.merge(sales_df, feedback_df, on='OrderID', how='outer')
    print(f"\n{outer_merged}\n")

    return inner_merged, left_merged, right_merged, outer_merged

inner_result, left_result, right_result, outer_result = demonstrate_merges(sales_df, feedback_df)
    
# 3. Concatenate DataFrames
def demonstrate_concatenation():
    print("\nConcatenation Operations:\n")

    quarter1 = sales_df.iloc[:1].copy()
    quarter2 = sales_df.iloc[1:3].copy()
    quarter3 = sales_df.iloc[3:].copy()
    print(f"\nQuarter-1{quarter1}\n")
    print(f"\nQuarter-2{quarter2}\n")
    print(f"\nQuarter-3{quarter3}\n")

    quarter1['Quarter'] = 'Q1'
    quarter2['Quarter'] = 'Q2'
    quarter3['Quarter'] = 'Q3'


    vertical_concatenated = pd.concat([quarter1, quarter2, quarter3], axis=0)
    print(f"Vertical Concatenation: {vertical_concatenated.shape} rows")
    print(f"{vertical_concatenated}\n")


    sales_metrics = sales_df[['OrderID', 'Sales']].set_index('OrderID')
    product_metrics = sales_df[['OrderID', 'Product']].set_index('OrderID')
    print(f"\n{sales_metrics}\n")
    print(f"\n{product_metrics}\n")

    horizontal_concatenated = pd.concat([sales_metrics, product_metrics], axis=1)
    print(f"Horizontal Concatenation: {horizontal_concatenated.shape} columns")
    print(f"{horizontal_concatenated}\n")

    return vertical_concatenated, horizontal_concatenated

vertical_result, horizontal_result = demonstrate_concatenation()



def combine_with_feedback_fill(sales_df, feedback_df):
    print("\nCombining Data with Missing Value Filling:")
    combined_df = pd.merge(sales_df, feedback_df, on='OrderID', how='left')

    if 'Feedback_Score' in combined_df.columns:
        combined_df['Feedback_Score'] = combined_df['Feedback_Score'].fillna(0)
        print("Missing Feedback_Score values filled with 0")
    else:
        print("Feedback_Score column not found in merged data")

    return combined_df

combined_data = combine_with_feedback_fill(sales_df, feedback_df)
print(f"\n{combined_data}")

# 5. Pivot Table Function
def create_sales_pivot(df):
    print("\nPivot Table Creation:")
    required_cols = ['Product', 'Month', 'Sales']

    try:
        pivot_df = df.pivot_table(
            index='Product',
            columns='Month',
            values='Sales',
            fill_value=0
        )

        print("Pivot Table created successfully!")
        return pivot_df

    except Exception as e:
        print(f"Error creating pivot table: {e}")
        return None

pivot_table = create_sales_pivot(sales_df)

if pivot_table is not None:
    print(f"\nPivot Table Shape: {pivot_table.shape}")
    print("\nPivot Table:")
    print(pivot_table)

    # Calculate and print sales sums
    print("\n" + "="*50)
    print("SALES SUMMARY")
    print("="*50)

    # Total sales per product (row sums)
    print("\nTotal Sales by Product:")
    product_totals = pivot_table.sum(axis=1)
    for product, total in product_totals.items():
        print(f"  {product}: Rs.{total:,.2f}")

    # Total sales per month (column sums)
    print("\nTotal Sales by Month:")
    month_totals = pivot_table.sum(axis=0)
    for month, total in month_totals.items():
        print(f"  {month}: Rs.{total:,.2f}")

    # Grand total
    grand_total = pivot_table.sum().sum()
    print("="*50)
    print(f"\nGrand Total Sales: Rs.{grand_total:,.2f}")
    print("="*50)

print("\nANALYSIS COMPLETE")
    '''),
    5:('''
Question 5:
1.Pivot the data to calculate the total revenue generated by each salesperson on each date.
2.Find the average revenue per sale for each product.
3.Find the maximum number of units sold in a single transaction by each salesperson.
4.Show the percentage of total revenue contributed by each region.
5.Determine which salesperson completed the most sales transactions (count of unique transactions)
6.Pivot the data to show both the total revenue and total units sold by each salesperson for each product.
7.Create a pivot table to display the total units sold in each region on each date.
   
Answer 5:
import pandas as pd
df=pd.read_csv('simple_sales_data.csv')
df


print('Total Revenue by Salesperson on Each Date:')
revenue_by_salesperson_date=df.pivot_table(values='revenue',index='date',columns='salesperson',aggfunc='sum',fill_value=0)
print(revenue_by_salesperson_date)

print('Average Revenue per sale for each Product:')
avg_revenue_product=df.groupby('product')['revenue'].mean().round(2)
print(avg_revenue_product)

print('Maximum number of Units Sold in a single transaction by each salesperson:')
max_units_sold_salesperson=df.groupby('salesperson')['units_sold'].max()
print(max_units_sold_salesperson)

print('Percentage of total revenue contributed by each region')
revenue_region=df.groupby('region')['revenue'].sum()
total_revenue=df['revenue'].sum()
percentage_revenue_region=(revenue_region/total_revenue*100).round(2)
print(percentage_revenue_region)

print('Sales Person ehich compeleted the most sales transaction')
max_sales=df.groupby('salesperson')['transaction_id'].count()
print(f"{max_sales.idxmax()} completed most transactions {max_sales.max()}")
print(max_sales)

print('pivot table to show both total revenue and total units sold by each salesperson for each product')
pivot_table=df.pivot_table(values=['revenue','units_sold'],index='salesperson',columns='product',aggfunc='sum',fill_value=0)
print(pivot_table)

print('pivot table to display total units sold in each region on each date')
pivot_table=df.pivot_table(values='units_sold',index='date',columns='region',aggfunc='sum',fill_value=0)
print(pivot_table)
    '''),
    6:('''
Question 6:
1.How did the team's points score change over the season?
2.What was the average attendance for the games throughout the season?
3.Which player scored the most points over the season? Create a bar chart showing the total points scored by each player.
4.How many games did the team score above a certain threshold
(e.g., 100 points)? Use a bar chart to show the number of games scored in different ranges (e.g., 80-90, 90-100, 100-110, etc.).
5.Which opponents did the team perform best against? Create a bar chart showing the points scored against each opponent.
6.How does the attendance of games compare for different opponents? Use a bar chart to visualize attendance figures based on different opponents.
7.How does the team's win-loss record compare to points scored? Create a grouped bar chart with wins and losses on one side and average points scored on the other.

Answer 6:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

games_df = pd.read_csv('basketball_games.csv')
players_df = pd.read_csv('basketball_players.csv')

games_df['Date'] = pd.to_datetime(games_df['Date'])

players_df

print("1. Team's Points Score Over the Season:")
plt.figure(figsize=(12, 6))
plt.plot(games_df['Date'], games_df['Team_Points'], marker='X', linewidth=2, markersize=8)
plt.title("Team's Points Score Over the Season", fontsize=10, fontweight='bold')
plt.xlabel('Date', fontsize=10)
plt.ylabel('Points Scored', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout() # - Minimizes the overlapping of tiles
plt.show()
    
# Calculate trend uisng linear regression
trend = np.polyfit(range(len(games_df)), games_df['Team_Points'], 1)[0]
trend_direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
print(f"   Points trend: {trend_direction} (slope: {trend:.2f})")

print(f"\n2. Average Attendance: {games_df['Attendance'].mean():.0f} people")
print(f"   Highest Attendance: {games_df['Attendance'].max():.0f} people")
print(f"   Lowest Attendance: {games_df['Attendance'].min():.0f} people")
plt.figure(figsize=(10, 6))
plt.hist(games_df['Attendance'], bins=8, alpha=0.7, edgecolor='black')
plt.title('Distribution of Game Attendance', fontsize=16, fontweight='bold')
plt.xlabel('Attendance', fontsize=12)
plt.ylabel('Number of Games', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

player_points = players_df.groupby('Player')['Points'].sum().sort_values(ascending=False)
top_scorer = player_points.index[0]
top_score = player_points.iloc[0]

print(f"\n3. Top Scorer: {top_scorer} with {top_score} total points")

plt.figure(figsize=(10, 6))
player_points.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Total Points Scored by Each Player', fontsize=16, fontweight='bold')
plt.xlabel('Player', fontsize=12)
plt.ylabel('Total Points', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


threshold = 100
games_above_threshold = len(games_df[games_df['Team_Points'] > threshold])
print(f"\n Games scoring above {threshold} points: {games_above_threshold}/{len(games_df)}")

# Create point ranges
bins = [80, 90, 100, 110, 120]
labels = ['80-89', '90-99', '100-109', '110-119']
games_df['Point_Range'] = pd.cut(games_df['Team_Points'], bins=bins, labels=labels, right=False)
point_range_counts = games_df['Point_Range'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
point_range_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Games by Points Scored Range', fontsize=16, fontweight='bold')
plt.xlabel('Points Range', fontsize=12)
plt.ylabel('Number of Games', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


opponent_performance = games_df.groupby('Opponent').agg({
    'Team_Points': 'mean',
    'Result': lambda x: (x == 'Win').sum()
}).round(1)
opponent_performance.columns = ['Avg_Points_Against', 'Wins']
opponent_performance = opponent_performance.sort_values('Avg_Points_Against', ascending=False)
print(f"\n Team Performance Against Opponents (by average points scored):")
for opponent, row in opponent_performance.iterrows():
    print(f"   {opponent}: {row['Avg_Points_Against']} avg points, {row['Wins']} wins")
plt.figure(figsize=(12, 6))
opponent_performance['Avg_Points_Against'].sort_values().plot(kind='barh', color='Blue')
plt.title('Average Points Scored Against Each Opponent', fontsize=16, fontweight='bold')
plt.xlabel('Average Points Scored', fontsize=12)
plt.ylabel('Opponent', fontsize=12)
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

opponent_attendance = games_df.groupby('Opponent')['Attendance'].mean().sort_values(ascending=False)
print(f"\n Average Attendance by Opponent:")
for opponent, attendance in opponent_attendance.items():
    print(f"   {opponent}: {attendance:.0f} average attendance")
plt.figure(figsize=(12, 6))
opponent_attendance.sort_values().plot(kind='barh', color='orange')
plt.title('Average Attendance by Opponent', fontsize=16, fontweight='bold')
plt.xlabel('Average Attendance', fontsize=12)
plt.ylabel('Opponent', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()


win_loss_stats = games_df.groupby('Result').agg({
    'Team_Points': ['count', 'mean'],
    'Game_ID': 'count'
}).round(1)
win_loss_stats.columns = ['Games_Count', 'Avg_Points', 'Total_Games']
win_loss_stats = win_loss_stats.reset_index()
print(f"\n Win-Loss Record vs Points Scored:")
for _,row in win_loss_stats.iterrows():
    print(f"   {row['Result']}: {row['Games_Count']} games, {row['Avg_Points']} avg points")
# Create grouped bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Win-Loss record
results = win_loss_stats['Result']
game_counts = win_loss_stats['Games_Count']
colors = ['green' if result == 'Win' else 'red' if result == 'Loss' else 'gray' for result in results]
ax1.bar(results, game_counts, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Win-Loss-Tie Record', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Games', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
# Average points by result
avg_points = win_loss_stats['Avg_Points']
ax2.bar(results, avg_points, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('Average Points by Game Result', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Points', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

    '''),
    7:('''
    
    Question 7:
This dataset provides the number of Uber pickups in different regions of  New York City, categorized by date and time. 
1. Create a heatmap to visualize Uber pickups by day of  the week and hour of the day. 
2. Use a line chart to show the trend of Uber pickups  across a specific month. 
3. Create a bubble chart to display the number of pickups in different regions of New York City, with the size of the bubbles representing the total number  of pickups.

Answer 7:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "uber - uber.csv"
df = pd.read_csv(file_path)
df.head()


df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
import numpy as np
np.random.seed(42)
df['hour'] = np.random.randint(0, 24, df.shape[0])
df['day_of_week'] = df['date'].dt.day_name()


heatmap_data = df.groupby(['day_of_week', 'hour'])['trips'].sum().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlOrRd")
plt.title("Heatmap of Uber Pickups by Day of Week and Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.show()


df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
january_df = df[(df['month'] == 1) & (df['year'] == 2015)]
daily_trend = january_df.groupby('date')['trips'].sum()
plt.figure(figsize=(12, 6))
plt.plot(daily_trend.index, daily_trend.values, marker='o')
plt.title("Trend of Uber Pickups in January 2015")
plt.xlabel("Date")
plt.ylabel("Total Pickups")
plt.grid(True)
plt.show()


region_data = df.groupby('dispatching_base_number')['trips'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.scatter(region_data['dispatching_base_number'], region_data['trips'],
            s=region_data['trips'] / 10, alpha=0.6, edgecolors="w")
plt.title("Bubble Chart of Uber Pickups by Region (Base Number)")
plt.xlabel("Region (Dispatching Base Number)")
plt.ylabel("Total Pickups")
plt.xticks(rotation=45)
plt.show()
    '''),
    8:('''
Question 8:
The Titanic dataset contains information about passengers, such as age,  sex, passenger class, and whether they survived. 
1. Create a bar chart to visualize the survival rate based  on passenger class (Pclass). 
2. Use a pie chart to show the proportion of survivors vs.  non-survivors in the dataset. 
3. Create a stacked bar chart to compare the number of  survivors and non-survivors based on passenger class  and sex.

Answer 8:
import pandas as pd
df=pd.read_csv("titanic.csv")
df.head()

#Use only for the columns that have a value of 0
zero_cols=[col for col in df.columns if 'zero' in col]
df=df.drop(zero_cols,axis=1)
df.head()

import matplotlib.pyplot as plt
survival_rate = df.groupby('Pclass')['Survived'].mean()
plt.figure(figsize=(8, 6))
survival_rate.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

survival_counts = df['Survived'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(survival_counts, autopct='%.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue'])
plt.title('Proportion of Survivors vs. Non-Survivors')
plt.legend(['Non-Survivors', 'Survivors'], loc='best')
plt.axis('equal')
plt.show()

import matplotlib.pyplot as plt
survival_by_pclass_sex = df.groupby(['Pclass', 'Sex'])['Survived'].value_counts().unstack(fill_value=0)
survival_by_pclass_sex.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Survival vs. Non-Survival by Pclass and Sex')
plt.xlabel('Passenger Class and Sex')
plt.ylabel('Number of Passengers')
labels = [f'PClass {p} - Sex {s}' for p, s in survival_by_pclass_sex.index]
plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')
plt.legend(title='Survival Status', labels=['Non-Survivors', 'Survivors'])
plt.tight_layout()
plt.show()
    '''),
    9:('''
This dataset contains various features of houses (e.g., square  footage, number of rooms, neighborhood, etc.) and their corresponding sale prices. 
1. Create a scatter plot to visualize the relationship  between GrLivArea (above ground living area) and SalePrice. 
2. Use a heatmap to visualize the correlation  between numerical features like GrLivArea,OverallQual, TotalBsmtSF, etc. 
3. Create a bubble chart to represent the  relationship between GrLivArea, SalePrice, and YearBuilt, where the size of the bubbles corresponds to OverallQual

Answer 9:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
house_df = pd.read_csv("Dataset_9.csv")
house_df.head()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=house_df, alpha=0.6)
plt.title('Scatter Plot: GrLivArea vs SalePrice')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.grid(True)
plt.show()


numerical_features = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'GarageArea', 'SalePrice']
corr_matrix = house_df[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Key Numerical Features')
plt.show()



plt.figure(figsize=(12, 7))
bubble_size = house_df['OverallQual'] * 20
plt.scatter(house_df['GrLivArea'], house_df['SalePrice'],
            s=bubble_size, c=house_df['YearBuilt'], cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('Bubble Chart: GrLivArea vs SalePrice (Bubble Size = OverallQual, Color = YearBuilt)')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.colorbar(label='Year Built')
plt.grid(True)
plt.show()
    '''),
    10:('''
This dataset includes data on FIFA 21 players, including attributes like  overall rating, potential, position, and club. 
1. Create a bar chart to compare the number of players across  different positions (e.g., forwards, midfielders, defenders). 
2. Use a donut chart to show the distribution of player ratings (Overall) by player position (e.g., how many forwards,  defenders, etc., fall into different rating ranges). 
3. Create a tree diagram to visualize the hierarchical  relationship between the overall ratings and players'  positions or clubs. 
4. What do the bar chart and donut chart reveal about the  distribution of player positions and ratings? How does the  tree diagram help visualize the organizational structure of  player data?

Answer 10:
import pandas as pd
import matplotlib.pyplot as plt
import  plotly.express as px
fifa_df = pd.read_csv("Dataset_10.csv")

def simplify_position(pos):
    if isinstance(pos, str):
        if 'GK' in pos:
            return 'Goalkeeper'
        elif any(x in pos for x in ['ST', 'LW', 'RW', 'CF']):
            return 'Forward'
        elif any(x in pos for x in ['CM', 'LM', 'RM', 'CAM', 'CDM']):
            return 'Midfielder'
        elif any(x in pos for x in ['CB', 'LB', 'RB', 'LWB', 'RWB']):
            return 'Defender'
    return 'Other'

fifa_df['Position_Group'] = fifa_df['player_positions'].apply(simplify_position)

position_counts = fifa_df['Position_Group'].value_counts()
plt.figure(figsize=(10, 6))
position_counts.plot(kind='bar', color='royalblue')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



rating_counts = fifa_df['Position_Group'].value_counts()
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(rating_counts, labels=rating_counts.index,
                                   autopct='%1.1f%%', startangle=90, pctdistance=0.85)
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()



subset = fifa_df[['club_name', 'Position_Group', 'overall']].dropna()
fig = px.treemap(subset,
                 path=['Position_Group', 'club_name'],
                 values='overall',
                 color='overall',
                 color_continuous_scale='Viridis',
                 title='Tree Map: Relationship Between Player Ratings, Positions, and Clubs')
fig.show()

    ''')
}

