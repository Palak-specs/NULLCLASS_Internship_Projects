import pandas as pd
import nltk as nl
from nltk.sentiment import SentimentIntensityAnalyzer

from dash import Dash, html, dcc
import dash

import pytz

from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt
apps_df= pd.read_csv('C:/Users/User/Downloads/Play Store Data.csv')

reviews_df= pd.read_csv('C:/Users/User/Downloads/User Reviews.csv')
apps_df.head()
reviews_df.head()
apps_df.isnull()
apps_df.dropna()
reviews_df.isnull()
reviews_df.dropna()
apps_df.duplicated()
reviews_df.duplicated()
apps_df.drop_duplicates()
reviews_df.drop_duplicates()
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.dropduplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Reviews'],inplace=True)
apps_df=apps_df[apps_df['Rating']<5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)
apps_df['Installs'] = apps_df['Installs'].astype(str)  
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '') \
                                         .str.replace('+', '', regex=False)

apps_df['Installs'] = pd.to_numeric(apps_df['Installs'], errors='coerce') 

apps_df['Price'] = apps_df['Price'].astype(str)

apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=False)

apps_df['Price'] = pd.to_numeric(apps_df['Price'], errors='coerce')

apps_df.dtypes
merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner' )
merged_df.head()
def convert_size(size):
    if'M' in size:
        return float (size.replace('M',''))
    elif"k" in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan

apps_df['Size'] = apps_df['Size'].apply(lambda x: float(x.replace('M', '')) if isinstance(x, str) and 'M' in x else np.nan)

apps_df['Installs'] = pd.to_numeric(apps_df['Installs'], errors='coerce')
apps_df['Reviews'] = pd.to_numeric(apps_df['Reviews'], errors='coerce')

apps_df['Log_installs'] = np.log(apps_df['Installs'] + 1)
apps_df['Log_Reviews'] = np.log(apps_df['Reviews'] + 1)

apps_df['Log_installs']=np.log(apps_df['Installs'])
apps_df['Log_Reviews']=np.log(apps_df['Reviews'])
def rating_group (rating):
    if rating>=4:
        return 'Top rated app'
    elif rating >=3:
        return'Above avarage'
    elif rating>=2:
        return 'Average'
    else :
        return 'Below average'
apps_df['Rating group']=apps_df['Rating'].apply(rating_group)
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer() 
review= "This app is boring! I hate the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)
reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
reviews_df.head()
apps_df['Last Updated']= pd.to_datetime(apps_df['Last Updated'],errors='coerce')
apps_df['Year']=apps_df['Last Updated'].dt.year

apps_df.head()
import os

 html_files_path=",/"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)
plot_containers=""
import os
import plotly.io as pio

def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    
    plot_containers += f"""
    <div class="plot_container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

plot_containers=""
import plotly.express as px

# Get top 10 categories
Category_counts = apps_df['Category'].value_counts().nlargest(10)

# Create bar plot
fig1 = px.bar(
    x=Category_counts.index,
    y=Category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=Category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)

# Update layout
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=30, r=30, t=30, b=30)
)

fig1.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig1,"Category Graph 1.html",  "The top categories on the Play store dominated by tools, entertainment and productivityapps")

fig1.write_html("fig1_interactive.html")

#figure 2
type_counts = apps_df['Type'].value_counts()

# Create bar plot
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)

# Update layout
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    margin=dict(l=10, r=10, t=30, b=10)
)

#fig1.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig2,"Type Graph 2.html",  "Most apps on the Playstore are free,indicating a strategy to attract users first and monetize through ads or in app purchases")

fig2.write_html("fig2_interactive.html")

fig2.show()
#fig3
type_counts = apps_df['Type'].value_counts()

# Create bar plot
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)

# Update layout
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

#fig1.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig3,"Rating Graph 3.html",  "Ratings are skewed towards higher values.")
fig3.write_html("fig3_interactive.html")
fig3.show()
sentiment_counts=reviews_df['Sentiment_Score'].value_counts()

# Create bar plot
fig4 = px.bar(
    x=Category_counts.index,
    y=Category_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    #color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)

# Update layout
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

#fig4.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig4,"Sentiment Graph 4.html",  "Sentiments in Reviews show a mix of positive and negative feedback")
fig4.write_html("fig4_interactive.html")
install_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)

# Create bar plot
fig5 = px.bar(
    x=install_by_category.index,
    y=install_by_category.values,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    #color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)

# Update layout
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

#fig4.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig5,"Installs by Category Graph 5.html",  "The categorywith the most installs are social and communication apps")
fig5.write_html("fig5_interactive.html")
updates_per_year=apps_df['Last Updated'].dt.year.value_counts().sort_index()


fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of updates over the years ',
    #color=sentiment_counts.index,
    color_discrete_sequence=['#AB63FA'],
    width=800,
    height=500
)


fig6.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=50, r=30, t=30, b=30)
)


save_plot_as_html(fig6,"Upadates Graph 6.html",  "Updates have been increasing over the years, showing that developers actively maintaining and improving theeir apps.")
fig6.write_html("fig6_interactive.html")
import plotly.express as px


revenue_by_category= apps_df.groupby('Category')['Revenue'].sum().nlargest(10)


fig7 = px.bar(
    x=Category_counts.index,
    y=Category_counts.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Top Categories on Play Store',
    color=Category_counts.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)


fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

fig7.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig7,"Revenue Graph 7.html",  "The top categories on the Play store dominated by tools, entertainment and productivityapps")
fig7.write_html("fig7_interactive.html")
genre_counts= apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)

fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=Category_counts.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=800,
    height=500
)


fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=50, r=30, t=30, b=60)
)

fig8.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig8,"Genre Graph 8.html",  "Action and casual genres are the most common, reflecting users' preferences for engaging and easy-to-play games")
fig8.write_html("fig8_interactive.html")



fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=800,
    height=500
)


fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=50, r=50, t=50, b=50)
)

fig9.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig9,"Sctter Graph 7.html",  "The scatter plot shows a weak correlation between the last updatae and ratings,suggesting that more frequent updates don't alwaysresult in better ratings")
fig9.write_html("fig9_interactive.html")
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for paid vs free apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=800,
    height=500
)


fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    title_font=dict(size=16),
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=50, r=50, t=50, b=50)
)

fig10.update_traces(marker=dict(line=dict(color='white',width=1)))
save_plot_as_html(fig10,"Paid vs Free Graph 7.html",  "Paid apps generally have high ratings compare to free apps")
fig10.write_html("fig10_interactive.html")
df = apps_df[apps_df['Reviews'] > 1000]

def rating_bin(rating):
    if rating >= 1 and rating < 2:
        return '1-2 Stars'
    elif rating >= 2 and rating < 3:
        return '1-2 Stars'  
    elif rating >= 3 and rating < 4:
        return '3-4 Stars'
    elif rating >= 4 and rating <= 5:
        return '4-5 Stars'
    else:
        return 'Other'

apps_df['Rating_Stars'] = apps_df['Rating'].apply(rating_bin)

merged_df['Rating_Stars'] = merged_df['Rating'].apply(lambda x: round(x) if pd.notna(x) else None)


category_counts = merged_df.groupby('Category')['Rating_Stars'].count().reset_index(name='Total_Count')
category_counts = category_counts.sort_values(by='Total_Count', ascending=False)
top_5_categories = category_counts.head(5)['Category']

filtered_df = merged_df[merged_df['Category'].isin(top_5_categories)]

grouped = filtered_df.groupby(['Category', 'Rating_Stars', 'Sentiment']).size().reset_index(name='Count')

pivot_df = grouped.pivot_table(index=['Category', 'Rating_Stars'], columns='Sentiment', values='Count', fill_value=0).reset_index()

import plotly.express as px

fig11 = px.bar(pivot_df,
             x='Category',
             y=['Positive', 'Neutral', 'Negative'],
             color_discrete_map={
                 'Positive': 'green',
                 'Neutral': 'gray',
                 'Negative': 'red'
             },
             title='Sentiment Distribution Across Top 5 Categories and Rating Stars',
             barmode='stack',
             labels={'value': 'Review Count', 'variable': 'Sentiment'}
)

fig11.update_layout(xaxis={'categoryorder': 'total descending'})
fig11.show()

fig11.write_html("top_5_categories_stacked_sentiment_chart.html")

fig11.write_html("fig11_interactive.html")
df = df[~df['Android Ver'].str.contains('Varies', na=False)].copy()
df['Android Ver Cleaned'] = df['Android Ver'].str.extract(r'(\d+\.\d+)')[0]
df['Android Ver Cleaned'] = df['Android Ver Cleaned'].astype(float)





def size_to_mb(size):
    if pd.isnull(size) or size == 'Varies with device':
        return np.nan
    size = str(size).strip().upper()
    if size.endswith('M'):
        return float(size[:-1])
    elif size.endswith('K'):
        return float(size[:-1]) / 1024
    else:
        return np.nan

df['Size Cleaned'] = df['Size'].apply(size_to_mb)

import plotly.graph_objects as go

print(filtered_df['Type'].value_counts())


top_categories = df['Category'].value_counts().nlargest(3).index.tolist()


top_df = df[df['Category'].isin(top_categories)]


free_apps = top_df[
    (top_df['Installs'] >= 10000) &
    (top_df['Android Ver Cleaned'] > 4.0) &
    (top_df['Size Cleaned'] > 15) &
    (top_df['Content Rating'] == 'Everyone') &
    (top_df['App'].str.len() <= 30) &
    (top_df['Type'] == 'Free')
]


paid_apps = top_df[
    (top_df['Installs'] >= 10000) &
    (top_df['Android Ver Cleaned'] > 4.0) &
    (top_df['Size Cleaned'] > 15) &
    (top_df['Content Rating'] == 'Everyone') &
    (top_df['App'].str.len() <= 30) &
    (top_df['Type'] == 'Paid') &
    (top_df['Revenue'] >= 10000)  
]


combined_df = pd.concat([free_apps, paid_apps])


grouped = combined_df.groupby(['Category', 'Type']).agg({
    'Installs': 'mean',
    'Revenue': 'mean'
}).reset_index()
def is_display_time():
   ist = pytz.timezone('Asia/Kolkata')
   now = datetime.now(ist)
   return now.hour == 13 


if is_display_time():
    fig12 = go.Figure()

    fig12 = go.Figure()


    for category in top_categories:
        cat_df = grouped[grouped['Category'] == category]

 
        fig12.add_trace(go.Bar(
            x=cat_df['Type'],
            y=cat_df['Installs'],
            name=f'{category} - Avg Installs',
            yaxis='y1'
        ))

    
        fig12.add_trace(go.Scatter(
            x=cat_df['Type'],
            y=cat_df['Revenue'],
            name=f'{category} - Avg Revenue',
            yaxis='y2',
            mode='lines+markers'
        ))

        fig12.update_layout(
            title="Avg Installs vs Revenue for Free vs Paid Apps (Top 3 Categories)",
            xaxis_title="App Type",
            yaxis=dict(title="Average Installs"),
            yaxis2=dict(
                title="Average Revenue",
                overlaying='y',
                side='right'
        ),
        barmode='group'
    )


    fig12.show()
    fig12.write_html("Avg Installs vs Revenue for Free vs Paid Apps (Top 3 Categories)") 
else:
    print("Dual-axis chart is only available between 1 PM to 2 PM IST.")

import streamlit as st

df['Installs'] = df['Installs'].astype(float)
df['Size'] = df['Size'].astype(float)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')


filtered_df = df[
    (df['Rating'] >= 4.0) &
    (df['Size'] > 10) &
    (df['Last Updated'].dt.month == 1)
]


top_categories = (
    filtered_df.groupby('Category')['Installs']
    .sum()
    .nlargest(10)
    .index.tolist()
)


top_df = filtered_df[filtered_df['Category'].isin(top_categories)]


agg_df = top_df.groupby('Category').agg({
    'Rating': 'mean',
    'Reviews': 'sum'
}).reset_index()


fig13 = go.Figure()


fig13.add_trace(go.Bar(
    x=agg_df['Category'],
    y=agg_df['Reviews'],
    name='Total Reviews',
    yaxis='y1',
    marker_color='steelblue',
    offsetgroup=0  
))


fig13.add_trace(go.Bar(
    x=agg_df['Category'],
    y=agg_df['Rating'],
    name='Average Rating',
    yaxis='y2',
    marker_color='orange',
    offsetgroup=1  
))


fig13.update_layout(
    title='Total Reviews vs Average Rating for Top 10 App Categories',
    xaxis=dict(title='App Category'),
    yaxis=dict(
        title='Total Reviews',
        side='left',
        showgrid=False
    ),
    yaxis2=dict(
        title='Average Rating',
        side='right',
        overlaying='y',
        range=[0, 5],
        showgrid=False
    ),
    barmode='group',
    legend=dict(x=0.5, y=1.15, orientation='h', xanchor='center'),
    margin=dict(t=150),
    height=600
)


ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.now(ist).time()


start_time = datetime.strptime("15:00", "%H:%M").time()
end_time = datetime.strptime("17:00", "%H:%M").time()


if start_time <= current_time_ist <= end_time:
    
    fig.show()
else:
    print(" This chart is available only between 3 PM and 5 PM IST.")



fig13.write_html("fig13_interactive.html")
import seaborn as sns



india_tz = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(india_tz)
current_hour = current_time.hour


if 16 <= current_hour < 18:
    print(" Displaying graph: Time is within 4 PM to 6 PM IST.")

   
    df = df[df['App'].str.contains('C', case=False, na=False)]
    df['Reviews'] = pd.to_numeric(
    df['Reviews'].astype(str).str.replace(',', '', regex=False),
    errors='coerce'
).fillna(0).astype(int)


    df = df[df['Reviews'] >= 10]
    df = df[df['Rating'] < 4.0]

   
    category_counts = df['Category'].value_counts()
    valid_categories = category_counts[category_counts > 50].index
    filtered_df = df[df['Category'].isin(valid_categories)]

    
    if not filtered_df.empty:
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=filtered_df, x='Category', y='Rating')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Ratings by App Category\n(Apps with "C" in Name, >50 per Category, Rating < 4.0, ≥10 Reviews)')
        plt.tight_layout()
        plt.savefig("fig14_violinplot.png", dpi=300)              
        plt.show()
    else:
        print(" No data available after filtering to display the violin plot.")
else:
    print(" Graph hidden: Outside allowed time window (4 PM to 6 PM IST).")

from datetime import datetime
import pytz
import matplotlib.pyplot as plt


ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
current_hour = now_ist.hour


if 18 <= current_hour < 21:
    df_filtered = df[
        (~df['App'].str.lower().str.startswith(('x', 'y', 'z'))) &
        (df['Category'].str.startswith(('E', 'C', 'B'))) &
        (df['Reviews'] > 500)
    ]

   
    df_filtered['Last Updated'] = pd.to_datetime(df_filtered['Last Updated'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Last Updated'])  # Drop rows where date conversion failed
    df_filtered['YearMonth'] = df_filtered['Last Updated'].dt.to_period('M')

   
    monthly_installs = df_filtered.groupby(['YearMonth', 'Category'])['Installs'].sum().reset_index()
    monthly_installs['YearMonth'] = monthly_installs['YearMonth'].dt.to_timestamp()

   
    monthly_installs['Previous_Installs'] = monthly_installs.groupby('Category')['Installs'].shift(1)
    monthly_installs['MoM_Growth'] = (
        (monthly_installs['Installs'] - monthly_installs['Previous_Installs']) /
        monthly_installs['Previous_Installs']
    )

    
    plt.figure(figsize=(12, 6))
    categories = monthly_installs['Category'].unique()

    for cat in categories:
        cat_data = monthly_installs[monthly_installs['Category'] == cat]
        plt.plot(cat_data['YearMonth'], cat_data['Installs'], label=cat)

        
        for i in range(1, len(cat_data)):
            growth = cat_data.iloc[i]['MoM_Growth']
            if pd.notnull(growth) and growth > 0.2:
                plt.fill_between(
                    [cat_data.iloc[i - 1]['YearMonth'], cat_data.iloc[i]['YearMonth']],
                    [cat_data.iloc[i - 1]['Installs'], cat_data.iloc[i]['Installs']],
                    alpha=0.2
                )

    plt.title('Monthly Installs Trend (Highlighted: >20% MoM Growth)')
    plt.xlabel('Date')
    plt.ylabel('Total Installs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig15_monthly_installs_growth.png", dpi=300)
    plt.show()

else:
    print(" This graph is only visible between 6 PM and 9 PM IST.")

plot_containers_split=plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot:plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers
from datetime import datetime, time
import pytz

current_time = datetime.now(pytz.timezone("Asia/Kolkata")).time()

plot_containers += fig11.to_html(full_html=False)
    
if time(13, 0) <= current_time <= time(14, 0):
    plot_containers += fig12.to_html(full_html=False)
    
if time(15, 0) <= current_time <= time(15, 0):
    plot_containers += fig13.to_html(full_html=False)

if time(16, 0) <= current_time <= time(18, 0):
    plot_containers += fig14.to_html(full_html=False)

if time(18, 0) <= current_time <= time(21, 0):
    plot_containers += fig15.to_html(full_html=False)
    
dashboard_html="""
<!DOCTYPE html>
<html lang="en">
<head>
     <meta charset="UTF-8">
     <meta name= viewport" content="width=device-width,initial-scale-1.0">
     <title> Google Play Store Review Analytics</title>
     <style>
         body {{
             font-family: Arial,sans_serif;
             background-color: #333;
             color:#fff;
             margin:0;
             padding:0;
         }}
         .header {{ 
             display: flex;
             align-items:center;
             justify-content:center;
             padding:20px;
             backgroun-color:# 444
          }} 
          .header img {{
              margin:0.10px;
              height:50px;
          }}
          .container {{
              display: flex;
              flex-wrap:wrap;
              justify_content: center;
              padding:20px;
          }} 
          .plot-container{{
              border: 2px solid #555
              margin:10px;
              padding:10px;
              width:{plot_width}px;
              height:{plot_height}px;
              overflow: hidden;
              position:relative;
              cursor:pointer;
           }}
           .insights {{
               display: name;
               position: absolute;
               right:10px;
               top:10px;
               background-color: rgba(0,0,0,0.7);
               pedding: 5 px;
               border-radius: 5px;
               color:#fff;
            }}
            .plot-container: hover  .insights  {{
                display: block;
            }}
            </style>
            <script>
                functiom opePlot (filename) {{
                    window.open(filename,  '_blank');
                    }}
            </script>
        </head>  
        <body>
            <div Class "header">
                <img src="https://images.app.goo.gl/ghSZx4VgbhZ2ZsA56">
                <h1>Google Play Store Reviews Analytics </h1>
                <img src="https://images.app.goo.gl/ghSZx4VgbhZ2ZsA56">
            </div>
            <div class="container">
                {plots}
            </div>
        </body>
        </html>
        """ 

final_html = dashboard_html.format(plots=plot_containers, plot_width=800, plot_height=600)

dashboard_path=os.path.join(html_files_path,"web page.html")
with open(dashboard_path, "w" , encoding="utf-8") as f:
    f.write(final_html)
import webbrowser
import os

webbrowser.open ('file://'+os.path.realpath(dashboard_path))