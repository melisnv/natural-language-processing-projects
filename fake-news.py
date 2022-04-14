import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

import spacy

nlp = spacy.load('en_core_web_sm')

true_df = pd.read_csv("./datas/True.csv")
false_df = pd.read_csv("./datas/Fake.csv")

true_df['class'] = 1
false_df['class'] = 0

fake_news_df = pd.concat([true_df,false_df])
fake_news_df.head()

fake_news_df = fake_news_df.sample(10000,random_state=786).reset_index(drop=True)
fake_news_df.head()

labels = ['Fake','Real']
values = fake_news_df['class'].value_counts()/fake_news_df['class'].shape[0]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(
    title_text="Target Balance",
    title_font_color="black",
    legend_title_font_color="yellow",
    paper_bgcolor="white",
    plot_bgcolor='black',
    font_color="black",
)
fig.show()