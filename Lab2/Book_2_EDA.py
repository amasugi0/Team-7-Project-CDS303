import numpy as np
import seaborn as sns
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from pca import pca



#PART 1 IMPORTING/CLEANING/TRANSFORMING

#Importing data

twitch_data = pd.read_csv("/Users/tobster/Downloads/twitchdata-update.csv")

#Eliminating non-numeric data & transforming the data to values between 0 and 1

twitch_data['Watch time(Minutes)'].fillna(twitch_data['Watch time(Minutes)'].mean(), inplace=True)
twitch_data['Stream time(minutes)'].fillna(twitch_data['Stream time(minutes)'].mean(), inplace=True)
twitch_data['Peak viewers'].fillna(twitch_data['Peak viewers'].mean(), inplace=True)
twitch_data['Average viewers'].fillna(twitch_data['Average viewers'].mean(), inplace=True)
twitch_data['Followers'].fillna(twitch_data['Followers'].mean(), inplace=True)
twitch_data['Followers gained'].fillna(twitch_data['Followers gained'].mean(), inplace=True)
twitch_data['Views gained'].fillna(twitch_data['Views gained'].mean(), inplace=True)

twitch_data['Partnered'].fillna(twitch_data['Partnered'].value_counts().index[0], inplace=True) #Boolean
twitch_data['Mature'].fillna(twitch_data['Mature'].value_counts().index[0], inplace=True) #Boolean 
twitch_data['Language'].fillna(twitch_data['Language'].value_counts().index[0], inplace=True) #Categorical
twitch_data['Channel'].fillna(twitch_data['Channel'].value_counts().index[0], inplace=True) #Categorical

#Categorical Encoding for string columns

cat_encoder = OneHotEncoder()

#twitch_data["Channel"] = twitch_data["Channel"].astype('category')
#twitch_data.dtypes
#twitch_data["Channel_cat"] = twitch_data["Channel"].cat.codes
#twitch_data.head()

twitch_data["Channel"] = twitch_data["Channel"].astype('category')
twitch_data.dtypes
twitch_data["Channel"] = twitch_data["Channel"].cat.codes

twitch_data["Language"] = twitch_data["Language"].astype('category')
twitch_data.dtypes
twitch_data["Language"] = twitch_data["Language"].cat.codes

twitch_data.head()

twitch_dict = {0:'Watch time(Minutes)',1:'Stream time(minutes)',2:'Peak viewers',3: 'Average viewers', 4:'Followers',5:'Followers gained',6:'Views gained'}

#Normalizing the dataset

scaler = MinMaxScaler()
data_norm = scaler.fit_transform(twitch_data[["Watch time(Minutes)", "Stream time(minutes)", "Peak viewers", "Average viewers", "Followers", "Followers gained", "Views gained", "Partnered", "Mature"]])
norm_names = scaler.get_feature_names_out()
scaled_data = pd.DataFrame(data_norm, columns=norm_names)

twitch_data.describe()

#PART 2 EXPLORATORY DATA ANALYSIS

#Histograms for all numerical predictors
    
watch_time_histogram = scaled_data.hist(column=['Watch time(Minutes)'], bins=30)
plt.ylabel("Frequency")
plt.title("Total Viewer Watch Time in Minutes During 2020 Distribution")

stream_time_histogram = scaled_data.hist(column=['Stream time(minutes)'], bins=30)
plt.ylabel("Frequency")
plt.title("Stream Time in Minutes During 2020 Distribution")


peak_viewers_histogram = scaled_data.hist(column=['Peak viewers'], bins=30)
plt.ylabel("Frequency")
plt.title("Peak Number of Viewers Between 2019-2020 Distribution")


average_viewers_histogram = scaled_data.hist(column=['Average viewers'], bins=30)
plt.ylabel("Frequency")
plt.title("Average Number of Viewers Between 2019-2020 Distribution")


followers_histogram = scaled_data.hist(column=['Followers'], bins=30)
plt.ylabel("Frequency")
plt.title(" Total Followers in 2020 Distribution")


followers_gained_histogram = scaled_data.hist(column=['Followers gained'], bins=30)
plt.ylabel("Frequency")
plt.title("Followers Gained Between 2019-2020 Distribution")


views_gained_histogram = scaled_data.hist(column=['Views gained'], bins=30)
plt.ylabel("Frequency")
plt.title("View Count Gained Between 2019-2020 Distribution")
plt.show()

#Scattergrams

for index in range(0,len(twitch_dict)-1):
    for index1 in range(index,len(twitch_dict)-1):
        plt.scatter(scaled_data[twitch_dict[index]],scaled_data[twitch_dict[index1+1]],s=4)
        plt.xlabel(twitch_dict[index])
        plt.ylabel(twitch_dict[index1+1])
        plt.show()


#Box plots for all numerical predictors

box_plot = plt.boxplot(scaled_data[list(scaled_data.columns)[0:7]])
plt.xticks([1,2,3,4,5,6,7], ['TWT', 'ST', 'PV','AV','F','FG','VG'])
plt.show()

#Correlation matrix

sn.heatmap(scaled_data[list(scaled_data.columns)[0:7]].corr())
plt.show()

#PART 3 DATA REDUCTION

model = pca(normalize=True, n_components=None)
results = model.fit_transform(scaled_data)
results_matrix = pd.DataFrame(model.results['loadings'])
print(results_matrix)
model.results
model.results['topfeat']

twitch_PC = results["PC"]

plt.scatter(twitch_PC["PC1"],twitch_PC["PC2"])

scaled_data
