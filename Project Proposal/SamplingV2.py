import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load the data from an excel file

stroke_dataframe = pd.read_csv("/Users/tobster/Downloads/Stroke_Dataset.csv")

# Seperate the target variable from the main dataframe

targets = stroke_dataframe["Stroke"]
del stroke_dataframe["Stroke"]
del stroke_dataframe["id"]

# Since the target classes are unbalanced, we must randomly sample from the non stroke instances till we have a 1:1
# ratio


under = RandomUnderSampler(sampling_strategy=1)
X, y = under.fit_resample(stroke_dataframe, targets)

# Split the data into training and testing sets

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.25)

# Create a decision tree classifier

base_model = tree.DecisionTreeClassifier(random_state=0)

# Fine tune the hyper parameters of the model

search_space = {"criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 4, 5, 6],
                "min_samples_split": [5, 10, 15, 20],
                "min_samples_leaf": [2, 3, 4, 5, 10, 15, 20, 25]}


GS = GridSearchCV(estimator=base_model, param_grid=search_space, refit=True)
GS.fit(X_train, Y_train)

# Figure out what the optimal hyper parameters are

print(type(GS.best_params_))
fitted_model = GS.best_estimator_
print(type(fitted_model))

print(fitted_model.score(x_test, y_test))


# Create visualization of decision tree using graphviz

plt.figure(figsize=(16, 10))
dot_data = tree.export_graphviz(fitted_model, feature_names=list(stroke_dataframe.columns),
                                class_names=["No Stroke", "Stroke"], filled=True, rounded=True,
                                special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("Stroke_Tree_Classifier")