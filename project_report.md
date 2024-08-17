# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### MennatAllah Mohamed Hassan

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When attempting to submit predictions to Kaggle, I encountered issues with negative prediction values from the models. This was observed across 3 different experiments:

* 1- Initial Raw Submission [Model: initial]:
The model trained on the raw dataset without performing any data analysis or feature engineering produced predictions with a significant number of errors.
* 2- Added Features Submission (EDA + Feature Engineering) [Model: add_features]:
After conducting exploratory data analysis (EDA) and feature engineering, the model's performance improved. However, it still produced some negative prediction values due to errors in the dataset or model inaccuracies.
* 3 - Hyperparameter Optimization (HPO) [Model: hpo]: 
Hyperparameter optimization (HPO) was performed to fine-tune the models' hyperparameters for better performance. Despite optimization efforts, some models continued to generate negative predictions.

To address the issue of negative prediction values and ensure compatibility with Kaggle's submission requirements, the following changes were implemented:
* Replacement of Negative Values: All negative prediction values obtained from the predictors were replaced with 0. This modification ensured that the predictions were within the acceptable range for submission to Kaggle.

By incorporating these changes, I successfully prepared the predictions from each experiment for submission to Kaggle, thereby overcoming the issue of negative values and ensuring compliance with Kaggle's submission guidelines

### What was the top ranked model that performed?
The top-ranked model that performed was the WeightedEnsemble_L3, optimized through hyperparameter tuning (HPO). It achieved a validation RMSE score of 37.367292 and the best Kaggle score of 0.45908, demonstrating superior performance in predicting bike sharing counts.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
1- Exploratory Analysis Insights:
* Temperature and "atemp" are highly correlated, indicating similarity. Only one variable should be included in the model to avoid multicollinearity.
* Higher weather ratings correspond to reduced bike rentals, indicating worsened weather conditions.
* Humidity has an inverse correlation with bike rental count, implying fewer rentals during humid weather.
* Holidays are associated with fewer bike rentals.
* "Casual" and "registered" variables are not present in the test dataset and should be excluded from modeling.
2- Feature Engineering Steps:
* Extracted year, month, day, and hour into separate columns from the datetime feature in both the train and test datasets.
* Changed the data type of "season" and "weather" columns to category.
* One-hot encoded the "weather" and "season" categories to create individual features for each category, prefixing with "weather_" and "season_" respectively.
* Dropped the original "season" and "weather" columns from the dataset.
* Dropped the  "atemp" column from the dataset.
3- Visualization Insights:
* Heatmaps provided insights into the relationships between weather, temperature, and bike rental counts.

These steps helped in preparing the data for modeling by extracting relevant temporal information and encoding categorical variables into a format suitable for model.

### How much better did your model preform after adding additional features and why do you think that is?
By separating the date into distinct features such as year, month, day, and hour, we provide the model with more granular information about temporal patterns. This allows the model to better capture seasonality trends and time-of-day effects, which are crucial factors influencing bike rental counts. As a result,he model's performance improved significantly. The initial Kaggle score was 1.80372, but it decreased to 0.48139 after incorporating these features.


## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?

* After hyperparameter optimization and feature addition, the model's performance significantly improved.
* The initial Kaggle score was [1.80372], but it decreased to [0.45908 ] after hyperparameter optimization.
* Hyperparameters were tailored for each model type, including GBM, NN_TORCH, CAT, KNN, XT, XGB, and RF, to guide the optimization process effectively. here are the hyperparameters: 
1- 'GBM': {'num_boost_round': 100, 'num_leaves': Int: lower=26, upper=66},
2- 'NN_TORCH': {'num_epochs': 10, 'learning_rate': Real: lower=0.0001, upper=0.01, 'activation': Categorical['relu', 'softrelu', 'tanh'], 'dropout_prob': Real: lower=0.0, upper=0.5},
3- 'CAT': {'depth': Int: lower=4, upper=7},
4- 'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Uniform'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Distance'}}],
5- 'XT': {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
6- 'XGB': {},
7- 'RF': {'criterion': 'squared_error','ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}}


### If you were given more time with this dataset, where do you think you would spend more time?
Given more time with this dataset, I would prioritize:
###### 1- Feature Engineering: Explore additional feature transformations and combinations such as.
* Weather Encoding: Created dummy variables to capture different weather conditions.
* Seasonal Features: Added binary indicators for seasons and months.
* Time-related Features: Generated features like day of the week and hour of the day.
* Interaction Terms: Created combined effects between correlated features.
###### 2- Model Selection and Evaluation: Experiment with a wider range of models and conduct more extensive evaluation.
###### 3- Hyperparameter Tuning: Further refine model hyperparameters for optimization.
###### 4-Time-Series Analysis: Delve deeper into temporal patterns and seasonality.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

|model|NN|GBM|RF|score|
|--|--|--|--|--|
|initial|default|default|default|1.80372 |
|add_features|default|default|default|0.48139|
|hpo|NN Tuning|RF Tuning|GBM Tuning| 0.45908 |

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
* The project aimed to predict bike sharing demand, starting with an initial model trained on raw data. Subsequent iterations involved exploratory data analysis (EDA), feature engineering, and hyperparameter optimization (HPO) to improve predictive accuracy.
* EDA revealed insights into weather, temperature, humidity, and seasonal trends, guiding the creation of additional features. These features captured temporal patterns and interactions between variables.
* HPO fine-tuned model hyperparameters, resulting in a significant performance boost. The final model achieved a Kaggle score of 0.45908, demonstrating the effectiveness of iterative development and optimization.
* Given more time, further emphasis would be placed on feature engineering, model selection, and time-series analysis to enhance predictive capabilities.
