# Task Overview
The goal of this task is to develop a machine learning model to predict whether a customer session on an online platform will result in a purchase (Revenue=True) based on customer behavior metrics. This prediction can help businesses identify potential buyers, optimize marketing strategies, and improve resource allocation.

# Dataset Overview
The dataset contains 18 features derived from customer interaction data, categorized as behavioral metrics, engagement metrics, temporal features, user characteristics and target variable.

# Exploratory Data Analysis
## Target Variable Distribution
- The Revenue variable is highly imbalanced: Not Purchased (False): ~83.3% of the data; Purchased (True): ~16.7% of the data.
- This imbalance requires special handling, such as oversampling or adjusting classification thresholds.
## Visitor Type and Revenue
- Returning visitors are significantly more likely to make a purchase compared to new visitors.
- A count plot of VisitorType with Revenue hue revealed this relationship.
## Bounce Rate and Page Value
- Customers who made a purchase (Revenue=True) have lower Bounce Rate and higher Page Value, indicating that engaging content and valuable pages correlate positively with conversions.
- Density plots show clear separation between Revenue=True and Revenue=False.
## Temporal Patterns
- Conversion rates vary by month, with notable spikes in holiday seasons.
- Sessions occurring on weekends and during special days exhibit higher purchase likelihood.
## PCA Visualization
- Principal Component Analysis (PCA) was used to reduce high-dimensional data into 2D and 3D spaces. In this task, PCA was applied to visualize how the features differentiate customer sessions based on the target variable (Revenue).
- Scatter plots showed some clustering of Revenue=True sessions in specific regions, suggesting separability in feature space.
## Correlation Heatmap
- Correlation Heatmap is a common visualization in data analysis, used to understand the relationships between numerical features and identify potential redundancies or key drivers for the target variable.
- Page Value may show a high positive correlation with Revenue, indicating that it is a key predictor.
- Features with high pairwise correlations (e.g.,Browser,TrafficType, Administrative, Administrative_Duration ) might be redundant.

# Final Results and Model Selection
Final Model: XGBoost
## Why Selected:
While the Random Forest and XGBoost may yield similar metrics (e.g., accuracy, F1-score, AUC-ROC) on the current dataset, XGBoost offers the following additional benefits:
- Better control over class imbalance and regularization.
- Detailed insights into feature importance for interpretation.
- Greater computational efficiency and scalability for large datasets.
- Advanced customization options, making it more flexible in future applications.
Final Decision: Choosing XGBoost as the best model provides not only comparable metrics but also strategic advantages in terms of efficiency, interpretability, and flexibility, making it the preferred choice for deployment.

## Why are overall Metrics kind of Similar Across Models?
### Impact of Class Imbalance
In highly imbalanced datasets, most machine learning models tend to focus on correctly predicting the majority class because it dominates the data distribution. As a result, the overall metrics like accuracy or even AUC-ROC might not vary significantly across models, as these metrics are heavily influenced by the performance on the majority class. In this case, 83% of the samples belong to the "Not Purchased" class, simply predicting "Not Purchased" most of the time can yield high accuracy without truly capturing the minority class behavior.
### Dataset Complexity
If the dataset lacks strong feature separability, even advanced models like XGBoost may not outperform simpler models like Logistic Regression by a wide margin. This can happen if the key features are not sufficiently informative and the patterns in the data are inherently difficult to learn.
### Similar Decision Boundaries
All three models could be producing similar decision boundaries because the underlying patterns in the data are relatively linear, which favors Logistic Regression. Or Non-linear patterns exist but are weak or sparsely distributed, which means Random Forest and XGBoost cannot leverage their strengths in capturing complex relationships effectively.

## Future improvement direction
- Data category imbalance: apply more data augmentation or adjust classification thresholds.
- Model optimization: Use more powerful models (e.g., XGBoost, LightGBM); Perform hyperparameter tuning.
- Feature engineering: try feature interactions, nonlinear transformations, etc.
- Evaluate performance using cross-validation and multiple metrics (F1, recall).
- Check data quality and correlation between features.


