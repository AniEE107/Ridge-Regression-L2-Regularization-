# Ridge-Regression-L2-Regularization-
# Overview & Purpose
This Jupyter Notebook provides a comprehensive demonstration of Ridge Regression, also known as L2 Regularization. In linear regression, especially when dealing with multicollinearity or a large number of features, models can become overly complex and prone to overfitting. Ridge Regression addresses this by adding a penalty term (L2 norm of the coefficients) to the ordinary least squares cost function. This penalty shrinks the coefficients towards zero (but not exactly to zero), which helps in reducing model complexity, handling multicollinearity, and improving the model's generalization ability.

The primary purpose of this notebook is to:

Illustrate Ridge Regression: Provide a clear, practical example of how Ridge Regression works.

Demonstrate Regularization Impact: Show how different regularization strengths (alpha values) affect model coefficients and the regression line, leading to a smoother fit.

Address Overfitting: Highlight how L2 regularization helps in building more robust and interpretable models.

Apply to Real and Synthetic Data: Showcase Ridge Regression's application on both a real-world dataset (Diabetes) and a synthetic dataset to provide diverse examples.

# Key Concepts & Functionality
The notebook covers the following aspects of L2 Regularization:
Initial Setup: Imports necessary libraries including numpy, pandas, matplotlib.pyplot, and various modules from sklearn for linear models, metrics, and preprocessing.

Diabetes Dataset Analysis:
Loads the load_diabetes dataset from scikit-learn, which is a common benchmark for regression tasks.
Prints the dataset description (data.DESCR) to understand its features and target.
Splits the data into training and testing sets using train_test_split.
Trains a standard LinearRegression model as a baseline.
Evaluates the baseline model's performance using R2 score and Root Mean Squared Error (RMSE).
Synthetic Data Generation and Visualization:
Generates a synthetic 1-dimensional regression dataset with added noise. This controlled environment is excellent for visually demonstrating the effect of regularization on the regression curve.
A scatter plot of the synthetic data points is generated.

Polynomial Features:
PolynomialFeatures (with degree=16) is used to create higher-order polynomial terms from the single feature. This intentionally introduces high dimensionality and potential for overfitting, setting the stage for Ridge Regression to demonstrate its benefits.

Ridge Regression Implementation:
The Ridge model from sklearn.linear_model is used.
A Pipeline is constructed to combine PolynomialFeatures and Ridge, allowing for a streamlined application of the transformation and the model.
A function get_preds_ridge is defined to train the pipeline with a given alpha and return predictions.
The notebook iterates through different alpha values (e.g., 0, 20, 200) to show the effect of varying regularization strength.
Visualization of Ridge's Effect (on Synthetic Data):
Multiple regression lines are plotted on the same graph, each corresponding to a different alpha value.
This visualization clearly shows how increasing alpha leads to a smoother, less complex regression line, effectively reducing variance and preventing overfitting to the noise in the training data.
The plot includes a legend, title, and axis labels for clarity.

# Technologies Used
Python
NumPy (for numerical operations)
Pandas (for data manipulation, though less prominent in the provided snippet)
Matplotlib (for plotting and visualization)
Scikit-learn (LinearRegression, Ridge, r2_score, mean_squared_error, train_test_split, load_diabetes, Pipeline, PolynomialFeatures)
Jupyter Notebook
