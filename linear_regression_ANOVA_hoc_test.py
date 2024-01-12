# Hypothesis testing with Python

# IMPORT libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv('google_data_analitics\\marketing_sales_data_2.csv')
# Display the first five rows
print(data.head(5))

# DATA EXPLORATION
# Create a boxplot with TV and Sales
plt.figure(figsize=(7, 5))
sns.boxplot(x=data['TV'], y=data['Sales'], order=['Low', 'Medium', 'High'], 
            medianprops={"color": 'red', 'linewidth': 2}, palette=sns.color_palette("Set2"))
plt.title('Sales VS the TV Promotion Budget')
plt.show()
# There is considerable variation in Sales across the TV groups. 
# The significance of these differences can be tested with a one-way ANOVA.

# Create a boxplot with Influencer and Sales
plt.figure(figsize=(7, 5))
sns.boxplot(x=data['Influencer'], y=data['Sales'], order=['Nano', 'Micro', 'Macro', 'Mega'],
           medianprops={"color": 'red', 'linewidth': 2}, palette=sns.color_palette("Set2"))
plt.title('Sales VS the Influencer Size')
plt.show()
# There is some variation in Sales across the Influencer groups, but it may not be significant.

# REMOVE MISSING DATA
# Drop rows that contain missing data and update the DataFrame.
print(f'Our dataset has {data.shape[0]} rows and {data.shape[1]} columns before cleaning.')
print(data.isna().sum())
data = data.dropna(axis=0)
# Confirm the data contains no missing values.
print(f'After cleaning our dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
print(data.isna().sum())

# MODEL BUILDING
#  To take a subset from our dataset
ols_data = data[['TV', 'Sales']]
# Define the OLS formula
ols_formula = 'Sales ~ C(TV)'
# Create an OLS model
OLS = ols(formula=ols_formula, data=ols_data)
# Fit the model
model = OLS.fit()
# Save the results summary
model_results = model.summary()
# Display the model results
print(model_results)
# TV was selected as the preceding analysis showed a strong relationship 
# between the TV promotion budget and the average Sales. Influencer was not selected 
# because it did not show a strong relationship to Sales in the analysis.

# CHECK MODEL ASSUMPTIONS
# 1. The linearity assumption
# Because our model does not have any continuous independent variables,
# the linearity assumption is not required.

# 2. The independent observation assumption 
# This assumption states that each observation in the dataset is independent. 
# As each marketing promotion (row) is independent from one another, the independence assumption is not violated.

# 3. The normality assumption (the normal distribution for residuals)
# Calculate the residuals.
residuals = model.resid

fig, axes = plt.subplots(1,2, figsize=(10, 5))
# Create a histogram with the residuals
sns.histplot(residuals, ax=axes[0], color='orange', label='Residuals')
# Set the x label of the residual plot
axes[0].set_xlabel('Residual Value')
# Set the title of the residual plot
axes[0].set_title('Histogram of Residuals')
axes[0].legend()

# Create a Q-Q plot of the residuals
sm.qqplot(residuals, line='s', ax=axes[1], label='Residuals')
# Set the title of the Q-Q plot
axes[1].set_title('Normal Q-Q plot')
# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance
axes[1].legend()
plt.tight_layout()
plt.show()
# There is reasonable concern that the normality assumption is not met 
# when TV is used as the independent variable predicting Sales. 
# The normal q-q forms an 'S' that deviates off the red diagonal line, which is not desired behavior.
# However, for the purpose of the lab, continue assuming the normality assumption is met.

# 4. The constant variance (homoscedasticity) assumption
fitted_values = model.fittedvalues # get fitted values
# Create a scatterplot of residuals against fitted values
fig = sns.scatterplot(x=fitted_values, y=residuals, color='orange', label='Fitted values')
# Set the x-axis label
fig.set_xlabel('Fitted values from the model')
# Set the y-axis label
fig.set_ylabel('Residuals')
# Set the title
fig.set_title('Fitted values vs Residuals')
fig.legend()
# Add a line at y = 0 to visualize the variance of residuals above and below 0
fig.axhline(y=0, color='red')
plt.show()
# The variance where there are fitted values is similarly distributed, 
# validating that the constant variance assumption is met.

# RESULTS AND EVALUATION
# Display the model results summary
print(model_results)

# Using TV as the independent variable results in a linear regression model with  𝑅2=0.874. 
# In other words, the model explains  87.4% of the variation in Sales. 
# This makes the model an effective predictor of Sales.

# The default TV category for the model is High, because there are coefficients for the other two 
# TV categories, Medium and Low. According to the model, Sales with a Medium or Low TV category 
# are lower on average than Sales with a High TV category. For example, the model predicts that 
# a Low TV promotion would be 208.813 (in millions of dollars) lower in Sales on average than a High TV promotion.

# The p-value for all coefficients is  0.000, meaning all coefficients are statistically significant at  𝑝=0.05.
# The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders.
#  For instance, there is a  95% chance the interval  [−215.353,−202.274] contains the true parameter 
# of the slope of 𝛽𝑇𝑉𝐿𝑜𝑤, which is the estimated difference in promotion sales when a Low TV promotion 
# is chosen instead of a High TV promotion.

# Given how accurate TV was as a predictor, the model could be improved with a more granular view 
# of the TV promotions, such as additional categories or the actual TV promotion budgets. 
# Further, additional variables, such as the location of the marketing campaign 
# or the time of year, may increase model accuracy.

# PERFORM  a one-way ANOVA test
# The one-way ANOVA test can help us to determine whether there is 
# a statistically significant difference in Sales among groups (TV's budget groups).
# Create an one-way ANOVA table for the fit model
one_way_ANOVA = sm.stats.anova_lm(model, type=2)
print(one_way_ANOVA)

# H0: There is NO difference in Sales based on the TV promotion budget.
# Ha: There is A difference in Sales based on the TV promotion budget.

# The F-test statistic is 1971.46 and the p-value is  8.81∗10−256(i.e., very small). 
# Because the p-value is less than 0.05, we would reject the null hypothesis (H0)
# that there is no difference in Sales based on the TV promotion budget.

# The results of the one-way ANOVA test indicate that we can reject 
# the null hypothesis (H0) in favor of the alternative hypothesis. 
# There is a statistically significant difference in Sales among TV groups.

# PERFORM an ANOVA post hoc test
# We use a post hoc test (the Tukey's HSD post hoc test) to compare 
# if there is a significant difference between EACH pair of categories for TV.
# Perform the Tukey's HSD post hoc test
tukey_oneway = pairwise_tukeyhsd(endog=ols_data['Sales'], groups=ols_data['TV'], alpha=0.05)
print(tukey_oneway.summary())
# The first row, which compares the High and Low TV groups, 
# indicates that we can reject the null hypothesis that 
# there is no significant difference between the Sales of these two groups.
# We can also reject the null hypotheses for the two other 
# pairwise comparisons that compare High to Medium and Low to Medium.

# The post hoc test was conducted to determine which TV groups are different 
# and how many are different from each other. This provides more detail 
# than the one-way ANOVA results, which can at most determine 
# that at least one group is different. Further, using the Tukey HSD controls 
# for the increasing probability of incorrectly rejecting a null hypothesis from peforming multiple tests.

# CONSIDERATIONS
# High TV promotion budgets result in significantly more sales than both medium and low TV promotion budgets. 
# Medium TV promotion budgets result in significantly more sales than low TV promotion budgets.

# Specifically, following are estimates for the difference between the mean sales 
# resulting from different pairs of TV promotions, as determined by the Tukey's HSD test:
# - Estimated difference between the mean sales resulting from High and 
# Low TV promotions: $208.81 million (with 95% confidence that the exact value 
# for this difference is between 200.99 and 216.64 million dollars).
# - Estimated difference between the mean sales resulting from High and 
# Medium TV promotions: $101.51 million (with 95% confidence that the exact value 
# for this difference is between 93.69 and 109.32 million dollars).
# - Difference between the mean sales resulting from Medium and Low TV promotions: $107.31 million 
# (with 95% confidence that the exact value for this difference is between 99.71 and 114.91 million dollars).

# The linear regression model estimating Sales from TV had an R-squared of 0.874, 
# making it a fairly accurate estimator. The model showed 
# a statistically significant relationship between the TV promotion budget and Sales.

# The results of the one-way ANOVA test indicate that the null hypothesis 
# that there is no difference in Sales based on the TV promotion budget
#  can be rejected. Through the ANOVA post hoc test, 
# a significant difference between all pairs of TV promotions was found.

# The difference in the distribution of sales across TV promotions was determined significant 
# by both a one-way ANOVA test and a Tukey’s HSD test.
