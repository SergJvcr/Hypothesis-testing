import numpy as np
import pandas as pd
from scipy import stats

aqi = pd.read_csv('google_data_analitics\\c4_epa_air_quality.csv')

# Data Exploration
print(aqi.describe(include='all'))
print(aqi.head(10))
print(f'The shape ot the dataset aqi: {aqi.shape}')
print(aqi['state_name'].value_counts())

# Statistical Tests
# Steps for conducting hypothesis testing:
# -Formulate the null hypothesis and the alternative hypothesis.
# -Set the significance level.
# -Determine the appropriate test procedure.
# -Compute the p-value.
# -Draw your conclusion.

# Hypothesis 1: ROA is considering a metropolitan-focused approach. 
# Within California, they want to know if the mean AQI in Los Angeles County 
# is statistically different from the rest of California.

# Create dataframes for each sample being compared in your test

aqi_la_cali = aqi[aqi['county_name'] == 'Los Angeles']
aqi_rest_cali = aqi[(aqi['state_name']=='California') & (aqi['county_name']!='Los Angeles')]

mean_aqi_la = aqi_la_cali['aqi'].mean()
mean_aqi_rest_cali = aqi_rest_cali['aqi'].mean()

print(f'The mean of AQI in LA = {mean_aqi_la}')
print(f'The mean of AQI from the rest of California = {mean_aqi_rest_cali}')
print(f'The absolute difference between AQI in LA and AQI from the rest of California is {mean_aqi_la - mean_aqi_rest_cali}')

# 𝐻0: There is no difference in the mean AQI between Los Angeles County and the rest of California.
# 𝐻a: There is a difference in the mean AQI between Los Angeles County and the rest of California.

# For this analysis, the significance level is 5%
significance_level = 0.05
print(f'The significance level = {significance_level} or {significance_level*100}%')

# Compute the p-value here
statistic_t_score, p_value = stats.ttest_ind(a=aqi_la_cali['aqi'], b=aqi_rest_cali['aqi'], equal_var=False)

print(f'Statistic (t-score) for this two-sample t-test is {round(statistic_t_score, 4)}')
print(f'The p-value for this two-sample t-test is {round(p_value, 4)} or {round(p_value*100, 2)}%')
print('''With a p-value (0.049) being less than 0.05 as your significance level is 5%),
       reject the null hypothesis in favor of the alternative hypothesis.''')

# Hypothesis 2: With limited resources, ROA has to choose between New York
# and Ohio for their next regional office. Does New York have a lower AQI than Ohio?

# Create dataframes for each sample being compared in your test
aqi_ny = aqi[aqi['state_name'] == 'New York']
aqi_ohio = aqi[aqi['state_name'] == 'Ohio']

mean_aqi_ny = aqi_ny['aqi'].mean()
mean_aqi_ohio = aqi_ohio['aqi'].mean()

print(f'The mean of AQI in NY = {mean_aqi_ny}')
print(f'The mean of AQI in Ohio = {mean_aqi_ohio}')
print(f'The absolute difference between AQI in NY and AQI in Ohio is {mean_aqi_ohio - mean_aqi_ny}')

# 𝐻0: The mean AQI of New York is greater than or equal to that of Ohio.
# 𝐻a: The mean AQI of New York is below that of Ohio.
# For this analysis, the significance level is 5% - we already create this in the 41 row

# Compute the p-value here
statistic_t_score_2, p_value_2 = stats.ttest_ind(a=aqi_ny['aqi'], b=aqi_ohio['aqi'], alternative='less', equal_var=False)

# alternative{‘two-sided’, ‘less’, ‘greater’}, optional
# Defines the alternative hypothesis. The following options are available (default is ‘two-sided’):
# ‘two-sided’: the means of the distributions underlying the samples are unequal.
# ‘less’: the mean of the distribution underlying the a sample is less than 
# the mean of the distribution underlying the b sample.
# ‘greater’: the mean of the distribution underlying the a sample is greater than 
# the mean of the distribution underlying the second sample.

print(f'Statistic (t-score) for this two-sample t-test is {round(statistic_t_score_2, 4)}')
print(f'The p-value for this two-sample t-test is {round(p_value_2, 4)} or {round(p_value_2*100, 2)}%')
print('''With a p-value (0.030) of less than 0.05 (as your significance level is 5%) and a t-statistic < 0 (-2.036), 
      reject the null hypothesis in favor of the alternative hypothesis.
      Therefore, we can conclude at the 0.05 significance level that New York has a lower mean AQI than Ohio.''')

# Hypothesis 3: A new policy will affect those states with a mean AQI of 10 or greater.
# Can you rule out Michigan from being affected by this new policy?

# Create dataframes for each sample being compared in your test
aqi_michigan = aqi[aqi['state_name'] == 'Michigan']

mean_aqi_michigan = aqi_michigan['aqi'].mean()
print(f'The mean value of AQI in Michigan is {round(mean_aqi_michigan, 2)}')

# 𝐻0: The mean AQI of Michigan is less than or equal to 10.
# 𝐻𝐴: The mean AQI of Michigan is greater than 10.
# For this analysis, the significance level is 5% - we already create this in the 41 row

# Compute the p-value here
statistic_t_score_3, p_value_3 = stats.ttest_1samp(aqi_michigan['aqi'], 10, alternative='greater')
# b=10 - this is because we utilize a one-sample 𝑡-test with cheking the H0 and Ha

print(f'Statistic (t-score) for this two-sample t-test is {round(statistic_t_score_3, 4)}')
print(f'The p-value for this two-sample t-test is {round(p_value_3, 4)} or {round(p_value_3*100, 2)}%')
print('''With a p-value (0.940) being greater than 0.05 (as your significance level is 5%)
      and a t-statistic < 0 (-1.74), fail to reject the null hypothesis. 
      Therefore, we cannot conclude at the 0.05 significance level that Michigan's mean AQI
       is greater than 10. This implies that Michigan would not be affected by the new policy.''')

# Hypothesis 1: the results indicated that the AQI in Los Angeles County was in fact different from the rest of California.
# Hypothesis 2: Using a 5% significance level, we can conclude that New York has a lower AQI than Ohio based on the results.
# Hypothesis 3: Based on the tests, you would fail to reject the null hypothesis, meaning we can't conclude that 
# the mean AQI is greater than 10. Thus, it is unlikely that Michigan would be affected by the new policy.