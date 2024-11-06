# General backlog
- Generally similar to configuration 12, except I'm adding a `life` indicator which sums up every encoded categorical variable which would represent a characteristic of one's lifestyle. Makes sense as they are ordinal. Life sums every encoded variable in a positive way, as modifying it with penalties penalizes the performance.

# Notes for Feature Selection:
- According to Pearson Correlation there are no correlated variables between numerical variables
- According to Spearman correlation there is a strong correlation ($\sim 0.8$) between `weight` and `bmi_class`
- According to RFE the following are selected:
    - `{'i': 5, 'selected': ['age', 'gender', 'height', 'weight', 'bmi_class']}`

- The following variables have a feature importance at least higher than $3\%$:
    - weight	0.236120
    - bmi_class	0.235415
    - age	0.086846
    - height	0.083978
    - gender	0.066053
    - life	0.038125
    - alcohol_freq	0.029160
    - meals_perday	0.029102
    - eat_between_meals	0.028861
    - parent_overweight	0.028335
    - veggies_freq	0.027013
    - transportation	0.021881
    - devices_perday	0.020024
    - caloric_freq	0.016985
    - water_daily	0.016073
    - physical_activity_perweek	0.014653
    - siblings	0.014622
    - monitor_calories	0.004747
    - smoke	0.002006

- Spearman rank correlation to determine independence:
    - alcohol_freq is IMPORTANT for prediction (Spearman's correlation = 0.13, p = 0.0000)
    - caloric_freq is IMPORTANT for prediction (Spearman's correlation = 0.05, p = 0.0432)
    - devices_perday is NOT an important predictor (Spearman's correlation = -0.04, p = 0.1098)
    - eat_between_meals is IMPORTANT for prediction (Spearman's correlation = -0.37, p = 0.0000)
    - gender is NOT an important predictor (Spearman's correlation = -0.01, p = 0.6089)
    - monitor_calories is IMPORTANT for prediction (Spearman's correlation = -0.08, p = 0.0011)
    - parent_overweight is IMPORTANT for prediction (Spearman's correlation = 0.31, p = 0.0000)
    - physical_activity_perweek is IMPORTANT for prediction (Spearman's correlation = -0.16, p = 0.0000)
    - smoke is NOT an important predictor (Spearman's correlation = -0.03, p = 0.1801)
    - transportation is NOT an important predictor (Spearman's correlation = 0.05, p = 0.0518)
    - veggies_freq is NOT an important predictor (Spearman's correlation = 0.02, p = 0.4897)
    - water_daily is IMPORTANT for prediction (Spearman's correlation = 0.09, p = 0.0005)
    - bmi_class is IMPORTANT for prediction (Spearman's correlation = 0.32, p = 0.0000)
    - meals_perday is IMPORTANT for prediction (Spearman's correlation = -0.10, p = 0.0001)
    - siblings is NOT an important predictor (Spearman's correlation = 0.02, p = 0.5166)
