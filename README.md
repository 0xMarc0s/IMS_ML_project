# General backlog
- Generally similar to configuration 12, except I'm adding a `life` indicator which sums up every encoded categorical variable which would represent a characteristic of one's lifestyle. Makes sense as they are ordinal. Life sums every encoded variable in a positive way, as modifying it with penalties penalizes the performance.

# Notes for Feature Selection:
- According to RFE the following are selected:
    - `{'i': 7, 'selected': ['age', 'gender', 'height', 'meals_perday', 'weight', 'bmi_class', 'life']}`

- According to lasso method, the only following (numerical) is selected:
    - `weight` (the rest has $0$)

- The following variables have a feature importance at least higher than $3\%$:
    - weight	0.246493
    - bmi_class	0.228433
    - age	0.090671
    - height	0.082738
    - gender	0.061708
    - life	0.033230
    - meals_perday	0.031686
    - eat_between_meals	0.030832

- Spearman rank correlation to determine independence:
    - alcohol_freq is IMPORTANT for prediction (Spearman's correlation = 0.13, p = 0.0000)
    - caloric_freq is NOT an important predictor (Spearman's correlation = 0.05, p = 0.0674)
    - devices_perday is NOT an important predictor (Spearman's correlation = -0.04, p = 0.1463)
    - eat_between_meals is IMPORTANT for prediction (Spearman's correlation = -0.36, p = 0.0000)
    - gender is NOT an important predictor (Spearman's correlation = -0.01, p = 0.6089)
    - monitor_calories is IMPORTANT for prediction (Spearman's correlation = -0.08, p = 0.0015)
    - parent_overweight is IMPORTANT for prediction (Spearman's correlation = 0.31, p = 0.0000)
    - physical_activity_perweek is IMPORTANT for prediction (Spearman's correlation = -0.15, p = 0.0000)
    - smoke is NOT an important predictor (Spearman's correlation = -0.03, p = 0.2438)
    - transportation is IMPORTANT for prediction (Spearman's correlation = 0.05, p = 0.0491)
    - veggies_freq is NOT an important predictor (Spearman's correlation = 0.02, p = 0.5233)
    - water_daily is IMPORTANT for prediction (Spearman's correlation = 0.09, p = 0.0005)
    - bmi_class is IMPORTANT for prediction (Spearman's correlation = 0.32, p = 0.0000)
    - meals_perday is IMPORTANT for prediction (Spearman's correlation = -0.10, p = 0.0000)
    - siblings is NOT an important predictor (Spearman's correlation = 0.02, p = 0.5359)
