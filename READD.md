# 🐈 🧮 Badge Preference and Willingness to Participate in Tutorial Experiences
### Overview
Below are instructions on how to run a linear regression analysis and paired t-test on a dataset from a badge preference study. The analysis aims to investigate whether participants' willingness to talk about their conference tutorial experience is influenced by the type of badge they receive (stats vs. cat) and their desire to have and wear the badge. 

### Prerequisites
Before running the code, ensure you have the following Python packages installed:
1. **pandas** (for data manipulation)
2. **statsmodels** (for running the regression analysis)
3. **scipy** (for running the paired t-test analysis)

You can install these packages using the following commands:
```bash
pip install pandas
pip install statsmodels
pip install scipy
```

### Data Description
The `preference_data.csv` file contains the following columns:
- `participant_id`: Unique identifier for each participant.
- `badge`: Badge type (either "cat" or "stats").
- `qid`: Question ID (q1 for "I want to have this button," q2 for "I want to wear this button during the conference," and q3 for "Having this button will make me want to talk about my experience in this tutorial").
- `response`: Response to the Likert scale question (1-5).


## [1] Linear regression analysis
### Overview
- **Dependent Variable:** The response to the question about willingness to talk about the tutorial experience (Likert scale 1-5).
- **Independent Variables:**
  - Badge Type (`badge_dummy`): 1 for stats badge, 0 for cat badge.
  - Desire to Have the Badge (`have_button_response`): Response to the question about wanting to have the badge.
  - Desire to Wear the Badge (`wear_button_response`): Response to the question about wanting to wear the badge during the conference.

The provided Python code performs the following steps:
1. Load the dataset: The dataset is loaded using `pandas.read_csv()` from the `preference_data.csv` file.
2. Filter the data: The data is filtered to focus on responses related to the third question (willingness to talk about the tutorial experience).
3. Create a dummy variable: The badge type is encoded as a dummy variable where 1 represents the stats badge, and 0 represents the cat badge.
4. Add covariates: Responses to the first two questions (desire to have and wear the badge) are added as covariates to the filtered dataset.

### How to run the code
1. Save the code provided below to a Python file, e.g., `badge_analysis.py`.
2. Ensure that `preference_data.csv` is in the same directory as your Python file or update the file path accordingly.
3. Run the code using your Python interpreter.

### Expected output
The code will output a summary of the linear regression results, including:
- Coefficients: Show how each independent variable (badge type, desire to have, desire to wear) influences the dependent variable (willingness to talk about the tutorial experience).
- p-values: Indicate whether the independent variables have statistically significant effects on the dependent variable.
- R-squared value: Provides information on how much variance in the dependent variable is explained by the model.


## [2] Paired t-test analysis
### Overview
- **Dependent Variable:** The response to the question about willingness to talk about the tutorial experience (Likert scale 1-5).
- **Hypotheses:**
  - Null hypothesis (H₀): "There is no significant difference between participants' likelihood to talk about their conference experience for the stats badge versus the cat badge."
  - Alternative hypothesis (H₁): "Participants are more likely to talk about their conference experience when receiving the stats badge compared to the cat badge."
 
The provided Python code performs the following steps:
1. Load the dataset: The dataset is loaded using `pandas.read_csv()` from the `preference_data.csv` file.
2. Filter the data: The data is filtered to focus on responses related to the third question (willingness to talk about the tutorial experience).
3. Reshape the data: The data is pivoted to have participants as the index, with separate columns for responses to the cat and stats badges.
4. Perform the paired t-test: The ttest_rel function from `scipy.stats` is used to perform the paired t-test between the responses for the cat badge and the stats badge.


### Expected output
- t-statistic: A measure of the size of the difference between the two badge conditions (cat and stats). Larger values indicate a greater difference.
- p-value: Indicates whether the observed difference is statistically significant.


## Contact
For any issues running the code or questions about the analysis, feel free to reach out to dayeonki@umd.edu!
