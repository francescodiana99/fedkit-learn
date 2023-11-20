COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

CATEGORICAL_COLUMNS = [
    'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'
]

TRAIN_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

BACKUP_URL = 'https://archive.ics.uci.edu/static/public/2/adult.zip'

SPLIT_CRITERIA = {
    'doctoral': {'age': (0, 120), 'education': 'Doctorate'},
    'prof-school-junior': {'age': (0, 35), 'education': 'Prof-school'},
    'prof-school-mid-senior': {'age': (36, 50), 'education': 'Prof-school'},
    'prof-school-senior': {'age': (51, 120), 'education': 'Prof-school'},
    'bachelors-junior': {'age': (0, 35), 'education': 'Bachelors'},
    'bachelors-mid-senior': {'age': (36, 50), 'education': 'Bachelors'},
    'bachelors-senior': {'age': (51, 120), 'education': 'Bachelors'},
    'masters': {'age': (0, 120), 'education': 'Masters'},
    'associate': {'age': (0, 120), 'education': 'Associate'},
    'hs-grad': {'age': (0, 120), 'education': 'HS-grad'},
    'compulsory': {'age': (0, 120), 'education': 'Compulsory'}
}

# SPLIT_CRITERIA = {
#     'young_doctoral': {'age': (0, 37), 'education': 'Doctorate'},
#     'mid_senior_doctoral': {'age': (38, 50), 'education': 'Doctorate'},
#     'senior_doctoral': {'age': (51, 120), 'education': 'Doctorate'},
#     'prof-school': {'age': (0, 120), 'education': 'Prof-school'},
#     'masters': {'age': (0, 120), 'education': 'Masters'},
#     'bachelors': {'age': (0, 120), 'education': 'Bachelors'},
#     'associate': {'age': (0, 120), 'education': 'Associate'},
#     'hs-grad': {'age': (0, 120), 'education': 'HS-grad'},
#     'compulsory': {'age': (0, 120), 'education': 'Compulsory'}
# }

