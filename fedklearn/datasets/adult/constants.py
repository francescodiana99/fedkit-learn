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

SPLIT_CRITERIA = {'age_education':
    {
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
    },
    'age': {
        '17_22':{'age': (17, 22)},
        '23_28':{'age': (23, 28)},
        '29_34':{'age': (29, 34)},
        '35_40':{'age': (35, 40)},
        '41_46':{'age': (41, 46)},
        '47_52':{'age': (47, 52)},
        '53_58':{'age': (53, 58)},
        '59_64':{'age': (59, 64)},
        '65_70':{'age': (65, 70)},
        '71_76':{'age': (71, 76)},
        '77_82':{'age': (77, 82)},
        '83_120':{'age': (83, 120)}
    }
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

