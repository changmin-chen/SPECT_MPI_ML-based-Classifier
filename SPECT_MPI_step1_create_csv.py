import pandas as pd
from os.path import join

# readout xls data
fname = join('..', 'GroundTruth.xls')
data = pd.read_excel(fname)
data = data.iloc[0:192]
data = data[['# Patients','Evaluation','Age','Sex','Set Distribution']]

# extract colums, assign processed data
data['# Patients'] = data['# Patients'].apply(str) + '.jpg' # filename = paitentID.jpg
data['Evaluation'] = data['Evaluation'].transform(lambda x: 0 if x == 'Normal' else 1) # totally 42 Normal & 150 Abnormal

# sorting data depend on the set type
data['Set Distribution'] = data['Set Distribution'].replace('Validation', 'Train')
trainSet = data.loc[data['Set Distribution'] == 'Train']
testSet = data.loc[data['Set Distribution'] == 'Test']

# check processed data
print(trainSet.head())
print(testSet.head())

# write .csv file
trainSet.to_csv(join('..', 'trainSet.csv'), header=False, index=False) # 160 observations
testSet.to_csv(join('..', 'testSet.csv'), header=False, index=False) # 32 observations