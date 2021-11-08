import pandas as pd
from os.path import join

# readout xls data
fname = join('..', 'GroundTruth.xls')
data = pd.read_excel(fname)
data = data.iloc[0:192]
data = data[['# Patients','Evaluation','Age','Sex','Set Distribution']]

# extract colums, assign processed data
data['# Patients'] = data['# Patients'].apply(str) + '.jpg' # filename = paitentID.jpg
data['Evaluation'] = data['Evaluation'].transform(lambda x: 0 if x == 'Normal' else 1) # x150 Abnormal, x42 Normal

# sorting data depend on the set type
trainSet = data.loc[data['Set Distribution'] == 'Train']
testSet = data.loc[data['Set Distribution'] == 'Test']
validationSet = data.loc[data['Set Distribution'] == 'Validation']

# check processed data
print(trainSet.head())
print(testSet.head())
print(validationSet.head())

# write .csv file
trainSet.to_csv(join('..', 'trainSet.csv'), header=False, index=False)
testSet.to_csv(join('..', 'testSet.csv'), header=False, index=False)
validationSet.to_csv(join('..', 'validationSet.csv'), header=False, index=False)