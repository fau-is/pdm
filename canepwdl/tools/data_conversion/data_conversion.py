
import pandas as pd
import csv
import category_encoders as categoryEncoders  


def createMapping(dfSource, dfTarget, column):
    dfTarget[column] = dfSource[column]
    # fill gaps with dummy value
    dfTarget[column].fillna("dummy", inplace=True)
    
    # map unique values to integer
    uniqueRows = dfTarget[column].unique().tolist()
    uniqueRowsMapping = dict(zip(uniqueRows, range(len(uniqueRows))))
    dfTarget[column] = dfTarget[column].map(uniqueRowsMapping)
    
    print('Save mapping of '+column+' to csv file.')
    with open("./output/"+logName[:(len(logName)-4)]+"_"+column+'Mapping.csv', 'w') as csv_file:
    	writer = csv.writer(csv_file)
    	for key, value in uniqueRowsMapping.items():
    		writer.writerow([key, value])
            
    return dfTarget


	
def normalizeColumn(dfSource, dfTarget, column):
	dfTarget[column] = dfSource[column]
	#fill gaps with mean value
	dfTarget[column].fillna(dfTarget[column].mean(), inplace=True)
	dfTarget[column] = dfTarget[column] / dfTarget[column].max()
    
	return dfTarget



def encodeOneHot(dfSource, dfTarget, column):
    dfSource[column].fillna("dummy", inplace=True)
    oneHotEncoding = pd.get_dummies(dfSource[column])
    cols = oneHotEncoding.columns.tolist()
    
    for i, c in enumerate(cols):
        cols[i] = column + "_%s" % c
        
    oneHotEncoding.rename(columns=dict(zip(oneHotEncoding.columns.tolist(), cols)), inplace=True)     
    dfTarget = dfTarget.join(oneHotEncoding)
    
    return dfTarget



def encodeBinary(dfSource, dfTarget, column):
    binaryEncoder = categoryEncoders.BinaryEncoder(cols = [column])
    binaryEncoding = binaryEncoder.fit_transform(dfSource[column])
    dfTarget = dfTarget.join(binaryEncoding)
    
    return dfTarget



def encodeHash(dfSource, dfTarget, column, outputColumns):
    dfSource[column].fillna("dummy", inplace=True)  
    hashingEncoder = categoryEncoders.HashingEncoder(n_components=outputColumns)
    print('Creating Hash Encoding...')
    hashingEncoding = hashingEncoder.fit_transform(dfSource[column])
    
    cols = hashingEncoding.columns.tolist()
    for i, c in enumerate(cols):
        cols[i] = column + "_%s" % i
        
    hashingEncoding.rename(columns=dict(zip(hashingEncoding.columns.tolist(), cols)), inplace=True)
    dfTarget = dfTarget.join(hashingEncoding)     
    
    return dfTarget




message  = 'Which logfile should be processed?'
logName = input("%s " % message)
#logName = "bpi2019_sample.csv"

print('Start reading CSV file')
# make sure CSV is semicolon-separated, in utf-8 and headers do not include any spaces (else maybe this helps sep='\s*,\s*')

dfSource = pd.read_csv("./input/"+logName, encoding='cp1252', sep=';')#utf-8
print('Finished reading CSV file')

# iterate each column for conversion
dfTarget = dfSource[['case','event','time']].copy()

hashColumns = []

for column in dfSource:
    # control-flow attribute
    if column == 'case' or column == 'event':
        print('Create mapping of '+ column)
        dfTarget = createMapping(dfSource, dfTarget, column)
	
    else:
        # control-flow attribute
        if column == 'time':
            print('Transform timestamp to d.m.Y-H:M:S')
            dfTarget[column] = pd.to_datetime(dfTarget[column])
            dfTarget[column] = dfTarget[column].dt.strftime('%d.%m.%Y-%H:%M:%S')
            
        else:
            # contextual attributes
            uniqueRows = dfSource[column].unique().tolist()
            print('\n\nColumn ' + column + ' seems of type ' + str(dfSource[column].dtype) + ' and has ' + str(
				len(uniqueRows)) + ' unique entries. ' + str(dfSource[column].isnull().sum()) + ' cells are empty.')
            message = 'Include the column as integer-mapped, normalized, one-hot, binary or hash encoded column(s) in the dataset? (int/norm/onehot/bin/hash/no)'
            answer = input("%s" % message)
            
            if answer == 'norm':
                dfTarget = normalizeColumn(dfSource, dfTarget, column)

            elif answer == 'int':
                dfTarget = createMapping(dfSource, dfTarget, column)
                    
            elif answer == 'onehot':
                dfTarget = encodeOneHot(dfSource, dfTarget, column)
                    
            elif answer == 'bin':
                dfTarget = encodeBinary(dfSource, dfTarget, column)
                
            elif answer == 'hash':
                message = 'How many output columns? (default=8)'
                answer = input("%s" % message)
                try:
                    outputColumns = int(answer)
                except ValueError:
                    outputColumns = 8
                dfTarget = encodeHash(dfSource, dfTarget, column, outputColumns)
            
                
                

print(dfTarget.head(10))
print('Write converted log to CSV')
dfTarget.to_csv("./output/"+logName[:(len(logName)-4)]+'_converted.csv', encoding='utf-8', sep=';', index=False)