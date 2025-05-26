import os

import pandas as pd
import numpy as np
import pingouin as pg
import pydmr

root = os.getcwd()

resultspath = os.path.join(root, 'build', 'Tables')


def table_demographics():

    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    demographics = [
        'Age', 'BMI', 'Height', 'Weight', 
        'Crea', 'Urea', 
        'AvrALP', 'AvrALT', 'AvrAlb', 'AvrBili', 'AvrConBili', 'AvrConTotBili', 
    ]
    dmr = pydmr.read(file, 'pandas', parameter=demographics, study=['control', 'screening'])
    df = dmr['pars'].pivot(columns='subject', index='parameter', values='value')
    df = df.T.describe().T.round(1)

    name = lambda x: f"{pydmr.metadata(file, x, 'description').replace('Average ', '')} ({pydmr.metadata(file, x, 'unit')})"
    df = df.rename(name, axis=0)

    file = os.path.join(resultspath, 'table_demographics.csv')
    df.to_csv(file)


def table_lft_visits():

    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')

    init = ['PreALP', 'PreALT', 'PreAlb', 'PreBili', 'PreConBili', 'PreConTotBili']
    dmr = pydmr.read(file, 'pandas', study='control', parameter=init)
    df0 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    init = ['InitALP', 'InitALT', 'InitAlb', 'InitBili', 'InitConBili', 'InitConTotBili']
    dmr = pydmr.read(file, 'pandas', study='drug', parameter=init)
    df1 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    # Extract only cases with follow-ups
    df0 = df0.loc[df1.index]

    df0 = df0.rename(lambda x: x.replace('Pre', ''), axis=1)
    df1 = df1.rename(lambda x: x.replace('Init', ''), axis=1)
    diff = df1-df0

    # perform ttest
    result = [
        pg.ttest(df0[var].values, df1[var].values, paired=True) 
        for var in df0.columns
    ]
    result = pd.concat(result)
    result['parameter'] = df0.columns
    result = result.set_index('parameter')

    df0 = df0.describe().T
    df1 = df1.describe().T
    diff = diff.describe().T

    avr = np.around(df0['mean'].values, 1)
    ci = np.around(1.96*df0['std'].values, 1)
    col1 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)]

    avr = np.around(df1['mean'].values, 1)
    ci = np.around(1.96*df1['std'].values, 1)
    col2 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)] 

    avr = np.around(diff['mean'].values, 1)
    ci = np.around(1.96*diff['std'].values, 1)
    col3 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)]  

    data = {
        'V1': col1,
        'V2': col2,
        'Diff': col3,
        'p': result['p-val'].round(3),
    }
    table = pd.DataFrame(data, index=diff.index)
    
    file = os.path.join(resultspath, 'table_lft_visits.csv')
    table.to_csv(file)



def main():

    if not os.path.exists(resultspath):
        os.makedirs(resultspath)

    table_demographics()
    table_lft_visits()



if __name__=='__main__':
    main()
