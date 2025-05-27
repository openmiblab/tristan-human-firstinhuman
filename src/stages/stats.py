import os


import numpy as np
import pandas as pd
import pingouin as pg
import pydmr

from stages import data

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

resultspath = os.path.join(root, 'build', 'Tables')



def lft_between_visits():

    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')

    # Values at the start of the drug visit before administering the drug
    init = ['InitALP', 'InitALT', 'InitAlb', 'InitBili', 'InitConBili', 'InitConTotBili']
    dmr = pydmr.read(file, 'pandas', study='drug', parameter=init)
    df1 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    # Values before the control scan
    init = ['PreALP', 'PreALT', 'PreAlb', 'PreBili', 'PreConBili', 'PreConTotBili']
    dmr = pydmr.read(file, 'pandas', study='control', parameter=init)
    df0 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    # Extract only baseline cases with follow-ups
    df0 = df0.loc[df1.index]

    # Build the difference
    df0 = df0.rename(lambda x: x.replace('Pre', ''), axis=1)
    df1 = df1.rename(lambda x: x.replace('Init', ''), axis=1)
    diff = df1-df0

    # Perform paired ttest between visits
    result = [
        pg.ttest(df0[var].values, df1[var].values, paired=True) 
        for var in df0.columns
    ]
    result = pd.concat(result)
    result['parameter'] = df0.columns
    result = result.set_index('parameter')

    # Build average and confidence intervals (control visit)
    df0 = df0.describe().T
    avr = np.around(df0['mean'].values, 1)
    ci = np.around(1.96*df0['std'].values, 1)
    col1 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)]

    # Build average and confidence intervals (drug visit)
    df1 = df1.describe().T
    avr = np.around(df1['mean'].values, 1)
    ci = np.around(1.96*df1['std'].values, 1)
    col2 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)] 

    # Build average and confidence intervals (difference)
    diff = diff.describe().T
    avr = np.around(diff['mean'].values, 1)
    ci = np.around(1.96*diff['std'].values, 1)
    col3 = [f"{avr[i]} ({ci[i]})" for i in range(avr.size)]  

    # Save results as csv
    file = os.path.join(resultspath, 'table_lft_between_visits.csv')
    data = {
        'V1': col1,
        'V2': col2,
        'Diff': col3,
        'p': result['p-val'].round(3),
    }
    table = pd.DataFrame(data, index=diff.index)
    table.to_csv(file)


def ttest():
    
    # Concatenate all data
    file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')
    vals = pydmr.read(file, 'pandas')['pars']

    # Perform t-tests and save if df output
    output = None
    for par in np.sort(vals.parameter.unique()):
        data_par = vals[vals.parameter==par]
        stats = pg.pairwise_tests(
                data=data_par, 
                dv='value', within='study', subject='subject', 
                return_desc=False, effsize='odds-ratio',
        )
        stats['parameter'] = par
        if output is None:
            output = stats
        else:
            output = pd.concat([output, stats])

    # Save results
    file = os.path.join(resultspath, 'ttest.csv')
    output.to_csv(file, index=False, na_rep="NaN")


def univariate():
    
    # Sort table with averages
    avrgs = pd.read_csv(os.path.join(resultspath, 'averages.csv'))
    avrgs = avrgs.set_index('parameter') 

    # Sort t-test table and drop uninformative columns
    stats = pd.read_csv(os.path.join(resultspath, 'ttest.csv')) 
    stats = stats.set_index('parameter') 
    cols = ['Contrast', 'A', 'B', 'Paired', 'Parametric', 'dof', 'alternative']
    stats = stats.drop(columns=cols) 

    # Combine averages and t-test results
    output = pd.concat([avrgs, stats], axis=1)

    # Rename columns
    output.rename(columns={"p-unc": "p-value"}, inplace=True)
    
    b_avr = output['control Mean'].values
    b_err = output['control 95%CI'].values
    r_avr = output['drug Mean'].values
    r_err = output['drug 95%CI'].values
    c_avr = output['Effect Mean'].values
    c_err = output['Effect 95%CI'].values

    output['control'] = [
        f'{b_avr[i]} ({b_err[i]}) ' for i in range(b_avr.size)]
    output['drug'] = [
        f'{r_avr[i]} ({r_err[i]}) ' for i in range(r_avr.size)]
    output['Effect size (%)'] = [
        f'{c_avr[i]} ({c_err[i]}) ' for i in range(c_avr.size)]
    
    # Lookup group, units and add as column
    unit = data.lookup_vals(output.index.values, 'unit')
    name = data.lookup_vals(output.index.values, 'name')
    output['Name'] = [f"{name[i].replace('Average ', '')} ({unit[i]})" for i in range(len(unit))]
    output['Group'] = data.lookup_vals(output.index.values, 'group')

    # Sort and format values
    output = output.sort_values(by=['Group', 'p-value'])
    output.loc[:,'T'] = np.around(output['T'].values, 1)
    output.loc[:,'p-value'] = np.around(output['p-value'].values, 3)
    
    # Retain most informative columns
    output = output[
        ['Name', 'Group', 'control', 'drug', 
         'Effect size (%)', 'T', "p-value"]
    ]
    
    # Split up for aorta and liver, rename groups and save to csv.
    output_liver = output[output.Group != 'MRI - aorta']
    output_liver = output_liver.replace({'Blood - liver function test': 'LFT', 'MRI - liver': 'MRI'})
    output_liver.to_csv(os.path.join(resultspath, 'table_liver_univariate.csv'))

    output_aorta = output[output.Group == 'MRI - aorta']
    output_aorta = output_aorta.replace({'MRI - aorta': 'MRI'})
    output_aorta.to_csv(os.path.join(resultspath, 'table_aorta_univariate.csv'))


def correlations_control():
    
    file = os.path.join(resultspath, 'vals_control.csv')
    vals_control = pd.read_csv(file).set_index('parameter')
    file = os.path.join(resultspath, 'vals_screening.csv')
    vals_screening = pd.read_csv(file).set_index('parameter')
    vals = pd.concat([vals_control, vals_screening]).T

    corr = []
    vars = set(vals.columns) - set(data.EXCLUDE_EFFECT)
    for X in vars:
        for Y in vars:
            x = vals[X].values
            y = vals[Y].values
            c = pg.corr(x,y)
            c['X'] = X
            c['Y'] = Y
            corr.append(c)
    corr = pd.concat(corr)
    corr.rename(columns={'p-val':'p-unc'}, inplace=True)

    correlations_control_submatrix(corr, 'MRI - liver', 'corr_control')
    correlations_control_submatrix(corr, 'MRI - aorta', 'corr_aorta_control')
    correlations_control_submatrix(corr, 'Blood - liver function test', 'corr_blood_control')
    correlations_control_submatrix(corr, 'Screening', 'corr_screening')


def correlations_control_submatrix(corr, X, filename):

    Y = 'MRI - liver'

    # select parameters
    xpars = data.lookup_params('group', X)
    ypars = data.lookup_params('group', Y)
    corr = corr[(corr.X.isin(xpars)) & (corr.Y.isin(ypars))]

    # Pivot to wide format
    vals = corr.pivot(values='r', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_vals.csv')
    vals.to_csv(file, na_rep="NaN")

    pval = corr.pivot(values='p-unc', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_pval.csv')
    pval.to_csv(file, na_rep="NaN")


def correlations_effect():

    file = os.path.join(resultspath, 'effect_size_absolute.csv')
    effect = pd.read_csv(file, index_col=0)
    #corr = pg.pairwise_corr(effect)
    corr = []
    vars = set(effect.columns)
    for X in vars:
        for Y in vars:
            x = effect[X].values
            y = effect[Y].values
            c = pg.corr(x,y)
            c['X'] = X
            c['Y'] = Y
            corr.append(c)
    corr = pd.concat(corr)
    corr.rename(columns={'p-val':'p-unc'}, inplace=True)

    correlations_effect_submatrix(corr, 'MRI - liver', 'corr_liver_effect')
    correlations_effect_submatrix(corr, 'MRI - aorta', 'corr_aorta_effect')
    correlations_effect_submatrix(corr, 'Blood - liver function test', 'corr_blood_effect')


def correlations_effect_submatrix(corr, X, filename):

    Y = 'MRI - liver'

    # select parameters
    xpars = data.lookup_params('group', X)
    ypars = data.lookup_params('group', Y)
    corr = corr[(corr.X.isin(xpars)) & (corr.Y.isin(ypars))]

    # Pivot to wide format
    vals = corr.pivot(values='r', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_vals.csv')
    vals.to_csv(file, na_rep="NaN")

    pval = corr.pivot(values='p-unc', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_pval.csv')
    pval.to_csv(file, na_rep="NaN")








def main():

    lft_between_visits()
    ttest()
    univariate()
    correlations_control()
    correlations_effect()
    


if __name__=='__main__':
    main()



