import os
import pandas as pd
import numpy as np
import pydmr
import miblab

root = os.getcwd()

# To control how pandas interprets the csv's
TYPES = {
    "subject": str, 
    "visit": str, 
    "group": str,
    "name": str,
    "value": float,
    "unit": str,
    "stdev": float,
    "parameter": str,
    "cluster": str
}

# Exclude variables that are analytically related to others from correlation
# analysis
EXCLUDE_CORREL = [
#    'RE_R1l',
    'Th_i', 'Th_f', 'Th', 
    'khe_i', 'khe_f', 'kbh_i', 'kbh_f',
    'Kbh', 'Khe', 
    'khe_slope', 'kbh_slope',
    #'AvrALP', 'AvrALT', 'AvrAlb', 'AvrBili', 'AvrConBili', 'AvrConTotBili',
    'PreALP', 'PreALT', 'PreAlb', 'PreBili', 'PreConBili', 'PreConTotBili',
    'PostALP', 'PostALT', 'PostAlb', 'PostBili', 'PostConBili', 'PostConTotBili',
]


def read(effect=False):
    if effect:
        file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')
    else:
        file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    data = pydmr.read(file, 'table')
    cols = ['subject', 'visit', 'parameter', 'value']
    return (
        pd.DataFrame(data['pars'], columns=cols),
        pd.DataFrame(data['sdev'], columns=cols),
    )


IND = {
    'name': 0,
    'unit': 1,
    'type': 2, 
    'group': 3,
    'label': 4,
    'value_map': 5,
    'cluster': 6,
}

def lookup_vals(parameters, prop):
    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    data = pydmr.read(file)['data']
    if isinstance(parameters, str):
        return data[parameters][IND[prop]]
    else:
        return [data[p][IND[prop]] for p in parameters]

def lookup_params(prop, value):
    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    data = pydmr.read(file)['data']
    return [p for p in data if data[p][IND[prop]]==value]


def extend_data():

    # Replace metadata with static metadata containing clusters
    file = os.path.join(root, '_static', 'MRI_metadata')
    meta = pydmr.read(file)
    folder = os.path.join(root, 'build', 'Data')
    for dataset in [
            'tristan_humans_healthy_controls_all_results.dmr.zip', 
            'tristan_humans_healthy_rifampicin_all_results.dmr.zip',
        ]:
        file = miblab.zenodo_fetch(dataset, folder, '15514373')
        dmr = pydmr.read(file)
        dmr['data'] = meta['data']
        dmr['columns'] = meta['columns']
        pydmr.write(file, dmr)

    # Combine all data in a single Data.dmr file
    files = [
        os.path.join(folder, 'tristan_humans_healthy_controls_all_results.dmr.zip'),
        os.path.join(folder, 'tristan_humans_healthy_rifampicin_all_results.dmr.zip'),
        miblab.zenodo_fetch('tristan_humans_leeds_covariates.dmr.zip', folder, '15514373'),
    ]
    file = os.path.join(folder, 'all_data.dmr')
    pydmr.concat(files, file)
    pydmr.drop(file, subject=['SHF-007', 'SHF-010', 'SHF-016'])
    data = pydmr.read(file, 'nest')

    # Compute standard deviations for derived parameters
    for subj in data['pars']:
        for study in data['pars'][subj]:
            try:
                v = data['pars'][subj][study]
                sd = data['sdev'][subj][study]
                Th_sd = 0.5 * np.sqrt(sd['Th_i']**2 + sd['Th_f']**2)
                khe_sd = 0.5 * np.sqrt(sd['khe_i']**2 + sd['khe_f']**2)
                kbh_sd = np.sqrt(
                    (Th_sd * (1 - v['ve']) / v['Th']**2)**2 + 
                    (sd['ve'] / v['Th'])**2
                )
                sd['khe'] = khe_sd
                sd['kbh'] = kbh_sd
                sd['Th'] = Th_sd
            except KeyError:
                pass

    # Define derived parameters
    data['data'] = data['data'] | {
        'khe_slope': ['khe change per time', 'mL/min/100cm3/hr', 'float', 'MRI - liver', 'Dk(he)', 'NaN', 'HC'],
        'kbh_slope': ['kbh change per time', 'mL/min/100cm3/hr', 'float', 'MRI - liver', 'Dk(bh)', 'NaN', 'HC'],
        'R1_45min': ['R1 at 45mins', '1/sec', 'float', 'MRI - liver', 'R1(45)', 'NaN', 'SQ'],
        'DR1_45min': ['Delta R1 at 45mins', '1/sec', 'float', 'MRI - liver', 'DR1(45)', 'NaN', 'SQ'],
        'R1_scan2': ['R1 (scan 2)', '1/sec', 'float', 'MRI - liver', 'R1(2)', 'NaN', 'SQ'],
        'DR1_scan2': ['Delta R1 (scan 2)', '1/sec', 'float', 'MRI - liver', 'DR1(2)', 'NaN', 'SQ'],
        'AvrAlb': ['Average Albumin', 'g/L', 'float', 'Blood - liver function test', 'Alb', 'NaN', 'Blood - liver function test'],
        'AvrALP': ['Average ALP', 'U/L', 'float', 'Blood - liver function test', 'ALP', 'NaN', 'Blood - liver function test'],
        'AvrALT': ['Average ALT', 'U/L', 'float', 'Blood - liver function test', 'ALT', 'NaN', 'Blood - liver function test'],
        'AvrBili': ['Average Bilirubin', 'umol/L', 'float', 'Blood - liver function test', 'Bil', 'NaN', 'Blood - liver function test'],
        'AvrConBili': ['Average Conjugated Bilirubin', 'umol/L', 'float', 'Blood - liver function test', 'CBil', 'NaN', 'Blood - liver function test'],
        'AvrConTotBili': ['Average Conjugated/total bilirubin', '%', 'float', 'Blood - liver function test', 'CTBil', 'NaN', 'Blood - liver function test'],
    }

    # Compute derived parameters
    for subj in data['pars']:
        for study in data['pars'][subj]:
            try:
                v = data['pars'][subj][study]
                v['khe_slope'] = (v['khe_f']-v['khe_i'])/(v['t3']-v['t0'])
                v['kbh_slope'] = (v['kbh_f']-v['kbh_i'])/(v['t3']-v['t0'])
                v['R1_45min'] = 1/v['T1_2']
                v['DR1_45min'] = 1/v['T1_2'] - 1/v['T1_1']
                v['R1_scan2'] = 1/v['T1_3']
                v['DR1_scan2'] = 1/v['T1_3'] - 1/v['T1_1']
                for p in ['Alb', 'ALP', 'ALT', 'Bili', 'ConBili', 'ConTotBili']:
                    v['Avr'+p] = (v['Pre'+p] + v['Post'+p])/2
            except KeyError:
                pass

    # Save results
    pydmr.write(file, data, 'nest')



NO_EFFECT = [
    # Effect size in timings is irrelevant
    'BAT', 'BAT2', 'S02a', 'S02l',
    't0', 't1', 't2', 't3', 'dt1', 'dt2',
    't1_MOLLI', 't2_MOLLI', 't3_MOLLI',
    # Constant
    'H', 
    # Initial values only measured once (drug visit)
    'InitAlb', 'InitALP', 'InitALT', 'InitBili',
    'InitConBili', 'InitConTotBili',
]

# Exclude parameters that are similar from effect size analysis
EXCLUDE_EFFECT = [
    # Same effect size as AUC Conc
    'AUC35_R1l', 'AUC_R1l', 
    'AUC35_R1b', 'AUC_R1b', 
    # Very similar to AUC inf
    'AUC_Cl', 'AUC_Cb',
    # R1 rather than T1 (inhibition=reduction)
    'T1_1', 'T1_2', 'T1_3',
    # Very similar to khe and kbh
    'Kbh', 'Khe', 
    # Init and final values
    'khe_i', 'khe_f', 'kbh_i', 'kbh_f',
    'PreALP', 'PreALT', 'PreAlb', 'PreBili', 'PreConBili', 'PreConTotBili',
    'PostALP', 'PostALT', 'PostAlb', 'PostBili', 'PostConBili', 'PostConTotBili',
]

def effect_data():

    # Parameters with uninteresting effect sizes

    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    result = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')
    pydmr.drop(file, result, parameter=NO_EFFECT+EXCLUDE_EFFECT, study='screening', subject=['1','5'])


def main():
    extend_data()
    effect_data()

if __name__=='__main__':
    main()