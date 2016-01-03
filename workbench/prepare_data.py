from os import listdir
import pandas as pd

dict_agency = {'dod' : 'Department of Defense',
               'dhs' : 'Department of Homeland Security',
               'dt'  : 'Department of Transportation',
               'dc'  : 'Department of Commerce',
               'ded' : 'Department of Education',
               'den' : 'Department of Energy',
               'epa' : 'Environmental Protection Agency',
               'nasa': 'National Aeronautics and Space Administration',
               'dhhs': 'Department of Health and Human Services',
               'nsf' : 'National Science Foundation',
               'da'  : 'Department of Agriculture'}

def subset_data(agency=None, year_phase_I=None,
                path_dir_csv='', num_csv_files=None):
    '''
    INPUT
        agency: acronym ('dod') or full name ('Department of Defense')
                    as STRING
                    (see dict_agency for list of acronyms and full names)
                    if None, takes all departements
        year_phase_I: 'Award Year' for phase I
                    as INT
                    if None, takes all years

        path_dir_csv: path to directory containing pertinent csv files
                    as STRING
        num_csv_files: number of CSV files to take into account, if None, takes
            all files from directory
                    as INT

    OUTPUT:
        pandas DataFrame of companies with a phase I award in the given year,
        for the given agency
    '''
    #getting the list of csv files:
    list_csv = listdir(path_dir_csv)
    if type(num_csv_files) == int:
        list_csv = list_csv[:num_csv_files]

    print list_csv
    #number of files to load
    num_file_csv = len(list_csv)
    #loading the data into pandas DataFrames
    list_df = []
    for filename in list_csv:
        file_path = path_dir_csv + '/' + filename
        df_from_file = pd.read_csv(file_path)
        list_df.append(df_from_file)

    #one general pandas DataFrames
    df = pd.concat(list_df)

    #separating into phase I and phase II
    df_phaseI = df[df['Phase']=='Phase I']
    df_phaseII = df[df['Phase']=='Phase II']

    #tracking number to identify unique awards
    phase_I_tracking = df_phaseI['Agency Tracking #'].unique()
    phase_II_tracking = df_phaseII['Agency Tracking #'].unique()

    #tracking number of those in phase I that made it to phase II
    phase_I_to_II = set(phase_II_tracking).intersection(phase_I_tracking)

    #creating label 'to_phase_II'
    df_phaseI['to_phase_II'] = df_phaseI['Agency Tracking #']\
                        .apply(lambda x: is_in_phase_I_to_II(x, phase_I_to_II))

    if agency != None:
        if len(agency)<=5:
            agency = dict_agency[agency]
        mask_agency = (df_phaseI['Agency']==agency)
        df_phaseI = df_phaseI[mask_agency]

    if year_phase_I != None:
        mask_year = (df_phaseI['Award Year']==year_phase_I)
        df_phaseI = df_phaseI[mask_year]

    #reset index to 0, 1, ...
    df_phaseI.reset_index(inplace=True)

    return df_phaseI

def is_in_phase_I_to_II(x, phase_I_to_II):
    '''
    INPUT:
        x: tracking number (eg: 'F061-202-1036')
            as STRING
        phase_I_to_II: list of tracking numbers that appear in phase I and II
             as LIST
    OUTPUT: as BOOL tracking number appears for phase I and II
    '''
    return int(x in list(phase_I_to_II))
