import pandas as pd
import xlrd
import csv
from os import listdir

def xlsx_to_csv(filename, path_to_file, path_to_save, verbose=False):
    '''
    INPUT   filename (no extensions) -- as STRING
            path_to_file so that path_to_file/filename.xlsx is the file to save as csv -- as STRING
            path_to_save so that path_to_save/filename.csv is the saved csv file -- as STRING

    '''
    filename_xlsx = filename+'.xlsx'
    path_xlsx = path_to_file + '/' + filename_xlsx
    wb = xlrd.open_workbook(path_xlsx)
    sh = wb.sheet_by_name('Worksheet')
    path_csv = path_to_save+ '/' + filename +'.csv'

    your_csv_file = open(path_csv, 'wb')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    index = 0
    for rownum in xrange(sh.nrows):
        index += 1
        list_rows = []
        for word in sh.row_values(rownum):
            if type(word) == unicode:
                list_rows.append(word.encode('ascii', 'ignore'))
            else:
                list_rows.append(word)
        wr.writerow(list_rows)

    if verbose:
        print 'file %s was converted to csv'%filename
    your_csv_file.close()

def xlsx_to_csv_dir(path_to_dir, path_to_dir_csv, verbose=False):
    '''
    calls xlsx_to_csv on all files in directory 'dir' and adds the converted
    files to 'dir_csv'
    all files in directory 'dir' must be .xlsx
    INPUT path_to_dir: as STRING
          path_to_dir_csv: as STRING
    '''
    files_to_convert = listdir(path_to_dir)
    for filename in files_to_convert:
        filename = filename[:-5]
        xlsx_to_csv(filename, path_to_dir, path_to_dir_csv, verbose)
