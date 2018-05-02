from collections import Counter

def write_vw(data, docs_id, path, column_modality=None):
    """
    Create VW file from DataFrame.

    Arguments:
    ---------
    data - DataFrame with data needed to convert to VW
    docs_id - DataSeries with document indexes
    path - string, path to store result data
    column_modality_relation - dictionary with relation between names of columns of data and name of modalities.
    Key is the name of column and value is the name of modality.
    If None all columns will be used to create modalities.
    In this case column_modality_relation are the names of columns.
    """

    vw_data = []
    if column_modality is None:
        column_modality = {}
        for column_name in data.columns:
            column_modality[column_name] = column_name

    with open(path, 'w', encoding='utf-8') as f:
        for doc, row in zip(docs_id, data.iterrows()):
            f.write('{0} '.format(doc))
            for column, modality_name in column_modality.items():
                f.write('|@{} '.format(modality_name))
                line = Counter(str(row[1][column]).split())
                for key in line.keys():
                    f.write('{0}:{1} '.format(key, line[key]))
            f.write('\n')

    return vw_data