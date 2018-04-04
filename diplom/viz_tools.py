import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools


def create_confusion_matrix(y_test, y_pred, labels=None):
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(8)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion matrix', fontsize=16)

    if labels is not None:
        plt.xticks(np.arange(0, len(labels)), labels, rotation=90, fontsize=10)
        plt.yticks(np.arange(0, len(labels)), labels, fontsize=10)
    else:
        plt.xticks(np.arange(0, len(np.unique(y_test))), rotation=90, fontsize=10)
        plt.yticks(np.arange(0, len(np.unique(y_test))), fontsize=10)
    fmt = 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(np.arange(cnf_matrix.shape[0]), np.arange(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    # plt.savefig('confusion_matrix.pdf')

    return fig

def save_tex_table(table, path):
    """
    Takes table and saves it in document in latex format.
    :param table: table in csv
    :param path: path of doc to put table in
    :return:
    """
    with open(path, 'w') as output_file:
        output_file.write('\\begin{tabular}{'+''.join(['| r ' for x in table.columns.values]) +'| } \n')
        output_file.write('\\hline \n')

        headers = ' & '.join(table.columns.values)
        output_file.write(headers)
        output_file.write(' \\\\ \n\\hline \n')

        for row in table.iterrows():
            string = ' & '.join([str(round(x, 2)) for x in row[1].values])
            output_file.write(string +  ' \\\\ \n')
            output_file.write('\hline \n')

        output_file.write('\end{tabular}')

if __name__ == '__main__':
    frame = [{'A': 1, 'B': 2, 'C': 1},
             {'A': 10, 'B': -2, 'C': 10}]
    df = pd.DataFrame(frame)

    save_tex_table(df, 'example.txt')