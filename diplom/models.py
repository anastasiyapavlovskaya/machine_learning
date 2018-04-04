import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import glob
import pickle
import os

# to prepare data, feature_extraction
from vw_preprocess import write_vw
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import label_binarize

# to evaluate quality
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support, f1_score, log_loss
from sklearn.model_selection import train_test_split

# models
from sklearn import svm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import artm

# to create and visualize metrics
from viz_tools import create_confusion_matrix, save_tex_table

# for topic model vizualization
from tm_metrics_visualization import metrics_visualization

# to select features
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2

from collections import Counter

mpl.style.use('seaborn')


def svm_classifier(X, y, test_size=0.2):
    labels_decreasing_size_order = list(y.value_counts().index)

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    try:
        input_file = open('models_saved/svm.pkl', 'rb')
        clf = pickle.load(input_file)
    except FileNotFoundError:
        clf = svm.LinearSVC(penalty='l2')
        clf.fit(X_train, y_train)
        output_file = open('models_saved/svm.pkl', 'wb')
        pickle.dump(clf, output_file)

    y_pred = clf.predict(X_test)

    print('Accuracy_score: {}'.format(accuracy_score(y_test, y_pred)))
    # print(classification_report(y_test, y_pred, labels=labels_decreasing_size_order))

    cross_score = cross_val_score(clf, X_train, y=y_train, cv=8, scoring='accuracy')
    print('Cross_val_score: {:f} +- {:f}'.format(cross_score.mean(), cross_score.std()))

    create_confusion_matrix(y_test, clf.predict(X_test), labels=labels_decreasing_size_order).savefig('../../reports/svm_conf_matrix.png')

    return precision_recall_fscore_support(y_test, y_pred, labels=labels_decreasing_size_order)


def boosting_clf(X, y, test_size=0.2):
    labels_decreasing_size_order = list(y.value_counts().index)

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    try:
        input_file = open('models_saved/boosting.pkl', 'rb')
        clf = pickle.load(input_file)
    except FileNotFoundError:
        clf_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'n_jobs': -2,
            'random_state': 42
        }

        clf = LGBMClassifier(boosting_type='gbdt', **clf_params)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        output_file = open('models_saved/boosting.pkl', 'wb')
        pickle.dump(clf, output_file)

    y_pred = clf.predict(X_test)

    print('Accuracy_score: {}'.format(accuracy_score(y_test, y_pred)))
    # print(classification_report(y_test, y_pred, labels=labels_decreasing_size_order))

    # cross_score = cross_val_score(clf, X_train, y=y_train, cv=3, scoring='accuracy')
    # print('Cross_val_score: {:f} +- {:f}'.format(cross_score.mean(), cross_score.std()))

    create_confusion_matrix(y_test, clf.predict(X_test), labels=labels_decreasing_size_order).savefig('../../reports/boosting_conf_matrix.png')

    return precision_recall_fscore_support(y_test, y_pred, labels=labels_decreasing_size_order)


def forest_clf():
    clf = LGBMModel(boosting_type='rf')


def topic_model_clf(X, y, topic_num=30):
    labels_decreasing_size_order = list(y.value_counts().index)

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    file_train = 'temp_files/X_train.txt'
    file_test = 'temp_files/X_test.txt'

    temp_df = pd.DataFrame()
    temp_df['text'] = X_train
    temp_df['class_label'] = y_train
    write_vw(temp_df, X_train.index, file_train)

    temp_df = pd.DataFrame()
    temp_df['text'] = X_test
    write_vw(temp_df, X_test.index, file_test)


    if len(glob.glob(os.path.join('batches_train*.batch'))) < 1:
        batch_vectorizer_train = artm.BatchVectorizer(data_path=file_train,
                                                      data_format='vowpal_wabbit',
                                                      target_folder='batches_train',
                                                      gather_dictionary=True)
    else:
        batch_vectorizer_train = artm.BatchVectorizer(data_path='batches_train', data_format='batches',
                                                      gather_dictionary=True)

    if len(glob.glob(os.path.join('batches_test' + '*.batch'))) < 1:
        batch_vectorizer_test = artm.BatchVectorizer(data_path=file_test,
                                                     data_format='vowpal_wabbit',
                                                     target_folder='batches_test',
                                                     gather_dictionary=True)
    else:
        batch_vectorizer_test = artm.BatchVectorizer(data_path='batches_test', data_format='batches',
                                                     gather_dictionary=True)

    model = artm.ARTM(num_topics=topic_num,
                      class_ids={'@text': 5.0, '@class_label': 100.0},
                      cache_theta=True,
                      dictionary=batch_vectorizer_train.dictionary,
                      theta_columns_naming='title')

    scores = [
        artm.PerplexityScore(name='Perplexity', dictionary=batch_vectorizer_train.dictionary, class_ids=['@text']),
        artm.SparsityPhiScore(name='SparsityPhiText', class_id='@text'),
        artm.SparsityPhiScore(name='SparsityPhiClasses', class_id='@class_label'),
        artm.SparsityThetaScore(name='SparsityTheta'),
        artm.TopicKernelScore(name='TopicKernelText', probability_mass_threshold=0.1, class_id='@text'),
        artm.TopTokensScore(name='TopTokensText', class_id='@text', num_tokens=20),
        artm.TopTokensScore(name='TopTokensClasses', class_id='@class_label', num_tokens=10)
    ]

    regularizers = [
        artm.DecorrelatorPhiRegularizer(name='DeccorText', class_ids=['@text'], tau=10000),
        artm.SmoothSparsePhiRegularizer(name='SmoothPhiText', class_ids=['@text'], tau=0),
        artm.SmoothSparsePhiRegularizer(name='SmoothPhiClasses', class_ids=['@class_label'], tau=-1),
        # artm.SmoothSparsePhiRegularizer(name='SmoothBackgroundPhi', tau=100, topic_names=['background_topic']),
        artm.SmoothSparseThetaRegularizer(name='SmoothTheta', tau=-1.5),
        # artm.SmoothSparseThetaRegularizer(name='SmoothBackgroundTheta', tau=100, topic_names=['background_topic'])
    ]

    for r in regularizers:
        model.regularizers.add(r)
    for s in scores:
        model.scores.add(s)

    for i in tqdm(range(35)):
        model.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=1)

    p_cd = model.transform(batch_vectorizer=batch_vectorizer_test, predict_class_id='@class_label')

    # пооптимизируем это место
    y_pred = p_cd.idxmax(axis=0).astype(int)[[str(x) for x in X_test.index]].values
    # y_pred = p_cd[[str(x) for x in X_test.index]].idxmax(axis=0).values

    # metrics_visualization(target_pred=y_pred, target_true=y_test,
    #                       top_tokens_class=model.score_tracker['TopTokensClasses'],
    #                       top_tokens_text=model.score_tracker['TopTokensText'],
    #                       score_tracker=model.score_tracker,
    #                       scores_names=['Perplexity', 'SparsityPhiClasses',
    #                                     'SparsityPhiText', 'SparsityTheta'])

    print('Accuracy_score: {}'.format(accuracy_score(y_test, y_pred)))
    plt.hist(y_pred, color='g', label='pred')
    plt.hist(y_test, color='b', alpha=0.7, label='true')
    plt.title('Topic Model')
    plt.show()
    # print(classification_report(y_test, y_pred, labels=labels_decreasing_size_order))

    create_confusion_matrix(y_test, y_pred, labels=labels_decreasing_size_order).savefig('../../reports/topic_model_conf_matrix.png')

    micro_roc_auc = roc_auc_score(label_binarize(y_test, classes=list(range(0, 17))),
                                  p_cd.T,
                                  average='micro')
    macro_roc_auc = roc_auc_score(label_binarize(y_test, classes=list(range(0, 17))),
                                  p_cd.T,
                                  average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    log_loss_score = log_loss(y_test, p_cd.T)

    return (micro_roc_auc,
            macro_roc_auc,
            micro_f1,
            macro_f1,
            log_loss_score,
            precision_recall_fscore_support(y_test, y_pred, labels=labels_decreasing_size_order))


def logreg_clf(X, y, test_size=0.2):
    labels_decreasing_size_order = list(y.value_counts().index)

    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # сюда ещё можно эластичную сеточку приложить
    try:
        input_file = open('models_saved/logreg.pkl', 'rb')
        clf = pickle.load(input_file)
    except FileNotFoundError:
        clf = LogisticRegression(penalty='l2')
        clf.fit(X_train, y_train)
        output_file = open('models_saved/logreg.pkl', 'wb')
        pickle.dump(clf, output_file)

    y_pred = clf.predict(X_test)

    # print('Accuracy_score: {}'.format(accuracy_score(y_test, y_pred)))
    # print(classification_report(y_test, y_pred, labels=labels_decreasing_size_order))

    print('micro roc_auc_score:{}'.format(roc_auc_score(label_binarize(y_test, classes=list(range(0, 17))),
                                                        clf.predict_proba(X_test), average='micro')))
    print('macro roc_auc_score:{}'.format(roc_auc_score(label_binarize(y_test, classes=list(range(0, 17))),
                                                        clf.predict_proba(X_test), average='macro')))

    print('macro f1_score:{}'.format(f1_score(y_test, y_pred, average='macro')))
    print('micro f1_score:{}'.format(f1_score(y_test, y_pred, average='micro')))

    # cross_score = cross_val_score(clf, X_train, y=y_train, cv=8, scoring='accuracy')
    # print('Cross_val_score: {:f} +- {:f}'.format(cross_score.mean(), cross_score.std()))
    plt.bar(list(Counter(y_test).keys()), list(Counter(y_test).values()), color='g', label='true')
    plt.bar(list(Counter(y_pred).keys()), list(Counter(y_pred).values()), color='b', alpha=0.7, label='pred')
    plt.title('LogReg')
    plt.legend()
    plt.show()
    create_confusion_matrix(y_test, clf.predict(X_test), labels=labels_decreasing_size_order).savefig('../../reports/logreg_conf_matrix.png')

    return precision_recall_fscore_support(y_test, y_pred, labels=labels_decreasing_size_order)


def feature_selection(X, y, test_size=0.2):
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    selector = GenericUnivariateSelect(score_func=chi2, mode='percentile', param=70)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    print('Before selection shape:', X_train.shape)
    print('After selection shape:', X_train_selected.shape)

    return (X_train_selected< X_test_selected)


if __name__ == '__main__':
    data = pd.read_csv('../../data/all_categories_csv.csv',
                       index_col='Unnamed: 0',
                       low_memory=False)
    X = data['Lemmatized']
    y = data['Категория жалобы']

    X_tfidf = TfidfVectorizer().fit_transform(X)
    X_bow = CountVectorizer().fit_transform(X)

    report = pd.DataFrame()

    # print('1) tf-idf representation')
    # print('SVM model')
    # clf_report = svm_classifier(X_tfidf, y)
    # # report['svm_precision_tfidf'] = clf_report[0]
    # # report['svm_recall_tfidf'] = clf_report[1]
    # report['svm_fscore_tfidf'] = clf_report[2]

    # print('GRADIENT BOOSTING model')
    # clf_report = boosting_clf(X_tfidf, y)
    # # report['grad_boosting_precision_tfidf'] = clf_report[0]
    # # report['grad_boosting_recall_tfidf'] = clf_report[1]
    # report['grad_boosting_fscore_tfidf'] = clf_report[2]
    #
    print('LOGREG model')
    clf_report = logreg_clf(X_tfidf, y)
    # report['logreg_precision_tfidf'] = clf_report[0]
    # report['logreg_recall_tfidf'] = clf_report[1]
    # report['logreg_fscore_tfidf'] = clf_report[2]

    scores = []
    for i in np.linspace(0, 300, 6):
        print('TOPIC model with {} topics'.format(i))
        scores.append(topic_model_clf(X, y))
    # with open('../../reports/tm_scores.txt', 'w') as output_file:

    # report['tm_precision'] = clf_report[0]
    # report['tm_recall'] = clf_report[1]
    # report['tm_fscore'] = clf_report[2]

    # print('LOGREG model with features from topic model')
    # X_tm = pd.read_csv('theta_10000theams.csv', index_col='Unnamed: 0').transpose()
    # clf_report = logreg_clf(X_tm, data.loc[list(map(int, X_tm.index)), 'Категория жалобы'])
    #
    # report['logreg_precision_tm_features'] = clf_report[0]
    # report['logreg_recall_tm_features'] = clf_report[1]
    # report['logreg_fscore_tm_features'] = clf_report[2]
    #
    # report['support'] = clf_report[-1]
    # df_report = pd.DataFrame.from_dict(report)

    save_tex_table(report, '../../reports/report.txt')
