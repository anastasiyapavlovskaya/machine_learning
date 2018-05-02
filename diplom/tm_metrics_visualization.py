from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import itertools

def metrics_visualization(target_pred=None, target_true=None, target_names=None, conf_matrix_name=None, 
                          top_tokens_class=None, top_tokens_text=None,
                          show_docs=False, docs_data_path='../../data/X_train.csv', doc_count=3, theta_matrix=None,
                          score_tracker=None ,scores_names=['Perplexity', 'SparsityPhiClasses', 
                                                            'SparsityPhiText', 'SparsityTheta']):
    
    #write accuracy and other classification metrics
    if target_pred is not None and target_true is not None:
        print('accuracy: {}'.format(accuracy_score(target_true, target_pred)))
        if target_names is not None:
            print('\033[1m' + 'Classification report: \n', classification_report(target_true, target_pred, target_names=target_names))
        else:
            print('\033[1m' + 'Classification report: \n', classification_report(target_true, target_pred))
        
        cnf_matrix = confusion_matrix(target_true, target_pred)

        fig, ax = plt.subplots()
        if len(np.unique(target_true)) > 20:
            fig.set_figwidth(30)
            fig.set_figheight(30)
        else:
            fig.set_figwidth(8)
            fig.set_figheight(8)

        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title('Confusion matrix', fontsize=16)

        # ax.set_xticklabels((ax.get_xticks() +1).astype(int))
        # ax.set_yticklabels((ax.get_yticks() +1).astype(int))

        if target_names is not None:
            plt.xticks(np.arange(0, len(target_names)), target_names, rotation=90, fontsize=10)
            plt.yticks(np.arange(0, len(target_names)), target_names, fontsize=10)
        else:
            plt.xticks(np.arange(0, len(np.unique(target_true))), rotation=90, fontsize=10)
            plt.yticks(np.arange(0, len(np.unique(target_true))), fontsize=10)

        fmt = 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(np.arange(cnf_matrix.shape[0]), np.arange(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        if conf_matrix_name is not None:
            plt.savefig(conf_matrix_name)
        plt.show()
    
    #write top words and classes for each topic
    if top_tokens_text is not None:
        if top_tokens_class is not None:
            saved_top_tokens = top_tokens_text.last_tokens
            saved_top_classes = top_tokens_class.last_tokens
            for topic_name in saved_top_tokens.keys():
                print(topic_name)
                print(' '.join(saved_top_tokens[topic_name]))
                print(' '.join(saved_top_classes[topic_name]))
                print()
        else:
            saved_top_tokens = top_tokens_text.last_tokens
            for topic_name in saved_top_tokens.keys():
                print(topic_name)
                print(' '.join(saved_top_tokens[topic_name]))
                print()
        
    #shows doc_count documents for each topic(to track interpretability)
    if show_docs and theta_matrix is not None:
        documents = {}

        X_train = pd.read_csv(docs_data_path)
        for topic_name in theta_matrix.index:
            print(topic_name)
            for doc in theta_matrix.loc[topic_name, :].nlargest(n=doc_count).index:
                print(X_train[X_train['Unnamed: 0']==int(doc)]['Описание'].values[0])
                print()
    elif show_docs and theta_matrix is None:
        print('Pass theta_matrix to function to look at docs')
        
    #print and draw topic model scores
    if score_tracker is not None:
        plot_count=len(scores_names)
        
        if plot_count%2!=0:
            plt.plot(range(len(score_tracker[scores_names[0]].value)), score_tracker[scores_names[0]].value, 'g--')
            plt.title(scores_names[0])
            plt.show()
            scores_names=scores_names[1:]

        fig = plt.figure(figsize=(8*plot_count, 2*plot_count))
        for score_name, k in zip(scores_names, range(1, plot_count+1)):
            plt.subplot(plot_count//2, 2, k).plot(range(len(score_tracker[score_name].value)), score_tracker[score_name].value, 'g--')
            plt.title(score_name)
            print(score_name+': '+str(score_tracker[score_name].value[-1]))
        plt.show()

