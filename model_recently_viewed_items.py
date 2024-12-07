import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from evaluation_metrics import precision_at_k, recall_at_k

def model_recently_viewed_items(interactions, interactions_test):
    '''
        The model_recently_viewed_items function evaluates the RVI model by computing the precision and recall at k values.
    '''

    recently_interacted = interactions.sort_values(by=['visitorid', 'timestamp'], ascending=[True, False])
    top_recently_interacted = (
        recently_interacted.groupby('visitorid')['itemid']
        .apply(lambda x: x.drop_duplicates().tolist())
        .reset_index()
    )
    all_interactions = pd.concat([interactions, interactions_test])
    relevant_items = all_interactions.groupby('visitorid')['itemid'].apply(set).reset_index()
    relevant_items.columns = ['visitorid', 'relevant_items']
    test_interactions = interactions_test.groupby('visitorid')['itemid'].apply(set).reset_index()
    test_interactions.columns = ['visitorid', 'test_items']
    test_interactions = test_interactions.merge(relevant_items, on='visitorid', how='left')
    test_interactions.drop(columns=['test_items'], inplace=True)
    test_interactions = test_interactions.merge(top_recently_interacted, on='visitorid', how='left')
    test_interactions.rename(columns={'itemid': 'RVI_predicted'}, inplace=True)
    print("-----------------------------")
    for k in [1, 5, 10, 20, 50]:
        test_interactions['precision'] = test_interactions.apply(
            lambda row: precision_at_k(row['relevant_items'], row['RVI_predicted'], k), axis=1)
        test_interactions['recall'] = test_interactions.apply(
            lambda row: recall_at_k(row['relevant_items'], row['RVI_predicted'], k), axis=1)
        print(f'Precision@{k}:', np.mean(test_interactions['precision']), f'Recall@{k}:',
              np.mean(test_interactions['recall']))
    return test_interactions