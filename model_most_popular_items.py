import numpy as np
import warnings
warnings.filterwarnings("ignore")
from evaluation_metrics import precision_at_k, recall_at_k

def model_most_popular_items(interactions, item_popularity, test_interactions):
    '''
        The model_most_popular_items function evaluates the Most Popular model by computing the precision and recall at k values.
    '''

    top_items = item_popularity.sort_values(by='number_of_views', ascending=False)
    top_popular_items = top_items['itemid'].head(50).tolist()
    train_interactions = interactions.groupby('visitorid')['itemid'].apply(set)
    test_interactions['items_seen'] = test_interactions['visitorid'].apply(
        lambda x: train_interactions[x] if x in train_interactions.index else {})
    test_interactions['relevant_items_unseen'] = test_interactions.apply(
        lambda row: row['relevant_items'] - row['items_seen'], axis=1)
    for k in [1, 5, 10, 20, 50]:
        test_interactions['precision'] = test_interactions['relevant_items'].apply(
            lambda x: precision_at_k(x, top_popular_items, k))
        test_interactions['recall'] = test_interactions['relevant_items'].apply(
            lambda x: recall_at_k(x, top_popular_items, k))
        print(f'Precision@{k}:', np.mean(test_interactions['precision']), f'Recall@{k}:',
              np.mean(test_interactions['recall']))
    print("\n")
    for k in [1, 5, 10, 20, 50]:
        test_interactions['precision'] = test_interactions.apply(
            lambda row: precision_at_k(row['relevant_items_unseen'], list(set(top_popular_items) - row['items_seen']),
                                       k), axis=1)
        test_interactions['recall'] = test_interactions.apply(
            lambda row: recall_at_k(row['relevant_items_unseen'], list(set(top_popular_items) - row['items_seen']), k),
            axis=1)
        print(f'Precision@{k} on unseen:', np.mean(test_interactions['precision']), f'Recall@{k} on unseen:',
              np.mean(test_interactions['recall']))