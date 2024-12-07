from evaluation_metrics import precision_at_k, recall_at_k
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import sparse

def model_singular_value_decomposition(interactions, test_interactions, user_id_to_index, item_id_to_index, index_to_item_id,
                                       itemsPerUser):
    '''
        The function model_singular_value_decomposition evaluates the SVD model by computing the precision and recall values
        for different k values.
    '''

    user_item = pd.DataFrame(interactions.groupby(['visitorid', 'itemid'])['timestamp'].count().reset_index())
    user_wise_df = user_item.pivot(index='visitorid', columns='itemid', values='timestamp').fillna(0)
    sparse_user_wise_df = sparse.csr_matrix(user_wise_df)
    U, s, Vt = svds(sparse_user_wise_df, k=50)
    sigma = np.diag(s)
    predicted_interactions = np.dot(np.dot(U, sigma), Vt)
    print("-----------------------------")
    for k in [1, 5, 10, 20, 50]:
        recalls = []
        precisions = []
        recalls_unseen = []
        precisions_unseen = []

        for user, actual_items, actual_items_unseen in zip(test_interactions['visitorid'],
                                                           test_interactions['relevant_items'],
                                                           test_interactions['relevant_items_unseen']):
            specific_user_index = user_id_to_index[user]

            # Vectorized prediction for unseen items
            predictions = predicted_interactions[specific_user_index]
            seen_items = [item_id_to_index[id] for id in itemsPerUser[user]]
            predictions_unseen = [v for i, v in enumerate(predictions) if i not in seen_items]
            predictions = np.argsort(predictions)[-k:][::-1]
            predictions_unseen = np.argsort(predictions_unseen)[-k:][::-1]

            # Select top-k items
            top_k_items = [index_to_item_id[pred] for pred in predictions]
            top_k_items_unseen = [index_to_item_id[pred] for pred in predictions_unseen]

            # Compute precision and recall
            recalls.append(recall_at_k(actual_items, top_k_items, k))
            precisions.append(precision_at_k(actual_items, top_k_items, k))
            if len(actual_items_unseen) != 0:
                recalls_unseen.append(recall_at_k(actual_items_unseen, top_k_items_unseen, k))
                precisions_unseen.append(precision_at_k(actual_items_unseen, top_k_items_unseen, k))

        print({
            f'Precision@{k}': np.mean(precisions),
            f'Recall@{k}': np.mean(recalls),
            f'Precision@{k} on unseen': np.mean(precisions_unseen),
            f'Recall@{k} on unseen': np.mean(recalls_unseen)})