import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from evaluation_metrics import precision_at_k, recall_at_k
import heapq
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def model_collaborative_filtering(interactions, test_interactions, user_wise_df, user_id_to_index, index_to_user_id, itemsPerUser,
                             item_id_to_index, index_to_item_id):

    '''
        The model_collaborative_filtering function evaluates the user-user and item-item collaborative model by computing the precision
        and recall at k values.
    '''
    sparse_user_wise_df = sparse.csr_matrix(user_wise_df)
    similarities = cosine_similarity(sparse_user_wise_df, dense_output=False)

    print('\nUser User Collaborative Filtering:')
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
            user_similarities = similarities[specific_user_index].toarray().flatten()
            top_similar_indices = heapq.nlargest(10, range(len(user_similarities)), key=lambda i: user_similarities[i])
            similar_users = [index_to_user_id[idx] for idx in top_similar_indices][:10]

            # Vectorized prediction for unseen items
            predictions = user_wise_df.loc[similar_users].sum(axis=0)
            predictions = predictions[predictions != 0]
            unseen_items = list(set(predictions.index) - set(itemsPerUser[user]))
            predictions_unseen = predictions[unseen_items]

            # Select top-k items
            top_k_items = predictions.nlargest(k).index.tolist()
            top_k_items_unseen = predictions_unseen.nlargest(k).index.tolist()

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
            f'Recall@{k} on unseen': np.mean(recalls_unseen),
        })

    sparse_item_wise_df = sparse.csr_matrix(user_wise_df.T)
    item_similarities = cosine_similarity(sparse_item_wise_df, dense_output=False)

    print('\nItem Item Collaborative Filtering:')
    print("-----------------------------")
    for k in [1, 5, 10, 20, 50]:
        recalls = []
        precisions = []
        recalls_unseen = []
        precisions_unseen = []

        for user, actual_items, actual_items_unseen in zip(test_interactions['visitorid'],
                                                           test_interactions['relevant_items'],
                                                           test_interactions['relevant_items_unseen']):
            seen_items = itemsPerUser[user]

            # Generate predictions
            predictions = pd.Series(dtype=float)
            for item in seen_items:
                specific_item_index = item_id_to_index[item]
                item_similarities_row = item_similarities[specific_item_index].toarray().flatten()

                # Aggregate scores for items similar to the current item
                similar_item_indices = np.argsort(item_similarities_row)[-10:]  # Top 10 similar items
                for idx in similar_item_indices:
                    similar_item = index_to_item_id[idx]
                    if similar_item not in predictions:
                        predictions[similar_item] = 0
                    predictions[similar_item] += item_similarities_row[idx]

            # Exclude items the user has already interacted with for unseen predictions
            unseen_items = list(set(predictions.index) - set(seen_items))
            predictions_unseen = predictions.loc[unseen_items]

            # Select top-k items
            top_k_items = predictions.nlargest(k).index.tolist()
            top_k_items_unseen = predictions_unseen.nlargest(k).index.tolist()

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
            f'Recall@{k} on unseen': np.mean(recalls_unseen),
        })
