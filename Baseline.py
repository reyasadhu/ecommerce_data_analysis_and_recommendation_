import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.sparse.linalg import svds
import heapq

def load_data():
    '''
        load_data function is used to load the preprocessed data
    '''
    interactions = pd.read_csv("processed_data/interactions_train.csv")
    interactions_test = pd.read_csv("processed_data/interactions_valid.csv")
    item_popularity = pd.read_csv("processed_data/popularity_item.csv")
    user_popularity = pd.read_csv("processed_data/popularity_user.csv")
    product_features = pd.read_csv("processed_data/product_features.csv")

    usersPerItem = defaultdict(set)
    itemsPerUser = defaultdict(set)

    for idx, row in interactions.iterrows():
        user, item = row['visitorid'], row['itemid']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
    user_item = pd.DataFrame(interactions.groupby(['visitorid', 'itemid'])['timestamp'].count().reset_index())
    user_wise_df = user_item.pivot(index='visitorid', columns='itemid', values='timestamp').fillna(0)
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_wise_df.index)}
    index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(user_wise_df.columns)}
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items()}

    return (interactions, item_popularity, itemsPerUser, user_wise_df, user_id_to_index, index_to_user_id, item_id_to_index,
            index_to_item_id)

def precision_at_k(relevant, predicted, k=10):
    '''
        The precision_at_k function computes the precision at k value.
    '''
    if isinstance(predicted, set):
        predicted = list(predicted)
    predicted = predicted[:k]
    hits = len(set(predicted) & relevant)
    return hits / k

def recall_at_k(relevant, predicted, k=10):
    '''
        The recall_at_k function computes the recall at k value.
    '''
    if isinstance(predicted, set):
        predicted = list(predicted)
    predicted = predicted[:k]
    hits = len(set(predicted) & relevant)
    if len(relevant)==0:
        return 0
    return hits/len(relevant)

def model_recently_viewed_items(interactions, interactions_test):
    '''
        The model_recently_viewed_items function evaluates the RVI model by computing the precision and recall at k values.
    '''

    recently_interacted = interactions.sort_values(by=['visitorid', 'timestamp'], ascending=[True, False])
    top_recently_interacted = recently_interacted.groupby('visitorid')['itemid'].apply(set).reset_index()
    all_interactions = pd.concat([interactions, interactions_test])
    relevant_items = all_interactions.groupby('visitorid')['itemid'].apply(set).reset_index()
    relevant_items.columns = ['visitorid', 'relevant_items']
    test_interactions = interactions_test.groupby('visitorid')['itemid'].apply(set).reset_index()
    test_interactions.columns = ['visitorid', 'test_items']
    test_interactions = test_interactions.merge(relevant_items, on='visitorid', how='left')
    test_interactions.drop(columns=['test_items'], inplace=True)
    test_interactions = test_interactions.merge(top_recently_interacted, on='visitorid', how='left')
    test_interactions.rename(columns={'itemid': 'RVI_predicted'}, inplace=True)
    for k in [1, 5, 10, 20, 50]:
        test_interactions['precision'] = test_interactions.apply(
            lambda row: precision_at_k(row['relevant_items'], row['RVI_predicted'], k), axis=1)
        test_interactions['recall'] = test_interactions.apply(
            lambda row: recall_at_k(row['relevant_items'], row['RVI_predicted'], k), axis=1)
        print(f'Precision@{k}:', np.mean(test_interactions['precision']), f'Recall@{k}:',
              np.mean(test_interactions['recall']))
    return test_interactions

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

    for k in [1, 5, 10, 20, 50]:
        test_interactions['precision'] = test_interactions.apply(
            lambda row: precision_at_k(row['relevant_items_unseen'], list(set(top_popular_items) - row['items_seen']),
                                       k), axis=1)
        test_interactions['recall'] = test_interactions.apply(
            lambda row: recall_at_k(row['relevant_items_unseen'], list(set(top_popular_items) - row['items_seen']), k),
            axis=1)
        print(f'Precision@{k} on unseen:', np.mean(test_interactions['precision']), f'Recall@{k} on unseen:',
              np.mean(test_interactions['recall']))


def model_user_collaborative(interactions, test_interactions, user_wise_df, user_id_to_index, index_to_user_id, itemsPerUser,
                             item_id_to_index, index_to_item_id):

    '''
        The model_user_collaborative function evaluates the user-user and item-item collaborative model by computing the precision
        and recall at k values.
    '''
    sparse_user_wise_df = sparse.csr_matrix(user_wise_df)
    similarities = cosine_similarity(sparse_user_wise_df, dense_output=False)

    print('User User Similarities:')
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

    print('Item Item Similarities:')
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

def model_singular_value_decomposition(interactions, test_interactions, user_id_to_index, item_id_to_index, index_to_item_id):
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

