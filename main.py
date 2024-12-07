import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from data_preprocessing import load_data
from model_recently_viewed_items import model_recently_viewed_items
from model_most_popular_items import model_most_popular_items
from model_collaborative_filtering import model_collaborative_filtering
from model_singular_value_decomposition import model_singular_value_decomposition

if __name__ == '__main__':
    interactions, interactions_test,_, item_popularity,_  = load_data()
    interactions_test = interactions_test[interactions_test['visitorid'].isin(interactions['visitorid'].unique())]
    print("\n Performance on recently viewed items")

    test_interactions = model_recently_viewed_items(interactions, interactions_test)
    
    usersPerItem = defaultdict(set)
    itemsPerUser = defaultdict(set)

    for idx, row in interactions.iterrows():
        user, item = row['visitorid'], row['itemid']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        
    print("\n Creating interaction matrix by taking counts of interactions")
    user_item = pd.DataFrame(interactions.groupby(['visitorid', 'itemid'])['timestamp'].count().reset_index())
    user_wise_df = user_item.pivot(index='visitorid', columns='itemid', values='timestamp').fillna(0)
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_wise_df.index)}
    index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(user_wise_df.columns)}
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items()}

    print("\nPerformance on most popular items")
    model_most_popular_items(interactions, item_popularity, test_interactions)
    model_collaborative_filtering(interactions, test_interactions, user_wise_df, user_id_to_index, index_to_user_id, itemsPerUser,
                             item_id_to_index, index_to_item_id)
    print("\nPerformance on SVD matrix factorization")
    model_singular_value_decomposition(interactions, test_interactions, user_id_to_index, item_id_to_index, index_to_item_id,
                                       itemsPerUser)

    print("\nCreating interaction matrix by taking weights of interactions")
    event_map={'view':1, 'addtocart':2, 'transaction':3}
    train_interactions = interactions[['visitorid','itemid','event']]
    train_interactions = train_interactions.drop_duplicates()
    train_interactions['weight'] = train_interactions['event'].apply(lambda x:event_map[x])
    train_interactions = train_interactions.sort_values(by=['visitorid','itemid','weight'], ascending=[True, True, False])
    train_interactions = train_interactions.drop_duplicates(subset=['visitorid','itemid'], keep='first')
    user_wise_df = train_interactions.pivot(index='visitorid', columns='itemid', values='weight').fillna(0)

    model_collaborative_filtering(interactions, test_interactions, user_wise_df, user_id_to_index, index_to_user_id, itemsPerUser,
                             item_id_to_index, index_to_item_id)
    print("\nPerformance on SVD matrix factorization")
    model_singular_value_decomposition(interactions, test_interactions, user_id_to_index, item_id_to_index, index_to_item_id,
                                       itemsPerUser)

    
