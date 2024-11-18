import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import tensorflow_recommenders as tfrs



class TwoTowerModel(tfrs.models.Model):
  """ Retrieval model encompassing users as query and items as candidates"""

  def __init__(self, user_model, item_model, product_features):
    super().__init__()
    self.user_model = user_model
    self.item_model = item_model
    self.product_features = product_features

    # Query tower
    self.query_model = models.Sequential()
    self.query_model.add(self.user_model)
    self.query_model.add(layers.Dense(64, activation="relu"))
    self.query_model.add(layers.Dense(32, activation="relu"))
    self.query_model.add(layers.Dense(32))

    # Candidate tower
    self.candidate_model = models.Sequential()
    self.candidate_model.add(self.item_model)
    self.candidate_model.add(layers.Dense(64, activation="relu"))
    self.candidate_model.add(layers.Dense(32, activation="relu"))
    self.candidate_model.add(layers.Dense(32))

    # Retrieval task for loss function
    metrics = tfrs.metrics.FactorizedTopK(
            candidates=product_features.batch(128).map(self.item_model)
    )
    self.task = tfrs.tasks.Retrieval(metrics=metrics)
  
  def compute_loss(self, features, training=False):
    # Passing the embeddings into the loss function
    user_embeddings = self.query_model({
                                    "userid": features["visitorid"],
                                    "action": features["event"],
                                    "Weight": features["Weight"],
                                    "time": features["timestamp"],
                                    "user_number_of_views": features["user_number_of_views"],
                                    "user_number_of_addtocart": features["user_number_of_addtocart"],
                                    "user_number_of_purchases": features["user_number_of_purchases"],
                                    "number_of_unique_items": features["number_of_unique_items"]
                                    })
    
    product_embeddings = self.candidate_model(
        {"itemid": features["itemid"], 
         "property1": features["property1"], 
         "property2": features["property2"], 
         "property3": features["property3"], 
         "property4": features["property4"], 
         "property5": features["property5"], 
         "property6": features["property6"], 
         "property7": features["property7"], 
         "property8": features["property8"], 
         "property9": features["property9"], 
         "property10": features["property10"], 
         "avilable": features["available"],
         "categoryid": features["categoryid"],
         "parent_level_1": features["parent_level_1"],
         "parent_level_2": features["parent_level_2"],
         "parent_level_3": features["parent_level_3"],
         "parent_level_4": features["parent_level_4"],
         "parent_level_5": features["parent_level_5"],
         "item_number_of_views": features["item_number_of_views"],
         "item_number_of_addtocart": features["item_number_of_addtocart"],
         "item_number_of_purchases": features["item_number_of_purchases"],
         "number_of_unique_visitors": features["number_of_unique_visitors"]

         })
    
    interaction_weights = tf.cast(features["event"], tf.float32)
        
    # Convert views to negative samples and normalize weights
    interaction_weights = tf.where(
        interaction_weights == 1,  # views
        -1.0,  # negative weight for views
        tf.where(
            interaction_weights == 2,  # add to cart
            2.0,  # positive weight for add to cart
            3.0   # highest weight for purchase (action_id = 3)
        )
    )

    # Calculate the loss via task for query and candidate embeddings
    return self.task(user_embeddings, product_embeddings)