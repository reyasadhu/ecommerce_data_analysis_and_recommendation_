import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np


class UserModel(tf.keras.Model):
  """ Features Model representing the user and its features """
  def __init__(self, dataset, embedding_dim=64):
    super().__init__()

    self.dataset = dataset

    #Preprocessing

    self.user_vocab = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.user_vocab.adapt(self.dateset.map(lambda x: x["visitorid"]))

    # User Interactions
    self.user_actions = layers.experimental.preprocessing.StringLookup(mask_token=None)
    self.user_actions.adapt(self.dateset.map(lambda x: x["event"]))

    # User Interaction Weights
    self.user_action_weights = layers.experimental.preprocessing.IntegerLookup(mask_token=None)
    self.user_action_weights.adapt(self.dateset.map(lambda x: x["Weight"]))

    # User Continuous Timestamp
    self.user_time_norm = layers.experimental.preprocessing.Normalization()
    self.timestamps = np.concatenate(list(self.dateset.map(lambda x: x["timestamp"]).batch(1000)))
    self.user_time_norm.adapt(self.timestamps)

    # User Discrete Timestamp
    days = 120  # total days in interactions
    self.timestamps_disc = np.linspace(self.timestamps.min(), self.timestamps.max(), num=days)
    self.user_time_disc = layers.experimental.preprocessing.Discretization(self.timestamps_disc.tolist())

    #User popularity data
    self.user_views_norm = layers.experimental.preprocessing.Normalization()
    self.number_of_views = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_views"]).batch(1000)))
    self.user_views_norm.adapt(self.number_of_views)

    self.user_atc_norm = layers.experimental.preprocessing.Normalization()
    self.number_of_addtocart = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_addtocart"]).batch(1000)))
    self.user_atc_norm.adapt(self.number_of_addtocart)

    self.user_purchases_norm = layers.experimental.preprocessing.Normalization()
    self.number_of_purchases = np.concatenate(list(self.dataset.map(lambda x: x["user_number_of_purchases"]).batch(1000)))
    self.user_purchases_norm.adapt(self.number_of_purchases)

    self.user_n_items_norm = layers.experimental.preprocessing.Normalization()
    self.number_of_items = np.concatenate(list(self.dataset.map(lambda x: x["number_of_unique_items"]).batch(1000)))
    self.user_n_items_norm.adapt(self.number_of_items)


    # Dimensions for embedding into high dimensional vectors
    self.embedding_dim = embedding_dim

    #Embedding + norm layers
    self.user_embedding = models.Sequential()
    self.user_embedding.add(self.user_vocab)
    self.user_embedding.add(layers.Embedding(self.user_vocab.vocabulary_size(), self.embedding_dim))


    # User Interactions via category encoding
    self.user_action_encoding = models.Sequential()
    self.user_action_encoding.add(self.user_actions)
    self.user_action_encoding.add(
        layers.Embedding(
            self.user_actions.vocabulary_size(),
            self.embedding_dim))
    
    # Weights of user interactions
    self.user_action_weights = models.Sequential()
    self.user_action_weights.add(self.user_action_weights)
    self.user_action_weights.add(
        layers.Embedding(
            self.user_action_weights.vocabulary_size(), self.embedding_dim)
    )

    # Time Continuous
    self.time_continuous = models.Sequential()
    self.time_continuous.add(self.user_time_norm)

    # Time Discrete
    self.time_discrete = models.Sequential()
    self.time_discrete.add(self.user_time_disc)
    self.time_discrete.add(layers.Embedding(len(self.timestamps_disc) + 1, self.embedding_dim))

    



  # Forward pass
  def call(self, inputs):
    """ 
    Forward Pass with user features 
    """
    return tf.concat([
                      self.user_embedding(inputs["visitorid"]),
                      self.user_action_encoding(inputs["event"]),
                      self.user_action_weight_encoding(inputs["Weight"]),
                      self.time_continious(inputs["timestamp"]),
                      self.time_discrete(inputs["timestamp"])
    ], axis=1)
