"""
This module build the corresponding embedding based on each corresponding table
and then calculate the average user embedding as the final embedding for the recommendation.
Given the time limit, the algorithm and the hyperparameters have a lot of space for improvement
Author: Yifan Wang
Date: July 12, 2021
"""
import os
import pandas as pd
import numpy as np
from gensim.models import FastText

from utils import engine

# Folder to save models
MODEL_ARTIFACT_FOLDER = "artifacts"

# Load the Tables from the local sqlite db:
course_tags_df = pd.read_sql_table("course_tags", engine)
user_assessment_scores_df = pd.read_sql_table("user_assessment_scores", engine)
user_course_views_df = pd.read_sql_table("user_course_views", engine)
user_interests_df = pd.read_sql_table("user_interests", engine)


def clean_up_df(df: pd.DataFrame):
    """Simple cleaning to the dataframe

    Args:
        df (pd.DataFrame): [description]
    """
    df.dropna(inplace=True)


def get_avg_embedding(model, word_list: list, weight=None):
    """Get the average embedding given a list of word and the Fasttext model

    Args:
        model: Fasttext model
        word_list (list): a list of word
        weight: an array of weight to calcualte the weighted average

    Returns:
        A 1 dimensional numpy array which represents the average embedding
    """
    if isinstance(weight, type(None)):
        return np.mean([model.wv[word] for word in word_list], axis=0)

    all_embeddings = np.array([model.wv[word] for word in word_list])
    weight_array = np.array(weight).reshape(1, -1)  # 1XN
    avg_embedding = weight_array.dot(all_embeddings).ravel()
    return avg_embedding


def _normalize_weight_user_tag(x: str):
    """Calculate the normalized weight based on the assessment scores for the users

    Args:
        x (str):

    Returns:
        A numpy 1D array contains the normalized weight sum up to 1
    """
    string_list = x.split(" ")
    int_array = np.array(string_list, dtype=int)
    int_array = np.round(int_array / int_array.sum(), 3)
    return int_array


def model_train(corpus: list, epochs: int, **hpars):
    """Train a fasttext Gensim model

    Args:
        corpus (list): A corpus list, each doc contains a list of words
        epochs (int): Number of epochs

    Returns:
        The trained Fasttext model
    """
    model = FastText(**hpars)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model

def get_final_embedding(x: pd.Series):
    """The function to calculate the final embedding given the existing ones

    Args:
        x (pd.Series): 1 row of the embedding tables, contains each type of embedding in each column

    Returns:
        bytes: the final embedding in string type in order to load into sqlite
    """
    output: int = 0
    output += x["embedding2"] if not np.isnan(x["embedding2"]).any() else 0
    output += x["embedding3"] if not np.isnan(x["embedding3"]).any() else 0
    output += x["embedding4"] if not np.isnan(x["embedding4"]).any() else 0
    output /= 3.0  # average
    # save as string  in db, not directly readable, need to do np.fromstring(x) to make it back as array
    output: str = np.array_str(output)
    return output


if __name__ == "__main__":
    ###################################################
    # 1. Course Tags Table:
    # Convert the dataframe to the following format:
    # "tag1 tag2 tag3 tag4..." for each course

    # DUE TO TIME LIMIT, I CHOSE NOT TO USE THIS EMBEDDING FOR RECOMMENDATION
    ###################################################
    # clean_up_df(course_tags_df)
    # # Get the list of tags for each course in one row
    # unique_course_tags_df = course_tags_df.groupby("course_id").agg(
    #     {"course_tags": lambda x: " ".join(x)}
    # )
    # # Fasttext training:
    # corpus_list: list = unique_course_tags_df.course_tags.apply(
    #     lambda x: x.split(" ")).tolist()
    # model = model_train(
    #     corpus=corpus_list,
    #     epochs=20,
    #     vector_size=100,
    #     window=3,
    #     min_count=2
    # )
    # model.save(os.path.join(MODEL_ARTIFACT_FOLDER,
    #                         "course_tags_embedding.model"))
    # # Get avg embedding for each course:
    # avg_embeddings = [get_avg_embedding(model, doc) for doc in corpus_list]
    # course_embedding_using_tags = pd.DataFrame(
    #     {"embedding1": avg_embeddings},
    #     index=unique_course_tags_df.index)

    ###################################################
    # 2. user_assessment_scores_df
    # Convert the dataframe to the following format:
    #  "(tag1,score1) (tag2,score2)..." for each user
    ###################################################
    clean_up_df(user_assessment_scores_df)
    user_assessment_scores_df = user_assessment_scores_df[[
        "user_handle", "assessment_tag", "user_assessment_score"
    ]]  # ignore dates
    # transform the dataframe into the trainable format:
    user_assessment_scores_df.user_assessment_score = user_assessment_scores_df.user_assessment_score.astype(
        str)
    unique_user_tag_score_df = user_assessment_scores_df.groupby("user_handle").agg(
        {"assessment_tag": lambda x: " ".join(
            x), "user_assessment_score": lambda x: " ".join(x)}
    )
    unique_user_tag_score_df["tags_weight"] = unique_user_tag_score_df.user_assessment_score.apply(
        _normalize_weight_user_tag)
    unique_user_tag_score_df["assessment_tag"] = unique_user_tag_score_df.assessment_tag.apply(
        lambda x: x.split(" "))
    # Fasttext training:
    corpus_list: list = unique_user_tag_score_df["assessment_tag"].tolist()
    model = model_train(
        corpus=corpus_list,
        epochs=20,
        vector_size=100,
        window=3,
        min_count=2
    )
    model.save(os.path.join(MODEL_ARTIFACT_FOLDER,
               "user_assessment_scores_embedding.model"))
    # Get course embedding:
    avg_embeddings = [get_avg_embedding(
        model, word_list=corpus_list[idx],
        weight=unique_user_tag_score_df["tags_weight"].values[idx]) for idx in range(len(corpus_list))]
    # transform to array to calcualte
    user_embedding_using_assessment_tags = pd.DataFrame(
        {"embedding2": avg_embeddings},
        index=unique_user_tag_score_df.index
    )

    ###################################################
    # 3. user_course_views_df
    ###################################################
    # only need those 2 cols for embedding
    user_course_views_df = user_course_views_df[["user_handle", "course_id"]]
    clean_up_df(user_course_views_df)
    unique_user_course_view_df = user_course_views_df.groupby("user_handle").agg(
        {"course_id": lambda x: " ".join(x)}
    )
    # Fasttext training:
    corpus_list: list = unique_user_course_view_df.course_id.apply(
        lambda x: x.split(" ")).tolist()
    model = model_train(
        corpus=corpus_list,
        epochs=20,
        vector_size=100,
        window=3,
        min_count=2
    )
    model.save(os.path.join(MODEL_ARTIFACT_FOLDER,
                            "user_course_view_embedding.model"))
    # Get avg embedding for each course:
    avg_embeddings = [get_avg_embedding(model, doc) for doc in corpus_list]
    user_embedding_using_course_view = pd.DataFrame(
        {"embedding3": avg_embeddings},
        index=unique_user_course_view_df.index
    )

    ###################################################
    # 4. user_interests
    ###################################################
    user_interests_df = user_interests_df[["user_handle", "interest_tag"]]
    clean_up_df(user_interests_df)
    unique_user_interests_df = user_interests_df.groupby(
        "user_handle").agg({"interest_tag": lambda x: " ".join(x)})
    # Fasttext training:
    corpus_list: list = unique_user_interests_df.interest_tag.apply(
        lambda x: x.split(" ")).tolist()
    model = model_train(
        corpus=corpus_list,
        epochs=20,
        vector_size=100,
        window=3,
        min_count=2
    )
    model.save(os.path.join(MODEL_ARTIFACT_FOLDER,
               "user_interests_embedding.model"))
    # Get avg embedding for each course:
    avg_embeddings = [get_avg_embedding(model, doc) for doc in corpus_list]
    user_embedding_using_interest_tag = pd.DataFrame(
        {"embedding4": avg_embeddings},
        index=unique_user_interests_df.index
    )

    ###################################################
    # Aggregate the  embeddings into 1 for each user:
    #
    ###################################################

    # get all users: union of all user handles
    user_base1: set = set(user_embedding_using_assessment_tags.index)
    user_base2: set = set(user_embedding_using_course_view.index)
    user_base3: set = set(user_embedding_using_interest_tag.index)
    all_users = user_base1.union(user_base2).union(user_base3)
    # create the table stores the user embeddings
    all_user_embeddings = pd.DataFrame(index=all_users)

    all_user_embeddings = all_user_embeddings.join(
        user_embedding_using_assessment_tags, how="left")
    all_user_embeddings = all_user_embeddings.join(
        user_embedding_using_course_view, how="left")
    all_user_embeddings = all_user_embeddings.join(
        user_embedding_using_interest_tag, how="left")

    all_user_embeddings["embedding"] = all_user_embeddings.apply(
        get_final_embedding, axis=1)
    all_user_embeddings = all_user_embeddings[["embedding"]].reset_index()
    # to match the names in the sql table
    all_user_embeddings.columns = ["user_handle", "embedding_value"]

    all_user_embeddings.to_sql(
        name="user_embeddings",
        con=engine,
        if_exists="replace",
        index=False,
        index_label=["user_handle", "embedding_value"])
