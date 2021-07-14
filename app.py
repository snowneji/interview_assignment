import json
import os

import flask
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.sql import text

from utils import MODEL_ARTIFACT_FOLDER, engine

app = flask.Flask(__name__)

# Load the model artifacts when creating the webapp:
MODELS = {}
MODELS["user_assessment_scores_embedding"] = FastText.load(
    os.path.join(MODEL_ARTIFACT_FOLDER, "user_assessment_scores_embedding.model"))
MODELS["user_course_view_embedding"] = FastText.load(
    os.path.join(MODEL_ARTIFACT_FOLDER, "user_course_view_embedding.model"))
MODELS["user_interests_embedding"] = FastText.load(
    os.path.join(MODEL_ARTIFACT_FOLDER, "user_interests_embedding.model"))

# Pre-load embedding table: ( need different mechanism if the table is dynamically updated)
user_embedding_df:pd.DataFrame = pd.read_sql_table("user_embeddings", engine)
user_embedding_df.embedding_value = user_embedding_df.embedding_value.apply(lambda x: np.fromstring(x,dtype=np.float32))
user_embedding_array = np.array(user_embedding_df.embedding_value.tolist())


@app.route("/score", methods=["GET", "POST"])
def score():
    """The scoring function to serve the recommendation algorithm in realtime

    Returns:
        [type]: [description]
    """
    if flask.request.method == 'GET':
        return "Give me a POST request to get recommendations"

    # Get input data:
    input_dict = flask.request.json
    # Json Validation:
    # (didn't have time for this, intended to do json validation and return 406 if schema doesn't match)
    # Top n:
    top_n:int = input_dict['top_n']
    # Encode to user embedding
    query_user_embedding_vector:np.ndarray = encode(input_dict)
    # Calculate Cosine Similarity:
    query_user_embedding_vector = query_user_embedding_vector.reshape(1,-1)
    cosine_sims = cosine_similarity(query_user_embedding_vector,user_embedding_array).ravel()
    # Recommend the top N most similar user
    idx:list = (-cosine_sims).argsort()[:top_n].tolist()
    chosen_cosine_sims:list = cosine_sims[idx].tolist()
    chosen_user_handle:list = user_embedding_df.user_handle.iloc[idx].tolist()
    
    # Get User info
    connection = engine.connect()
    cmd = f'select * from display_table  where user_handle in {tuple(chosen_user_handle)};'
    result = connection.execute(text(cmd))
    user_info = result.fetchall()
    connection.close()

    # Construct output:
    output_dict = {
        chosen_user_handle[idx]:{
            "similarity":chosen_cosine_sims[idx],
            "assessment_tag":user_info[idx][1],
            "course_id":user_info[idx][2],
            "interest_tag":user_info[idx][3],
            }  for idx in range(len(chosen_user_handle))}
    
    return json.dumps(output_dict)



def encode(input_dict: dict):
    """Encode the API input to the user embedding vector, based on the average of the 3 trained embeddings

    Args:
        input_dict (dict): API input

    Returns:
        [np.ndarray]: final user embedding output
    """
    # Calculation
    n_embeddings: int = 0
    # Course View:
    if len(input_dict['course_view']) > 0:
        course_view_embedding = np.mean(
            [MODELS["user_course_view_embedding"].wv[_word] for _word in input_dict['course_view']], axis=0)
        n_embeddings += 1
    else:
        course_view_embedding = np.zeros(100,)
    # User Interests:
    if len(input_dict['user_interests']) > 0:
        user_interests_embedding = np.mean(
            [MODELS["user_interests_embedding"].wv[_word] for _word in input_dict['user_interests']], axis=0)
        n_embeddings += 1
    else:
        user_interests_embedding = np.zeros(100,)

    # Assessment:
    if len(input_dict['assessment']) > 0:
        # weighted average:
        assessment_embedding = np.array(
            [MODELS["user_assessment_scores_embedding"].wv[_word[0]] for _word in input_dict['assessment']])
        weights = np.array([_word[1]
                           for _word in input_dict['assessment']]).reshape(1, -1)
        assessment_embedding = weights.dot(assessment_embedding)
        n_embeddings += 1
    else:
        user_interests_embedding = np.zeros(100,)

    all_avg_embedding = (course_view_embedding +
                         user_interests_embedding + user_interests_embedding)/n_embeddings
    return all_avg_embedding


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
