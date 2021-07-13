import flask
import json

from utils import MODEL_ARTIFACT_FOLDER


app = flask.Flask(__name__)


# Load the model artifacts when creating the webapp:
AR


@app.route("/score",methods=["GET","POST"])
def score():
    """The scoring function to serve the recommendation algorithm in realtime

    Returns:
        [type]: [description]
    """
    if flask.request.method == 'GET':
        return "Give me a POST request to get recommendations"
    
    output = {"similarity":0.999}
    return json.dumps(output)


if __name__ == '__main__':
      app.run(host="0.0.0.0", port=80)