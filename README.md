# Project Introduction

    - Author: Yifan Wang
    - Recommendation Algorithm: 
    - Software Used: Python, Flask, Sqlalchemy(Sqlite)

This project constructs a database based on the provided sample customer tables,  builds a recommendation algorithm that recommends the similar users and wrap the algorithm into a consumable webservice.


# Special Note about the Modification:

In order to better demonstrate the model output, instead of letting the API take the `user_handle` as the input, I intentionally changed it to something like the following, so that we can better play with the input and evaluate the model output.

```
{
    "top_n":2, 
    "assessment": [["css", 90], ["python", 150]], 
    "course_view": ["data-science-big-picture", "python-beyond-basics"], 
    "user_interests": ["python", "data-analysis"]}

```

# Usage

## 0. Create an `artifacts` folder to save the later models, and make sure the data files are in `data_files_ml_engineer` folder:

```
mkdir artifacts
```


## 1. Create the Conda Environment and activate it:
```
conda env create -f env.yml
```

```
conda activate pluralsight_env
```

## 2. Create database and corresponding data models:

Those tables are created based on the provided user data: `course_tags`, `user_assessment_scores`, `user_course_views`, `user_interests`

The following 2 tables are created to save the user embeddings which are calcualted in the next step and to display user info: `user_embeddings`, `display_table`

```
python step1_create_db.py
```

## 3. Modeling:

The modeling idea is to training a Fasttext embedding for each table based on the relationship.

- For `course_tags`, we train an embedding for each tag based on how they appear together in each course, and use the average tags embedding to define each course in the vector space. **(I ended up didn't use this one given the time limit)**

- For `user_assessment_scores`, we train an embedding for each assessment tag based on which assessment tags each user takes together, and then use the weighted average (weighted by the assessment scores) assessment tags embedding to define each user in the vector space.

- For `user_course_views`, we train an embedding for each course based on how each user takes them together, and use the average course embedding to define each user in the vector space.

- For `user_interests`, we train an embedding for each interest tag based on how each user have them together, and use the average interest tag embedding to define each user in the vector space.

- The final user embedding is calcualted by the average of `user_assessment_scores` embedding, `user_course_views` embedding and `user_interests` embedding.

- It might take 2-3 minutes to finish the training.

To train the models, just run the following command:

```
python step2_build_model.py
```


## 4. Model Serving:

- Start the locally hosted Flask app:

```
sh step3_serve_model.sh
```

- Then the user can send a Curl `POST` request to the app to get the most similar user recommendation based on the embedding model, the following is an example:

```
curl --header "Content-Type: application/json" --request POST --data '{"top_n":2, "assessment": [["css", 90], ["python", 150]], "course_view": ["data-science-big-picture", "python-beyond-basics"], "user_interests": ["python", "data-analysis"]}'  http://localhost:80/score
```

Input field explanations: (Note: since the trained embeddings are Fasttext model, typos can be tolerated)

-- `top_n`: the top n users to recommend

-- `assessment`: a list of list, each inner list contains the assessment tag and the score, the score will be used as the weight when averaging the embedding vectors

-- `course_view`: a list of courses viewed by the user

-- `user_interests`: a list of interest tags for the user


The output will be in the following format:

```
{user_handle:{"similarity": 0.9, "assessment_tag": "tag1 tag2 tag3", "course_id": "course1 course2 course3", "interest_tag": "tag1 tag2 tag3"}
```

Here is the real output from the input example above:

```
{"738": {"similarity": 0.8897897005081177, "assessment_tag": "nan", "course_id": null, "interest_tag": "data-analysis python"}, "1026": {"similarity": 0.8597033023834229, "assessment_tag": "python", "course_id": "python-beyond-basics python-getting-started python-natural-language-processing tensorflow-getting-started bitcoin-decentralized-technology hive-complex-analytical-queries python-getting-started tree-based-models-classification angularjs-fundamentals angularjs-fundamentals blockchain-fundamentals python-fundamentals python-getting-started internet-of-things-cyber-security blockchain-fundamentals python-getting-started python-getting-started blockchain-fundamentals python-getting-started blockchain-fundamentals blockchain-fundamentals python-getting-started blockchain-fundamentals python-getting-started blockchain-fundamentals python-getting-started blockchain-fundamentals python-getting-started blockchain-fundamentals blockchain-fundamentals scala-thinking-functionally blockchain-fundamentals scala-thinking-functionally blockchain-fundamentals python-getting-started scala-thinking-functionally blockchain-fundamentals python-getting-started scala-thinking-functionally scala-thinking-functionally scala-thinking-functionally scala-thinking-functionally blockchain-fundamentals python-getting-started scala-thinking-functionally scala-thinking-functionally scala-thinking-functionally blockchain-fundamentals python-getting-started scala-thinking-functionally blockchain-fundamentals", "interest_tag": "python data-analysis python data-analysis python data-analysis"}}
```


# Write-up to answer the questions for the assignment

1. The similarity calculation is essentially based on user word embeddings and the cosine similary. The key here is how the embeddings were calculated. 3 user embedding models (specifically Fasttext) were calculated based on the following tables:

- For `user_assessment_scores`, we train an embedding for each assessment tag based on which assessment tags appear in the same context of each user, and then use the weighted average (weighted by the assessment scores) assessment tags embedding to get the user embedding.

- For `user_course_views`, we train an embedding for each course based on how each user takes them together, and use the average course embedding to define each user in the vector space.

- For `user_interests`, we train an embedding for each interest tag based on how each user have them together, and use the average interest tag embedding to define each user in the vector space.

After we calculated the 3 embedding models during training phase, during the scoring phase the deployed model scoring API will based on the new user assessment result, course view records and interests to find 3 embedding vectors for the user, and then the average will be calculated as the final user embedding. the final user embedding will be searched against all the pre-computed existing user embeddings and recommend the most similar users.



2. Given the time limit, the implementation is a barely working application, when we have larger scale of data and to serve the model in production environment:

- I'll add some type of message queue between the API and the database to buffer some traffic for the database

- I'll add more components to the app, things such as customized http code, input/output json validation module, monitoring, logging, etc

- I'll consider to move some of the sql queries to stored procedures so that is more elegant and easier to maintain

- In this case the user embedding table is preloaded, but if in actual production scenarios, we probably also need a thread do some periodical pulling for the table

- If we have some more network bound or I/O bound tasks, I'll probably optimize the code with multi-threading or async approach.

- Consider deploy and manage the app using Kubernetes which can auto-scale our applications


3. I didn't use the `course_tags` table due to the time limit,  otherwise we can definitely leverage the information contained in that table. Besides, instead of building 3 separate embeddings and calculate the average, which will lose some information, we probably can build some end2end contextual embedding and neural based model to do the recommendation.