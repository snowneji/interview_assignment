# Project Introduction

    - Author: Yifan Wang
    - Recommendation Algorithm: 
    - Software Used: Python, Flask, Sqlalchemy(Sqlite)

This project constructs a database based on the provided sample customer tables,  builds a recommendation algorithm that recommends the similar users and wrap the algorithm into a consumable webservice.



# Usage

## 1. Create the Conda Environment:
```
conda env create -f env.yml
```

## 2. Create database and corresponding data models:

Those tables are created based on the provided user data: `course_tags`, `user_assessment_scores`, `user_course_views`, `user_interests`

The following 1 table is created to save the user embeddings which are calcualted in the next step: `user_embeddings`

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


