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

- For `course_tags`, we train an embedding for each tag based on how they appear together in each course, and use the average tags embedding to define each course in the vector space.

- For `user_assessment_scores`, we train an embedding for each assessment tag based on which assessment tags each user takes together, and then use the weighted average (weighted by the assessment scores) assessment tags embedding to define each user in the vector space.

- For `user_course_views`, we train an embedding for each course based on how each user takes them together, and use the average course embedding to define each user in the vector space.

- For `user_interests`, we train an embedding for each interest tag based on how each user have them together, and use the average interest tag embedding to define each user in the vector space.

```
python step2_build_model.py
```
