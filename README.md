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
