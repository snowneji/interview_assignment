"""
This module contructs the corresponding relational db with sqlite,
    and using sqlalchemy's  ORM functionality to build the data model
    and load in the data using pandas' connection to sqlalchemy
Author: Yifan Wang
Date: July 7, 2021

"""
import os

import pandas as pd
from sqlalchemy import Column, Integer, Float, Date, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils import engine, DATA_FOLDER

Base = declarative_base()


class CourseTags(Base):
    """
    Course tag table based on the course_tags.csv file.
    Use course_id + course_tags as the composite key
    """
    __tablename__ = 'course_tags'
    __table_args__ = {'sqlite_autoincrement': True}
    # Composite key:
    course_id = Column(TEXT,primary_key=True, nullable=False) 
    course_tags = Column(TEXT,primary_key=True, nullable=True)


class UserAssessmentScores(Base):
    """
    User assessment scores table based on the user_assessment_scores.csv file.
    Use user_handle + assessment_tag as the composite key
    """
    __tablename__ = 'user_assessment_scores'
    __table_args__ = {'sqlite_autoincrement': True}
    # Composite key:
    user_handle = Column(Integer, primary_key=True, nullable=False) 
    assessment_tag = Column(TEXT, primary_key=True, nullable=False)
    user_assessment_date = Column(Date, nullable=False)
    user_assessment_score = Column(Integer, nullable=False)
   
class UserCourseViews(Base):
    
    __tablename__ = 'user_course_views'
    __table_args__ = {'sqlite_autoincrement': True}
    
    user_handle = Column(Integer, primary_key=True, nullable=False)
    view_date = Column(Date, primary_key=True, nullable=False)
    course_id = Column(TEXT, primary_key=True, nullable=False)
    author_handle = Column(Integer, nullable=False) 
    level = Column(TEXT, nullable=False) 
    view_time_seconds = Column(Integer, nullable=False) 
    
class UserInterests(Base):
    
    __tablename__ = 'user_interests'
    __table_args__ = {'sqlite_autoincrement': True}
    
    user_handle = Column(Integer, primary_key=True, nullable=False)
    interest_tag = Column(TEXT, primary_key=True, nullable=False)
    date_followed = Column(Date, nullable=False)


class UserEmbeddings(Base):
    """This table saves the pre-computed user embeddings
    Save the embedding vectors as string
    """
    __tablename__ = 'user_embeddings'
    __table_args__ = {'sqlite_autoincrement': True}
    
    user_handle = Column(Integer, primary_key=True, nullable=False)
    embedding_value = Column(TEXT, nullable=False)


if __name__ == "__main__":
    
    #Create the database
    Base.metadata.create_all(engine)
    
    # Load Data:
    # course tag:
    course_tags_df:pd.DataFrame = pd.read_csv(os.path.join(DATA_FOLDER,"course_tags.csv"))
    course_tags_df.to_sql(
        name = "course_tags",
        con = engine,
        if_exists = "replace",
        index = False,
        index_label = ["course_id","course_tags"])
    
    # user assessment scores:
    user_assessment_scores_df:pd.DataFrame = pd.read_csv(os.path.join(DATA_FOLDER,"user_assessment_scores.csv"))
    user_assessment_scores_df.to_sql(
        name = "user_assessment_scores",
        con = engine,
        if_exists = "replace",
        index = False,
        index_label = ["user_handle","assessment_tags"])

    # user course views:
    user_course_views_df:pd.DataFrame = pd.read_csv(os.path.join(DATA_FOLDER,"user_course_views.csv"))
    user_course_views_df.to_sql(
        name = "user_course_views",
        con = engine,
        if_exists = "replace",
        index = False,
        index_label = ["user_handle","view_date","course_id"])
    
    # user interests:
    user_interests_df:pd.DataFrame = pd.read_csv(os.path.join(DATA_FOLDER,"user_interests.csv"))
    user_interests_df.to_sql(
        name = "user_interests",
        con = engine,
        if_exists = "replace",
        index = False,
        index_label = ["user_handle","interest_tag"])
    

