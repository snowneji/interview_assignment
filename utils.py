from sqlalchemy import create_engine

MODEL_ARTIFACT_FOLDER = "artifacts"
DATA_FOLDER = "data_files_ml_engineer"

engine = create_engine("sqlite:///assignment_db.db")