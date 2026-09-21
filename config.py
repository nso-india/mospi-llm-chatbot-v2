import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path for static files
STATIC_DIRECTORY = os.path.join(BASE_DIR, "static")  # the folder to store static files
STATIC_ROUTE = "/static"  # URL path
STATIC_NAME = "static"  # name for FastAPI mount
