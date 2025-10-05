"""
This module is used to set the project root and load the .env file.
"""

import os
from dotenv import load_dotenv, find_dotenv


while os.getcwd().endswith("notebooks"):
    os.chdir("..")

print(os.getcwd())

# Look for the nearest .env starting from the CWD
env_path = find_dotenv(usecwd=True)
if env_path:
    print("found .env file")
    load_dotenv(env_path, override=False)
    print("Environment variables set:")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key = line.split("=", 1)[0]
                    value = os.environ.get(key, "")
                    print(f"  {key}")
