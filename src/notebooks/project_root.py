import os

while os.getcwd().endswith("notebooks"):
    os.chdir("..")

print(os.getcwd())