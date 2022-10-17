import os

# delete all the pychache files
os.system('find . -name "__pycache__" -exec rm -rf {} \;')
