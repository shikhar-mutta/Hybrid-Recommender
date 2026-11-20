import nbformat, sys
from nbclient import NotebookClient
nb = nbformat.read('hybrid_recommender.ipynb', 4)
try:
    NotebookClient(nb, timeout=3000, kernel_name='python3').execute()
    print("EXECUTED OK")
except Exception as e:
    print("FAILED:", str(e)[-1500:])
finally:
    nbformat.write(nb, 'hybrid_recommender.ipynb')
