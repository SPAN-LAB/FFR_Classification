import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import torch
except Exception:
    pass

from src.GUI.gui import main

if __name__ == "__main__":
    main()
    #asjdkl