# Some instructions

To analyze the effect of subaverage size, data amount, etc. on accuarcy, simply create a `.py` file in this directory, copy the contents of `example.py` into it, and **execute the code from the root directory**. 

## Detailed walkthrough

Step 1: Configure `SUBJECT_FILEPATHS`, found in `/src/analysis/config.py`, with the paths to the subject files.

```python
SUBJECT_FILEPATHS = ["Martian001.mat", "Martian051.mat"]
```

Step 2: Create your file in `src/server_analysis` and import the function(s). I'll call it `subaverage_ffnn.py`.

```python
from src.analysis import subaverage_size
from src.analysis import data_amount
```

Step 3: Call the `analyze` function of the appropriate module. 

```python
subaverage_size.analyze("FFNN")
```

Step 4: Run the function from the **ROOT DIRECTORY**. (Don't include the `.py` extension.)

```bash
python -m server_analysis.subaverage_ffnn
```
