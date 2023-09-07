# How to integrate DGR into CUGR2?
1. run `python data_process_CUGR2.py`, which will generate Tree Data information (stored in the run path of CUGR2) for our model as a input.

2. run `python main.py`, which reads input and generates output paths for CUGR2.

3. in CUGR2, run the binary file with the generated output as the input argument for `--dgr <path>`