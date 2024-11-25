
# Dependencies (pip install):
+ torch 
+ torch_scatter 
+ numpy
+ tracemalloc


# pre-requisite
0. Install our revised CUGR2 at 

```
git clone https://github.com/wadmes/cu-gr-2
```

1. put benchmarks to `$cugr2_dir/benchmark`

# How to integrate DGR into CUGR2?
0. run revised CUGR2 to obatin some required data, e.g. corresponding FLUTE tree, and layout information. (see `CUGR2.sh` as an example to run CUGR2)

1. run `python data_process_CUGR2.py`, which will generate Tree Data information of every benchmark(stored in the run path of CUGR2) for our model as a input.

2. run `python main_stochastic.py`, which reads input and generates output paths for CUGR2. You may need to update hyper-parameters to change data path. See `DGR.sh` as an example.

3. in CUGR2, run the binary file with the generated output as the input argument for `--dgr <path>`.  See `ours.sh` as an example. The CUGR2 will not run its own DP-based parttern routing, rather, it will use DGR results as a helper to decide routing pattern.

# All-in-one script
After installing depencies and pre-requisite, you can change the path variables in `DGR.sh` and `CUGR2.sh`, and simply run the following to get the DGR and CUGR2 results directly.

```
sh experiments.sh
```


# Hyper-paramter search
We provide an example script `test.py` to do hyper-parameter search.

`ray.sh` shows how to call `test.py` under different benchmarks and gpu index.


