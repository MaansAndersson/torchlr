# torchlr

Install

```bash
git clone git@github.com:MaansAndersson/torchlr.git
cd torchlr
python3 -m pip install .
```

It can be advantageous to install in a [`venv`](https://docs.python.org/3/library/venv.html).


## Examples

Run the `advection_diffusion` problem with `1024` unknowns in each direction and the `is` (index selection algorithm) `qdeim`.
```bash
python3 examples/advection_diffusion.py -n 1024 -is qdeim
```

For usage use `--help`. 
```bash
python3 examples/advection_diffusion.py --help 


usage: low-rank advection diffusion solver [-h] [-n PROBLEMSIZE] [-is INDEXSELECTION] [-t ENDTIME] [-pf PLOTFREQ] [-f FILENAME] [-s SEED]

```

## Using Accelerators

Each of the examples have the following lines in the begining.
```
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
```
We assume that if there is a `cuda` device available then we want it to be used.


## Clear cache 
At some point the cached functions might need to be removed
```
rm -rf /tmp/torchinductor_$(whoami)
```
