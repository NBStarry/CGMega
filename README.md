# CGMega
CGMega is a graph attention-based deep learning framework for cancer gene module dissection. CGMega leverages a combination of multi-omics data cross genome, epigenome, proteome and especially three-dimension (3D) genome levels.

For more information, please read our paper: [CGMega: dissecting cancer gene module with explainable graph attention network]().

## Documentation
CGMega documentation is available through [Documentation](https://sunyolo.github.io/CGMega.github.io/).

## Questions and Code Issues
If you are having problems with our work, please use the [Github issue page](https://github.com/NBStarry/CGMega/issues).

## Update Log
See [Documentation-Updates]()

2023.07.03
Complete the README.md content
Add neoloopfinder processing script batch_neoloop.sh
cpu option added to training device

## Environment
conda create -n cgmega python=3.8
source activate cgmega
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_geometric
pip install pandas transformers wandb