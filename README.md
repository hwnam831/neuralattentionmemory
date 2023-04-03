# Neural Attention Memory

This is the repository for Neural Attention Memory paper experiments.
Clone with `--recurse-submodules` to load the SCAN dataset.

## Requirements
- CUDA-capable GPU (Tested on RTX 3080 10GB)
- PyTorch >= 1.7
- CUDA >= 10 (Install with PyTorch)

## Running Experiments

-----

`AutoEncode.py` is the code for running the experiments as below.  

`--log true` will create a log file of the experiment.

```bash
python3 AutoEncode.py --net namtm --seq_type fib --digits 10 --log true
```

## Options

-----

Our program supports multiple command-line options to provide a better user experience. The below table shows major options that can be simply appended when running the program.

| Options      | Default | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| --net        | namtm      | Model to run <br>tf: Bert-like Transformer <br>ut: Universal Transformer <br>dnc: Differentiable Neural Computer<br>xlnet: XLNet<br>lstm: LSTM w attention <br>lsam: LSAM <br>namtm: NAM-TM <br>nojump: NAM-TM w.o. jmp transition |
| --seq_type   | fib     | task for prediction <br>fib: addition task (NSP)<br>copy: copy task (NSP)<br>palin: reverse task (NSP)<br>reduce: Sequence reduction task<br>scan: SCAN task |
| --digits     | 10      | Max number of training digits  | 
| --log        | false   | Log training/validation results                              |
| --exp        | 0       | Assign log file identifier when --log is true                |

See `Options.py` or `python3 AutoEncode.py --help` for more options.

## Copyright Notice
Some parts of this repository are from the following open-source projects.  
This repository follows the open-source policies of all of them.  
- DNC (`dnc/`): https://github.com/ixaxaar/pytorch-dnc
- Universal Transformer (`models/`): https://github.com/andreamad8/Universal-Transformer-Pytorch
- LSTM seq2seq (`Models.py`): https://github.com/pytorch/fairseq
- Number Sequence Prediction dataset (`NSPDataset.py, AutoEncode.py`): https://github.com/hwnam831/numbersequenceprediction
- SCAN dataset (`SCAN/`): https://github.com/brendenlake/SCAN
- XLNet (`XLNet.py`): https://github.com/huggingface/transformers

## TODO
- DNC-MDS (Done) - 55%?
- Change to Longint
- Ablation (No L/R, R/W prob, No erase)
- UT+Rel
- STM
- Priority Sort
- NFAR
- Associative Recall
- RAR
- babi 1k?
- TSP?