# Neural Attention Memory

This is the repository for Neural Attention Memory paper experiments.
Clone with `--recurse-submodules` to load the SCAN dataset.

## Requirements
- Python 3.8
- CUDA-capable GPU (Tested on RTX 4090 24GB. Reduce `--batch_size` if gpu memory is limited)
- PyTorch >= 1.7
- CUDA >= 10 (Install with PyTorch)
- Python libraries listed in requirements.txt

## Running Experiments

-----

`AutoEncode.py` is the code for running the experiments as below.  

`--log` will create a log file of the experiment.

```bash
python AutoEncode.py --net namtm --seq_type add --digits 10 --log
```

For 4-DYCK, run `python DYCK.py` to generate the data points first.
## Options

-----

Our program supports multiple command-line options to provide a better user experience. The below table shows major options that can be simply appended when running the program.

| Options      | Default | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| --net        | namtm      | Model to run <br>tf: Transformer <br>ut: Universal Transformer <br>dnc: Differentiable Neural Computer<br>lstm: LSTM w attention <br>stm: SAM Two-memory Model <br>namtm: NAM-TM <br>stack: Stack-RNN |
| --seq_type   | add     | task for prediction <br>add: addition task (NSP)<br>reverse: reverse task (NSP)<br>reduce: Sequence reduction task<br>dyck: 4-DYCK task |
| --digits     | 10      | Max number of training digits  | 
| --log        | false   | Log training/validation results                              |
| --exp        | 0       | Assign log file identifier when --log is true                |

See `Options.py` or `python AutoEncode.py --help` for more options.

## Copyright Notice
Some parts of this repository are from the following open-source projects.  
This repository follows the open-source policies of all of them.  
- DNC (`dnc/`): https://github.com/RobertCsordas/dnc
- Universal Transformer (`transformer_generalization/`): https://github.com/RobertCsordas/transformer_generalization
- LSTM seq2seq (`Models.py`): https://github.com/pytorch/fairseq
- Number Sequence Prediction dataset (`NSPDataset.py, AutoEncode.py`): https://github.com/hwnam831/numbersequenceprediction
- XLNet (`XLNet.py`): https://github.com/huggingface/transformers

