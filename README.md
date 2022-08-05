## Neural Attention Memory



### Running I-BERT

-----

I-BERT can be simply run from Bash. Below is the most core command line to run I-BERT. 

```bash
python3 AutoEncode.py --net ibert
```


### Options

-----

Our program supports multiple command-line options to provide a better user experience. The below table shows major options that can be simply appended when running the program.

| Options      | Default | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| --net        | tf      | Model for your task <br>ibert: I-BERT <br>xlnet: XLNet<br>lstm: LSTM seq2seq <br>tf: BERT <br>ibertpos: I-BERT with positional encoding <br>ibert2: I-BERT2 |
| --seq_type   | fib     | task for prediction <br>fib: addition task (NSP Dataset)<br>copy: copy task (NSP Dataset)<br>palin: reverse task (NSP Dataset)<br>ptbc: Penn Tree Bank Character<br>ptbw: Penn Tree Bank Word |
| --digits     | 12      | Max number of training digits <br>(Only supports for algorithmic tests) |
| --batch_size | 32      | Number of epochs                                             |
| --epochs     | 50      | Number of epochs                                             |
| --lr         | 3e-5    | Learning rate                                                |
| --log        | false   | Log training/validation results                              |
| --exp        | 0       | Assign log file identifier when --log is true                |

For example, if we want to run a `Penn Tree Bank Word` dataset with 100 epochs with I-BERT, we can try the following: 

```bash
python3 AutoEncode.py --net ibert --seq_type ptbw --epochs 100
```




### TODO

- CTC?
- LSAM in reduction?
- Add model save/load
- S2S version of fib/arith
- ID val / OOD Test

### AE tasks - LSAM?
- SCAN length
- NSP

## S2S tasks
- Multi 
- NSP S2S
