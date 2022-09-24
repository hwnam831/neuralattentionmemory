python AutoEncode.py --epochs 100 --net $1 --seq_type palin > aelog/$1_palin_$2.log
python AutoEncode.py --epochs 150 --net $1 --seq_type fib > aelog/$1_fib_$2.log
python AutoEncode.py --epochs 150 --net $1 --seq_type scan > aelog/$1_scan_$2.log
python AutoEncode.py --epochs 100 --net $1 --seq_type reduce > aelog/$1_reduce_$2.log