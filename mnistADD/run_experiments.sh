#!/bin/bash
for i in 1 2 3 4 5
do
    python stable_product.py --csv-path "results/1500/stable_product_p2_$i.csv" --n-examples-train 1500 --p 2
    python stable_product.py --csv-path "results/1500/stable_product_p6_$i.csv" --n-examples-train 1500 --p 6
    python prod_rl.py --csv-path "results/1500/prod_rl_$i.csv" --n-examples-train 1500
    python log_ltn.py --csv-path "results/1500/logltn_default_$i.csv" --n-examples-train 1500
    python log_ltn_max.py --csv-path "results/1500/logltn_max_$i.csv" --n-examples-train 1500
    python log_ltn_lseup.py --csv-path "results/1500/logltn_lseup_$i.csv" --n-examples-train 1500
    python log_ltn_sum.py --csv-path "results/1500/logltn_sum_$i.csv" --n-examples-train 1500
done

for i in 1 2 3 4 5
do
    python stable_product.py --csv-path "results/15000/stable_product_p2_$i.csv" --n-examples-train 15000 --p 2
    python stable_product.py --csv-path "results/15000/stable_product_p6_$i.csv" --n-examples-train 15000 --p 6
    python prod_rl.py --csv-path "results/15000/prod_rl_$i.csv" --n-examples-train 15000
    python log_ltn.py --csv-path "results/15000/logltn_default_$i.csv" --n-examples-train 15000
    python log_ltn_max.py --csv-path "results/15000/logltn_max_$i.csv" --n-examples-train 15000
    python log_ltn_lseup.py --csv-path "results/15000/logltn_lseup_$i.csv" --n-examples-train 15000
    python log_ltn_sum.py --csv-path "results/15000/logltn_sum_$i.csv" --n-examples-train 15000
done