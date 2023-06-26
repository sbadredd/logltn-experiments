#!/bin/bash
for CONFIG_FILE in configs/*.yml
do
    echo "$CONFIG_FILE"
    cp $CONFIG_FILE config.yml
    python train.py
done
