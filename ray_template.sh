#!/bin/bash
workers=( worker host names here )
rayexec=$(which ray)
if [[ $1 == "start" ]]; then
    set -x  # echo on
    PYTHONPATH=$PWD:$PWD/fairseq/:$PWD/fairseq/scripts:$PYTHONPATH ray start --head 2>&1 | grep -o "ray start --redis-address .*" | cut -d ' ' -f4 > config/redis_address
    # Using the trick here to control echo on/off: https://stackoverflow.com/questions/13195655/bash-set-x-without-it-being-printed
    { set +x; } 2>/dev/null  # echo off
    address=$(cat config/redis_address)
    for worker in "${workers[@]}"
    do
        numgpus=1
        if [[ $worker == raiders* ]]; then
            numgpus=2
        fi
        set -x
        ssh $worker "PYTHONPATH=$PWD:$PWD/fairseq/:$PWD/fairseq/scripts:$PYTHONPATH nohup $rayexec start --redis-address $(cat config/redis_address) --num-gpus=$numgpus > /dev/null 2>&1 &"
        { set +x; } 2>/dev/null
    done
elif [[ $1 == "stop" ]]; then
    for worker in "${workers[@]}"
    do
        set -x
        ssh $worker "nohup $rayexec stop > /dev/null 2>&1 &"
        { set +x; } 2>/dev/null
    done
    ray stop
    rm -f config/redis_address
fi
