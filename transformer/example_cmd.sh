#!/bin/bash

python ../fairseq/generate.py /home/tridao/learning-circuits/fairseq/data-bin/iwslt14.tokenized.de-en --batch-size 1 --beam 1 --remove-bpe --gen-subset test --quiet --cpu --path results/DynamicConvSparse/model.pt --sparse-dynamic-conv
