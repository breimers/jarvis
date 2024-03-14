#!/bin/bash

sys=$(uname -s)

if [[ $sys = "Linux" ]]; then
    export TOKENIZERS_PARALLELISM=true
fi

if [[ $sys = "Darwin" ]]; then
    export TOKENIZERS_PARALLELISM=true
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi