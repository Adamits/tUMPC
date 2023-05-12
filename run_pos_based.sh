#! /bin/bash

for language in English German Greek Icelandic Russian; do
    echo Running on ${language}
    mkdir -p predictions/${language}

    python src/pos_based.py \
        --clusters data/clustered/McCurdyEtAl/bible/${language}.clustered \
        --corpus data/train/bible/${language}.txt \
        --output predictions/${language}/${language};
done