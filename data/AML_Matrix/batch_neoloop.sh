#!/bin/bash

for hic in 018
do
    # enter corresponding directory first: cd ./data/AML_Matrix
    mkdir $hic
    mv $hic.cool $hic/
    cd $hic
    calculate-cnv -H $hic.cool -g hg38 -e MboI --output ${hic}_10kb.cnv
    segment-cnv --cnv-file ${hic}_10kb.cnv --binsize 10000 --output ${hic}_10k.seg --nproc 10 --ploidy 2
    cooler balance $hic.cool
    correct-cnv -H $hic.cool --cnv-file ${hic}_10k.seg --nproc 10
    # put sv file under $hic/ first
    assemble-complexSVs -O ${hic}_10kb -B $hic.sv -H $hic.cool
    neoloop-caller -O $hic.neo-loops.txt -H $hic.cool --assembly ${hic}_10kb.assemblies.txt --no-clustering --prob 0.95
    cd ..
done