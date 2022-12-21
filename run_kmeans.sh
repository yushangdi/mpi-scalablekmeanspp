#!/bin/bash

OUTPUTDIR="outputs"

[ -d ${OUTPUTDIR} ] || mkdir ${OUTPUTDIR}

datasets=("Mallat"
"UWaveGestureLibraryAll"
"NonInvasiveFetalECGThorax2"
"MixedShapesRegularTrain"
"MixedShapesSmallTrain"
"ECG5000"
"NonInvasiveFetalECGThorax1"
"StarLightCurves"
"HandOutlines"
"UWaveGestureLibraryX"
"CBF"
"InsectWingbeatSound"
"UWaveGestureLibraryY"
"ShapesAll"
"SonyAIBORobotSurface2"
"FreezerSmallTrain"
"Crop" 
"ElectricDevices"
)

workers=(96 48 36 24 12 4 1)

num_clusters=(8
8
42
5
5
5
42
3
2
8
3
11
8
60
2
2
24
7
)


for wk in "${workers[@]}"; do
ind=0
for dataset in "${datasets[@]}"; do
    command="mpiexec -np ${wk} ./mpi_main -i /home/ubuntu/datasets/UCR/${dataset}_X.dat -b -n ${num_clusters[$ind]} -o > outputs/${dataset}_${wk}th_kmeans.txt"
    echo "$command"
    eval "$command"
    let ind++
done
done


# https://scikit-learn.org/stable/computing/parallelism.html#parallelism