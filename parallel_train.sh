#!/usr/bin/env bash
mpirun -np 2 -hostfile hostfile -bind-to none -map-by slot -x NCCL_DEBUG=INFO python ~/machine_learning/Supervised_Learning/Kaggle_Plant-Seedlings-Classification_Distributed/train.py

