#!/bin/sh

for i in 0 5 10 13 14 15 16
do
  python3 ../main.py --config ../config/config_files/SelectedCrossTransformer/nexus/nexussubj$i.yaml
done