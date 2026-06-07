#!/bin/bash

for f in *.jpg; do
    convert ./"$f" ./"${f%.jpg}.eps"
done
