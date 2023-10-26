#!/bin/bash

rm api_reference/*
rm api_reference.rst
sphinx-apidoc -f -e -o api_reference ../colabseg

for file in api_reference/colabseg.*; do mv "$file" "${file#colabseg.}"; done

echo "=============" > api_reference.rst
echo "API reference" >> api_reference.rst
echo "=============" >> api_reference.rst
echo ".. toctree::" >> api_reference.rst
echo "   :maxdepth: 2" >> api_reference.rst
echo "" >> api_reference.rst

for file in api_reference/colabseg.*.rst; do
  new_filename=$file
  filename="${new_filename%.*}"
  echo "   $filename" >> api_reference.rst
done
