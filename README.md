# Quantization-based clustering algorithm
- This repository contains implementation of quantization based clustering algorithm proposed in [this paper](https://www.sciencedirect.com/science/article/pii/S0031320310000981).
- Also contains experiments which compare it to another implementations of algorithms from K-Means family from sci-kit learn library.

# Directory structure

## qbca.py
  - contains implementation of the proposed quantization-based clustering algorithm
## max_heap.py
  - simple implementation of max heap needed by the algorithm

## test_gaussian.py
  - compares QBCA with another algorithms using `external indices` on generated Gaussian data
## test_gaussian_internal.py
  - compares QBCA with another algorithms using `internal indices` on generated Gaussian data

## explore.R
  - exploring external indices using R

## test_image.py
  - compares QBCA with another algorithms using image segmentation on sample image "test_image.jpg"
## test_iris.py
  - compares QBCA with another algorithms using `external indices` on famous IRIS dataset

## test_iris_external.py
  - compares QBCA with another algorithms using `internal indices` on famous IRIS dataset

## visualize_gaussian_data.py
  - visualizing gaussian data that are used for experiments and generated

## visualize_gaussian_results.py
  - visualize `external indices` comparing QBCA to another algorithms

## figures/
  - contain figures that were generated for comparing algorithms and used in the report

## results/
  - contains CSV files containing results frm external indices comparing performance of different algorithms

## screenshots/
  - screenshots used in the paper when describing code optimalization
