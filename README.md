## Overview

This codebase performs classification and energy/angle regression of single isolated physics particles based on their calorimeter deposition showers.

The code can also be used to generate showers via a GAN architecture.

## SamplePrep

The SamplePrep/ folder contains code for generating and preparing MC samples.

This process involves the complete simulation, segmentation, and feature calculation steps for generating single-particle calorimeter shower samples.

Resampling and data repacking code are also in this folder. The resampling procedure takes currently-existing events and recalculates them as seen in different detector geometries. The data repacking code performs alterations on H5 format data to improve I/O performance.

## Training

Training/ contains the TriForce framework for classification and regression, as well as training code for our baseline comparisons.

Full instructions for training with TriForce are provided in the Training/TriForce/ folder.

## Analysis

Analysis/ contains scripts for generating paper plots.
