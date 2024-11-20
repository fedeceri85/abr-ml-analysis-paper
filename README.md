# ABR-ML Analysis

This repository contains routines for the analysis of ABR (Auditory Brainstem Response) data using Machine Learning techniques. The analysis is for the paper `A machine-learning-based approach to predict early hallmarks of progressive hearing loss` by Ceriani et al.

## Overview

This project employs machine learning approaches to develop predictive models for hearing assessment in mice.

## Requirements

- Python >= 3.10
- NumPy
- Pandas 
- Scikit-learn 
- PyQt5 (for DataExplorer)
- sktime 
- Required packages listed in `requirements.txt`



1. `notebooks`: jupyter notebooks for training/testing the ML models and preparing the figures. 
2. `src`: utility functions for the analysis, and the DataExplorer GUI for manual wave analysis and strain annotation.

Additionaly, the `data` folder with the original dataset and the `results` folder with the trained models and validation/testing results are needed.

A more general version of DataExplorer can be found here: https://github.com/fedeceri85/abrWaveAnalyser?tab=readme-ov-file