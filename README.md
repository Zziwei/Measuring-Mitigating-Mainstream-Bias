# Measuring-Mitigating-Mainstream-Bias
Code for the WSDM 2022 paper -- Fighting Mainstream Bias in Recommender Systems via LocalFine Tuning

## Requirements
python 3  
tensorflow 1.14.0  
pytorch 1.8.1   
numpy   
sklearn   
pandas   

## Excution
To run the VAE, DC, WL, LFT, and EnLFT models, just run command 'python3 model_name.py' in terminal. For example, to run LFT, use command 'python3 LFT.py'.

All model outputs have been uploaded in this repo. To see the detailed results for different user subgroups of different mainstream levels, run the corresponding jupyter notebook for different models. For example, to see the results of LFT, run the 'Data/ML1M/analysis-LFT.ipynb'.

