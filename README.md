# clab

<!--
clab
|── base
    |── algorithms
    |── common
    |── dataloader
        |── rl_data_generator.py
        |── test_dataloader.py
    |── env
        |── offline_env.py
    |── strategy
        |── bc_strategy.py
|── data
|── main
    |── main_bc.py
    |── main_test.py
|── run
    |── run_bc.py
    |── run_evaluate.py 
-->


## Data Processing
Download the traffic granularity data and place it in the biddingTrainENv/data/ folder.
The directory structure under data should be:
```
NeurIPS_Auto_Bidding_General_Track_Baseline
|── data
    |── traffic
        |── period-7.csv
        |── period-8.csv
        |── period-9.csv
        |── period-10.csv
        |── period-11.csv
        |── period-12.csv
        |── period-13.csv
        |── period-14.csv
        |── period-15.csv
        |── period-16.csv
        |── period-17.csv
        |── period-18.csv
        |── period-19.csv
        |── period-20.csv
        |── period-21.csv
        |── period-22.csv
        |── period-23.csv
        |── period-24.csv
        |── period-25.csv
        |── period-26.csv
        |── period-27.csv
        
```