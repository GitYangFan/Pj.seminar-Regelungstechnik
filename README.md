# Pj.seminar-Regelungstechnik
Let's make Hamster! 
This repository is used to store and exchange our code and data related to this project Hamster in Pj. Seminar Regelungstechnik. Let's unleash your creativity!

## The current version of Hamster code is uploaded here and also can be found in gitlab with branch "Baho"
- The type of the velocity message was changed to add time stamp
- Small changes were done in packages "hamster_driver", "hamster_interfaces" and "tracking_controller".
- Changes in the code are commented with "//Project seminar"



## If only interest in the two Algorithm mentioned in the report, turn to branch "Handong" and read the following guide:
### the codes of two methods introduced in the report are in files:
`KFRealTime.py`\
`MHERealTime.py`

### the code to run the simulation refer to file:
`run.py`\
It will save the simulation result as CSV file under the folder `./sim_result` \
please read the comments in `run.py` carefully.\
Before running, please ensure there are collected data under the folder `./data`

### Gaussian Process Regression
Refer to `Gaussian Process Regression.ipynb` and part of `plot_sim_result.py`

### Coordinate Correction
Refer to `Coordinate Correction.ipynb` and `o_R_e.csv`

### Attention
The code will record all the info during running. 
If the hardware to run the algorithms has memery limit, please add the funtionality that delete the redundance data.
E.g., cut the `self.dataset` of classes in  `KFRealTime.py` and `MHERealTime.py`, and correct `self.i`.
