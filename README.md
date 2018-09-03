## Bikesharing python project:

This is the NUS-ISS Data analytics ensemble CA's python source codes.
Env & Install:
- python3
- sklearn
- pandas
- tensorflow-keras
- matplotlib

#### How to run:
python3 process/Process.py
input the model name from (nn, rf, xgb, blend, stacking)

#### Some brief explanation of the folder structure:

bikesharing/model:
- NeuralNetwork.py - nn model
- RandomForest.py - rf model
- Xgboost.py - xgb model
- Stacking.py - ensemble stacking model

bikesharing/process:
- Preprocessing.py - data load and preprocessing
- Process.py - run script

bikesharing/profit:
- Profit.py - calculate the revenue/cost/profit

bikesharing/util:
- Util.py - utility functions

