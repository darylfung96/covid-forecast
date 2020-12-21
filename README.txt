21th December 2020
------------------
Daryl
- Created LSTM and trained on the covid forecasting
- Integrated matrix profile provided by Qian Liu
- Utilize matrix profile to feed in concatenation together with the values of covid forecasting
- Utilize the idea of relative positioning based on each location of the matrix profile and input in concatenation with values of covid forecasting
- Plotted the 5 vold validation loss
- Plotted the prediction results

18th December 2020
------------------
Qian Liu
- performed matrix profiling algorithm on covid forecasting dataset
- provide resources for discussion on how to integrate matrix profiling algorithm to forecast

17th December 2020
------------------
Daryl
- I have scrape the data from https://cmu-delphi.github.io/delphi-epidata/ using their api and attached the csv file
in this email. The columns in the csv file are as follows:

time_value: the date of the data
hospital admission: the percentage of newly admission to the hospital that are covid positive
hospital admission_adj: the percentage of newly admission to the hospital that are covid positive but with with systematic day-of-week effects removed
confirmed case: the daily confirmed cases
total confirmed case: the cumulative of confirmed cases
death case: the daily death cases
total death case: the cumulative of death cases