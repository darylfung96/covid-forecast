# A novel matrix profile-guided attention LSTM model for forecasting COVID-19 cases in USA

## Getting Started

### Getting Data
Run:

```
python data_scraping.py
```

The covid-19 data will be saved as covid-forecast.csv.

### Training models
Run:

```
python main.py
```

It will start running LSTM, LSTM attention, LSTM matrix attention, LSTM relative attention, and CNN-LSTM on the covid dataset.
It will also generate results in "result" folder. You will be able to find the loss and prediction there saved as npy.

### Evaluation/Results

In order to view the results, you can run:

```
python plot_result.py
```

Inside plot_result file, feel free to change the type to "loss" or "prediction" to get the result of the loss and the prediction of each models. 
The prediction will be saved into result/csv folder. The loss will be saved as temp.csv and you can look at the 
rmse,mae,mape loss for each k_fold training of the models.
