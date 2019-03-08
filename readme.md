##About

This is a template for solving regression problems using ANN.
Introduced grid search to find the optimal hyper parameters,
and then use these values in the final model.

The duplicated code for the final model was due to trying to find out why there were huge disparities in the losses.
But I suppose that's due to the way CV works.

Supplied MAE and MSE as cost function(s).
Using ANN instead of usual Linear Regression models.

Things Ive learned: regression problems are unlike classification problems,
thus 'accuracy' are not good scoring metrics to use.