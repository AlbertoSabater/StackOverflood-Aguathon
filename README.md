# StackOverflood - Aguathon

This repository contains the code of my solution for the "[I Aguathon](https://www.itainnova.es/blog/eventos/i-hackathon-del-agua-aguathon/#formulario)". This competition is aimed to predict the amount of water that Ebro's river will have in the following 24, 48 and 72 hours, given only the current amount of water in different points of the river, without using historical data.

With this code, I achieved a **5th position** in the final evaluation.

The folder "ftp" contains all the files needed to make predictions, create new training datasets and training new models.

The proposed solution builds a base dataset with the common amount of water in different points of the river, for each hour and day, by averaging data of consecutive days, hours and years. This average is weighted according to a certain gaussian function. The train and prediction datasets are made of the difference of the current amount of water in different points of the river with the amount of water from the base dataset at different time steps. 

To get predictions, I first train a LightGBM model with the built dataset. Additionally, both automatic and manually feature selection methods are used before training.

To get more insights about this solution, check the [PDF](https://github.com/AlbertoSabater/StackOverflood-Aguathon/blob/master/ftp/README%20-%20C%C3%B3digo%2C%20ejecuci%C3%B3n%20y%20desarrollo.pdf) (in Spanish) located under the "ftp" folder.
