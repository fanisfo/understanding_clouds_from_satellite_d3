## Kaggle challenge - Understanding Clouds from Satellite Images - Idea of solution
Data available on Kaggle.

Requirements available in `requirements.txt`.

What's inside each file?
* `data_exploration.ipynb` jupyter notebook to have an overlook on which kind of data we are working with;
* `dataset_loading.py` implementation of the data loader to return image and mask to be fed up to the architecture;
* `dice_loss.py` custom implementation of the dice loss, since there is no built in function in python;
* `main.py` to run the whole code;
* `unetmcl.py` custom implementation of a UNet for multi label classification;
* `utils.py` a function to encode the resulting pixels as requested in the kaggle challenge.

If you are interested in Data Exploration and Preprocessing, run `data_exploration.ipynb`.

After that, you can run `main.py` to start the image segmentation task.

This implementation **IS NOT WORKING** but it may be useful to have an idea on a possible solution.
