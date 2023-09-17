# Google-Street-View-Guesser

Usage : First run get_locations.py and get some randomized locations or you can use your location dataset too
Second run get_images and add your Google Street View Api key to it.
Third run change_name to change the naming format to index
Now you can run resnet50_train.py safely but don't forget to change the directories according to your computer.

Have achieved Train Mse 300, Validation Mse 700 with epoch 100 Which means average 4000km off the way.
Model can be improved by not using only images for example using texts on the image.
