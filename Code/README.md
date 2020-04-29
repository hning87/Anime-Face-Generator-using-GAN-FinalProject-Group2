# Final-Project-Group2

## Download Data
For download data correctly, please run the code “Download_from_GoogleDrive.py” first.
The dataset link are:
https://drive.google.com/open?id=11j_HC40a2t2q9ZtsLp4OPwF2xzhJCG1i
https://drive.google.com/file/d/1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79
provided by National Taiwan University.

## Data Preprocessing
The dataset originally contains images and tags.
For collecting the images data, please run the code “preprocessing.py” to get the completely dataset of the images.
After run this code, you will get a images file called “imgs.npy”

## DCGAN
For running the DCGAN, please use the code “DCGAN.py”

## WGAN
For running WGAN, please use the code “WGAN.py”.

## WGAN-GP
For running WGAN-GP, please use the code “WGANGP.py”

## Generate Images
After training the models, you can run the “Generate_Images.py” to test how the model performs.
In this code, you need to change the .pth file name at the bottom which included in the load pretrained model part.

