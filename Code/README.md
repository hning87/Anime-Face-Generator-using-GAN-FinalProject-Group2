# Code Instruction

## Download Data
To download data correctly, please run the code “Download_from_GoogleDrive.py” first.
The dataset link are:
https://drive.google.com/open?id=11j_HC40a2t2q9ZtsLp4OPwF2xzhJCG1i  
https://drive.google.com/file/d/1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79  
provided by National Taiwan University.

## Data Preprocessing
The dataset originally contains images and tags.  
For collecting the images data, please run the code **"data preprocessing.py"** to get the completely dataset of the images.  
After running this code, you will get a images file called **"imgs.npy"**

## DCGAN
For running the DCGAN, please use the code **"DCGAN.py"**

## WGAN
For running WGAN, please use the code **"WGAN.py"**

## WGAN-GP
For running WGAN-GP, please use the code **"WGANGP.py"**


After running each model, you will get a saved model called **"__GAN_g.pth"** and **"__GAN_d.pth"**
## Generate Images
After training the models, you can run the **"Generate_Images_.py"** to test how the model performs.  
In this code, you need to change the **"__GAN_g.pth"** file name at the bottom which included in the load pretrained model part.  
And also you can change the file name **"___.jpg"** at the bottom included in the save image part.

