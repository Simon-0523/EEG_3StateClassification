Please create an anaconda virtual environment by:

> $ conda create --name RCM python=3.8

Activate the virtual environment by:

> $ conda activate RCM

Install the requirements by:

> $ pip3 install -r requirements.txt


Unzip s01.zip and data_raw_state.zip, the s01.dat should in the .py file directory
# Run the code


To run the code for new training, please type the following command in the terminal:

The overall EEG sample in s01.dat includes data: 477×4×15360 and label: 477.

data_subject folder: data of each subject. For example, s01.dat contains data: 9×4×15360 and label: 9.

data_raw_anxiety folder: data saved after data segmentation.

> $ python main.py

# Reproduce the results
 Please run the code by:
> $ python main.py --reproduce

# Data Description
For example, s01. dat represents the data of a subject. This includes EEG data (9 × 4 × 15360) and labels (9).
The data_raw_anxiety folder stores the segmented data and labels.



