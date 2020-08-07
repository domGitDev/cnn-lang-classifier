
# packages

pip install tensorflow==2.2.0 keras==2.3.1 pandas==1.0.1 matplotlib==3.1.3 seaborn==0.10.0

# clone reposity from github

git clone https://github.com/domGitDev/cnn-lang-classifier.git

cd cnn-lang-classifier

# train model

python main.py -f ./data/lang_data.csv

# run inference

python inference.py -f ./data/lang_data.csv


# CSV data format

text      |   labels

sentence  |   text label
