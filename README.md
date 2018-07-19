# nmt
Creating a Neural Machine Translator for English-French
## Installation
Download the French-English corpus from http://www.manythings.org/bilingual/fra/
and put it into this folder. Run the script data\_prep.py to clean and split
the data into training and testing sets. Then run trainer.py to train the
neural machine translator and finally run tester.py with your own text within the
variable 'phrase' to translate it. Currently does French to English.
