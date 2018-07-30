# nmt
Creating a Neural Machine Translator for English-French
## Installation
Download the French-English corpus from http://www.manythings.org/anki/
and put it into this folder. Run the script prep\_data.py to clean and split
the data into training and testing sets. Then run trainer.py to train the
neural machine translator and finally run tester.py with your own text within the
variable 'phrase' to translate it. Currently does French to English, but loading
other datasets from that link should work as well as long as the prep\_data.py file
is changed to get that file instead.
