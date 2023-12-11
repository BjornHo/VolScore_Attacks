VolScore attacks
===
This is the Python implementation for the VolScore attacks. This attack is built and based on the Score attacks from Damie et. al (https://github.com/MarcT0K/Refined-score-atk-SSE), and I highly recommend to explore their implementation.
## Structure
* `Old_Attacks`: Some early attacks that I implemented.
* `experiments_results`: All the results as .csv files which I used to create my plots.
* `main.py`: Here you can run my experiments by uncommenting the functions at the bottom.
* `parser.py`: The functions to parse Enron, Apache and Wikipedia datasets.
* `plot_graphs.py`: It utilizes the created .csv files to create plots in .pdf format.
* `query_generator.py`: Create queries and apply countermeasures.
* `requirements.txt`: The requirements that need to be installed using `pip install -r requirements.txt`
* `score_attacker.py`: The logic to perform the different type of attacks.

## Requirements
* Python 3.11
* Install all the requirements from `requirements.txt`
* Enron dataset: https://www.cs.cmu.edu/~enron/
* Apache dataset: use the script `Apache_Dataset_Setup.sh`.
* Wikipedia dataset: https://dumps.wikimedia.org/simplewiki/ . I used the dump from Oct 1st, 2023.
* Tool from David Shapiro to convert Wiki dump to plaintext: https://github.com/daveshap/PlainTextWikipedia

