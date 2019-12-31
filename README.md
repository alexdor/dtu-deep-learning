# dtu-deep-learning

## Data

- Download train and valid pairs (article, title) of OpenNMT provided Gigaword dataset from [here](https://github.com/harvardnlp/sent-summary)
- For the Gigaword copy the following files from the train folder: `train.article.txt`, `train.title.txt`, `valid.article.filter.txt`and `valid.title.filter.txt` to `data/unfinished` folder
- To convert `.txt` file into `.bin` file and chunk them further, run (requires Python 3 and Tensorflow):

```bash
python make_data_files.py
```

- You will find the data in `data/chunked` folder and vocab file in `data` folder
