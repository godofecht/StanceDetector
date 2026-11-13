# StanceDetector

An improved stance-prediction model built on BERT and RoBERTa.

## What this is

Stance detection means working out the position a post takes towards a claim. This project reads replies in Twitter and Reddit conversations about rumours and sorts each reply into one of four stances towards the rumour:

- **support**: the reply backs the rumour up
- **deny**: the reply says it is false
- **query**: the reply asks for evidence or more detail
- **comment**: the reply does neither

The code in this repo uses a count-based model: it turns each post into word scores called TF-IDF and passes them through a small neural network (a set of connected artificial neurons that learns to turn inputs into outputs). The research it comes from adds count-based features like these to BERT and RoBERTa, two large pre-trained language models. The BERT and RoBERTa part is not in this repo.

You run it in Python. It prints its progress while it trains, then a table of how well it did on posts it has never seen.

## What you can learn from it

- Computer Science: natural language processing (NLP), the field of getting computers to work with human language.
- Computer Science: how text becomes numbers with TF-IDF, so a network can work with it.
- Computer Science: how a neural network classifier is trained with PyTorch, and why the data is split into training, development and test sets.
- Maths and Statistics: accuracy, precision, recall and F1 score, and why accuracy alone can mislead when one class is much more common.
- Media studies and Citizenship: how rumours spread online and how people respond to them.

## Run it

### On your own computer

**You will need:**

- [Python](https://www.python.org/downloads/) 3.
- [Git](https://git-scm.com/downloads).
- About 2 GB of free disk space for the Python packages.

These steps were run on macOS with Python 3.12, pandas 2.2.3, scikit-learn 1.9.0, SciPy 1.17.1 and PyTorch 2.10.0.

1. Download the code:

   ```bash
   git clone https://github.com/godofecht/StanceDetector.git
   cd StanceDetector
   ```

2. Make a virtual environment (a private folder of Python packages for this project) and install the packages:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas scikit-learn scipy torch matplotlib
   ```

   On Windows, run `venv\Scripts\activate` instead of the `source` line.

3. Unzip the RumourEval 2019 data, which is already in the repo:

   ```bash
   unzip rumoureval2019/rumoureval-2019-test-data.zip -d rumoureval2019/
   unzip rumoureval2019/rumoureval-2019-training-data.zip -d rumoureval2019/
   ```

   On Windows, you can instead right-click each zip file, choose **Extract All**, and extract it into the `rumoureval2019` folder.

4. Turn the raw posts into CSV files. The current `Process_Twitter_Data.py` and `Process_Reddit_Data.py` stop with `KeyError: 'inre_x'`, so use the 2023 versions from the git history:

   ```bash
   git show 86d4e9e9:Process_Twitter_Data.py > Process_Twitter_Data_2023.py
   git show 86d4e9e9:Process_Reddit_Data.py > Process_Reddit_Data_2023.py
   python Process_Twitter_Data_2023.py
   python Process_Reddit_Data_2023.py
   ```

   You should now have six files: `TwitterTrainDataSrc.csv`, `TwitterDevDataSrc.csv`, `TwitterTestDataSrc.csv` and the same three for Reddit. Pandas may print warnings about "chained assignment". They do not stop the scripts.

5. Open `StanceDetectorModel.py` in a text editor and put a `#` at the start of line 44, the line that begins `trainDf.drop(`. With freshly made CSV files there is no empty value for it to drop, and it stops with an `IndexError`.

6. Save this as `train.py` in the same folder:

   ```python
   from StanceDetectorModel import StanceDetector, Tfidf_Nn, tfidf
   from StanceDetectorModel import x_train, y_train, x_dev, y_dev, x_test, y_test
   from sklearn.metrics import classification_report

   model = Tfidf_Nn()
   stance_detector = StanceDetector(model, tfidf)
   history = stance_detector.fit(x_train, y_train, x_dev, y_dev, epochs=100, verbose=1)

   predictions, probabilities, real = stance_detector.get_predictions(x_test, y_test)
   print(classification_report(real, predictions, digits=4,
                               target_names=['support', 'deny', 'query', 'comment']))

   print(stance_detector.predict_new(model, 'example.txt'))
   ```

7. Run it:

   ```bash
   python train.py
   ```

   Training takes about ten seconds. It prints one line per epoch (one pass over the training data) for 100 epochs. The development accuracy should end near 0.67. Then it prints a table for the test set. The overall accuracy should be about 0.74 and the macro average F1 about 0.34. The last line labels the two posts in `example.txt`, which came out as support and comment.

To draw the learning curves from the earlier README, add this to the end of `train.py`:

```python
import matplotlib.pyplot as plt

plt.title("Learning Curve - Loss")
plt.plot(history["train_loss"], label="Training Loss")
plt.plot([float(x) for x in history["dev_loss"]], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

The notebook `stance_detector.ipynb` runs the same steps. It was written for Google Colab, so its first cell expects a `rumoureval2019.tar.bz2` file and its save cell writes to `/content/`.

## How it works

**The data.** RumourEval 2019 is a shared task (a competition with a fixed dataset) on rumours in Twitter and Reddit conversations. Each conversation starts with a source post that spreads a rumour. Every reply has a stance label made by people.

**Cleaning.** For each reply, the program joins three pieces of text: the reply, the post it answers, and the source post. Web links become `$URL$` and `@` mentions become `$MENTION$`, so the model treats every link and every username alike.

**TF-IDF.** TF-IDF stands for term frequency, inverse document frequency. It gives each word, and each pair of words next to each other, a score. The score is high when the word is common in this post and rare in posts overall, so words like "the" score low and telling words like "fake" score higher. Words that appear in fewer than 10 training posts, or in more than half of them, are ignored. Each post becomes a list of about 5,300 scores (5,343 in the notebook's run, 5,349 in the rerun for this guide).

**The network.** The scores go into a hidden layer of 128 neurons. A neuron adds up its inputs, each multiplied by a weight (a number that sets how much that input matters), and passes the total through tanh, which squashes it into the range -1 to 1. During training, dropout switches off a random 10% of the hidden neurons each time, which stops the network relying too much on any one of them. The output layer has 4 neurons, one per stance, and softmax turns them into four probabilities that add up to 1.

**Training.** The program compares the predictions with the labels and nudges every weight to reduce the error. About two thirds of the training replies are comments, so a lazy model could label everything a comment and still score well. To stop that, mistakes on rarer classes cost more: 8 times for support and query, and 20 times for deny.

| File or folder | What is in it |
| --- | --- |
| `StanceDetectorModel.py` | loads the data, builds TF-IDF, defines the network (`Tfidf_Nn`) and the training class (`StanceDetector`) |
| `Data_Cleaning_functions.py` | joins the reply with its context, replaces links and mentions, turns labels into numbers |
| `Process_Twitter_Data.py`, `Process_Reddit_Data.py` | read the raw posts and labels and write CSV files (see Run it for the working versions) |
| `DataPaths.py` | where the unzipped data lives |
| `stance_detector.ipynb` | a notebook that runs the whole pipeline, with saved results |
| `example.txt` | two example posts to label |
| `rumoureval2019/` | the RumourEval 2019 data, scorer and licence |

## Ideas to try

1. In `StanceDetectorModel.py`, change `HIDDEN_LAYER_UNITS` from 128 to 32 or 512. Does the test macro F1 change?
2. In the same file, set `class_weights` in `StanceDetector.__init__` to `[1.0, 1.0, 1.0, 1.0]`. Accuracy may go up while the model gets worse at spotting denials. Use the classification report to see why.
3. Change `min_df`, `max_df` or `ngram_range` in the `TfidfVectorizer` line in `StanceDetectorModel.py`. How does the number of features change?
4. Write your own replies in `example.txt`, one per line, and see how they are labelled.
5. In `Data_Cleaning_functions.py`, extend `processText` to handle hashtags or emoji. Does it help?
6. A bigger project: follow the paper in Credits and replace TF-IDF with a pre-trained BERT or RoBERTa model, then compare the macro F1.

## Technical details

Stance detection identifies the position or stance expressed by a user towards a particular topic or issue, especially on social media platforms like Twitter and Reddit. BERT (Bidirectional Encoder Representations from Transformers) is a language model pre-trained on large amounts of text that reads each word in the context of the words on both sides. RoBERTa builds on BERT with a changed training method and a larger training dataset.

Details of the model in this repo:

- Labels are mapped support 0, deny 1, query 2, comment 3.
- `TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1, 2))`, fitted on the training text. The input text column is `TextSrcInre`: the cleaned reply, then the cleaned previous post and source post.
- `Tfidf_Nn`: Linear(number of TF-IDF features, 128), tanh, Dropout(0.1), Linear(128, 4), Softmax. The forward pass returns the hidden-layer output and the class probabilities.
- `StanceDetector`: `nn.CrossEntropyLoss` with class weights [8.0, 20.0, 8.0, 1.0], Adam with learning rate 0.02, full-batch training. `fit` returns `train_loss`, `train_acc`, `dev_loss` and `dev_acc`. `predict`, `get_predictions` and `predict_new` return label predictions.
- Random seed 16 for NumPy and PyTorch.
- `CrossEntropyLoss` applies its own softmax, so the loss is computed on softmax outputs that have been through softmax twice. This still trains, but more slowly than with raw outputs.

Results saved in the notebook, on the 1,827-reply test set:

| Stance | Precision | Recall | F1 |
| --- | --- | --- | --- |
| support | 0.107 | 0.038 | 0.056 |
| deny | 0.252 | 0.267 | 0.260 |
| query | 0.156 | 0.204 | 0.177 |
| comment | 0.842 | 0.880 | 0.861 |

Accuracy 0.740, macro average F1 0.338. The rerun for this guide gave accuracy 0.741 and macro F1 0.337. Over 100 epochs the training loss falls from 1.39 to 0.85 while the development loss rises from about 1.38 to about 1.42, so the model starts to overfit (learn the training set too closely) well before the end.

Possible uses of stance detection:

- Social media analysis: public opinion on trending topics, political issues and social movements.
- Brand monitoring: customer stance towards products or services.
- Political analysis: stance towards candidates or policies.
- Misinformation detection: stance as a signal for possible misinformation or polarised content.

## Project status

The model trains and gives the results above when run as described in Run it. Known problems:

- The repo description and the earlier README describe a RoBERTa model. The code is a TF-IDF model, and no BERT or RoBERTa code is included.
- `Process_Twitter_Data.py` and `Process_Reddit_Data.py` were rewritten in 2024 and now stop with `KeyError: 'inre_x'`. The 2023 versions work.
- Line 44 of `StanceDetectorModel.py` assumes the training data has an empty value and fails when it does not.
- `StanceDetectorModel.py` reads all six CSV files as soon as it is imported, so it cannot be imported before step 4.
- `rumoureval2019/.DS_Store` is a stray macOS file.

## Licence

You can use, copy, change and share the code for any purpose, as long as you keep the copyright notice. See [LICENSE](LICENSE) (MIT).

The data in `rumoureval2019/` belongs to the RumourEval 2019 shared task and is licensed under Creative Commons Attribution 4.0 (CC BY 4.0) and the Twitter licence. If you use it, you must cite the shared task (see Credits). See [rumoureval2019/LICENSE](rumoureval2019/LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Credits

- Written by Abhishek Shivakumar.
- The research this builds on: Anushka Prakash and Harish Tayyar Madabushi (2020). [Incorporating Count-Based Features into Pre-Trained Models for Improved Stance Detection](https://aclanthology.org/2020.nlp4if-1.3/). In Proceedings of the 3rd NLP4IF Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda, pages 22 to 32.
- Data: Genevieve Gorrell, Ahmet Aker, Kalina Bontcheva, Elena Kochkina, Maria Liakata, Arkaitz Zubiaga and Leon Derczynski (2019). [SemEval-2019 Task 7: RumourEval, Determining Rumour Veracity and Support for Rumours](https://www.aclweb.org/anthology/S19-2147). Proceedings of the 13th International Workshop on Semantic Evaluation, pages 845 to 854.
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen and others (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). arXiv:1907.11692.
- Built with [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/) and [pandas](https://pandas.pydata.org/).
