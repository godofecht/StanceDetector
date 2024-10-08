# RoBERTa-Based Stance Detection

This repository contains the implementation of a stance detection model based on RoBERTa, an advanced extension of the BERT model. The work is grounded in the research presented in the paper ["Stance Prediction with RoBERTa"](https://aclanthology.org/2020.nlp4if-1.3.pdf), authored by Anushka Shankar under the supervision of Dr. Harish Tayyar.

## Overview

Stance detection involves identifying the position or stance expressed by a user towards a particular topic or issue, especially on social media platforms like Twitter and Reddit. This repository leverages the RoBERTa model, which builds on the robust capabilities of BERT by improving the training methodology and utilizing a larger dataset for better contextual understanding.

## Key Features

- **RoBERTa Model Integration:** Enhanced version of BERT, pre-trained on a larger corpus with more robust fine-tuning capabilities.
- **Stance Classification:** Identifies whether a text expresses support, opposition, or neutrality towards a given topic.
- **Cross-Platform Application:** Suitable for analyzing stance on multiple social media platforms, including Twitter and Reddit.
- **Customizable:** Easily adaptable for various domains and languages with additional fine-tuning.
- **Preprocessing Pipeline:** Includes text preprocessing steps such as tokenization, stop-word removal, and normalization, tailored for social media text.

## Installation
## Usage

```markdown
## Data Preparation

### Step 1: Extracting Data
The data is extracted and unzipped as follows:

```bash
tar -xjf rumoureval2019.tar.bz2
unzip rumoureval2019/rumoureval-2019-test-data.zip -d 'rumoureval2019/'
unzip rumoureval2019/rumoureval-2019-training-data.zip -d 'rumoureval2019/'
```

### Step 2: Processing Data
The extracted data is processed using the following scripts:

```bash
python Process_Twitter_Data.py
python Process_Reddit_Data.py
```

## Model Training

### Step 3: Loading and Preprocessing Data
Load the processed data and prepare it for training:

```python
from Data_Cleaning_functions import processStanceData
import pandas as pd

# Load datasets
twitterTrainDf = pd.read_csv('TwitterTrainDataSrc.csv')
redditTrainDf  = pd.read_csv('RedditTrainDataSrc.csv')
twitterDevDf   = pd.read_csv('TwitterDevDataSrc.csv')
redditDevDf    = pd.read_csv('RedditDevDataSrc.csv')
twitterTestDf  = pd.read_csv('TwitterTestDataSrc.csv')
redditTestDf   = pd.read_csv('RedditTestDataSrc.csv')

# Process data
trainDf = processStanceData(twitterTrainDf, redditTrainDf)
devDf = processStanceData(twitterDevDf, redditDevDf)
testDf = processStanceData(twitterTestDf, redditTestDf)
```

### Step 4: Training the Model
Train the stance detection model using the prepared data:

```python
from StanceDetectorModel import StanceDetector, Tfidf_Nn, tfidf

model = Tfidf_Nn()
stanceDetector = StanceDetector(model, tfidf)
history = stanceDetector.fit(x_train, y_train, x_dev, y_dev, epochs=100, verbose=1)
```

## Model Evaluation

### Step 5: Visualizing Performance
Plot the training and validation loss and accuracy:

```python
import matplotlib.pyplot as plt

# Plot Loss
plt.title("Learning Curve - Loss")
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["dev_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

![Training and Validation Loss](path_to_your_loss_plot.png)

## Conclusion
The model shows improvement over the epochs, as indicated by the decreasing training loss. The validation accuracy stabilizes, suggesting the model is learning effectively without overfitting.

```

## Applications

- **Social Media Analysis:** Identify public opinion on trending topics, political issues, and social movements by analyzing stance on platforms like Twitter and Reddit.
- **Brand Monitoring:** Understand customer sentiment and stance towards products or services.
- **Political Sentiment Analysis:** Gauge voter sentiment and stance towards candidates or policies.
- **Misinformation Detection:** Detect stance to identify potential sources of misinformation or polarized content.

## Contributions

Contributions are welcome! If you have suggestions, find a bug, or want to contribute to the development, please fork the repository and create a pull request. Ensure your code adheres to the existing style and includes adequate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## References

- Shankar, A., & Tayyar, H. (2020). [Stance Prediction with RoBERTa](https://aclanthology.org/2020.nlp4if-1.3.pdf). In Proceedings of the Third Workshop on NLP for Internet Freedom: Censorship, Disinformation, and Propaganda.
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). arXiv preprint arXiv:1907.11692.

---
