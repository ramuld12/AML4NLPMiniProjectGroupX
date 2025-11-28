# AML4NLPMiniProjectGroupX
Members: Alexander Rode (arod), Rasmus Herskind (rher)

Consists of two notebooks:
- experiments/baseline.ipynb: Baseline experiments for Bag-of-Words and Term Frequency-Inverse Document Frequency
- experiments/bert-base.ipynb: Experiments and visualisation of using Google's `bert-base-uncased` and `bert-base-cased`

## How to run code
We use python version 3.11.14, and torch with CUDA

Recommend to use conda. Then create an environment like:
```
conda env create -f environment.yml
```
If using pip, install dependencies:
```
pip install -r requirements.txt
```
Then run the cells in the .ipynb files under the experiments folder or run them with `python 'python_file_here.py'`


# Central problem, domain, data characteristics
We are doing sentiment analysis on the [Standford IMDB Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb).  
Our goal is to do Binary Classification of each review. That is, we want to predict whether a specific review is positive or negative

## Dataset description

The dataset consists of 25000 training samples and 25000 test samples. Some concrete examples are:

| text                                                                                                                                                                                                                                                                            | label |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Ned aKelly is such an important story to Australians but this movie is awful. It's an Australian story yet it seems like it was set in America. Also Ned was an Australian yet he has an Irish accent...it is the worst film I have seen in a long time                         | 0     |
| This is just a short comment but I stumbled onto this movie by chance and I loved it. The acting is great, the story is simple and touching, and the lines, especially from the 4-yr-old Desi, are so cute and sad. Seek it out.                                                | 1     |
| What an inspiring movie, I laughed, cried and felt love. For a true story,it does give you hope and that miracles do happen. It has a great cast. Ellen Burstyn, Samantha Mathis, Jodelle Ferland(she's 4 or 5yrs. old) what a actress. Its on Showtime. A Must See Movie!! :)= | 1     |

Here `0` represents a negative review and `1` represents a positive review. Both the training and test samples are balanced, meaning that 12500 samples are positive and 12500 are negative.

We have further created a validation dataset from 10% of the training samples.

# Central method
We started by creating two baseline models:
1. **Bag-of-words vectorizer** with Logistic Regression
2. **Term Frequency-Inverse Document Frequency** with Logistic Regression

After finishing the baselines we chose to use and compare Googles [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) and [`bert-base-cased`](https://huggingface.co/google-bert/bert-base-cased) tokenizers.
Our goal here is to compare the two different models to see if casing makes a difference in classifying movie reviews.

## Model Architecture
As described in the original [paper](https://arxiv.org/abs/1810.04805), the bert-model has the following architecture:
- 12 Transformer layers
- 768 Hidden layer size
- 12 attention heads
- ~110M total parameters

The choice of BERT comes down to a few, but strong reasons:
- It is a pretrained model, that with a GLUE score of 80.5% [see original [paper](https://arxiv.org/abs/1810.04805), table 1], can achieve state-of-the-art performance for different downstream tasks. 
- It understands the meaning of a word based on the other words in the sentence, and thus has a strong contextual understanding.
- It was made to be easily adaptable to different downstream tasks, which made it easier to implement our task, namely sentiment analysis.

## Training
We used [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) with the following hyper-parameters:
- Learning rate: 2e-5
- Train batch size: 8
- Evaluation batch size: 16
- Weight decay: 0.01
- Epochs: 10

We also added early stopping with patience of 2, meaning the model stops training if it does not improve its validation loss after 2 epochs.  
Evaluation happens after each epoch.

# Key experiments & results

We tried to run multiple different variants of the experiments, but in general each experiment followed the same pattern of aggressively overfitting.

## bert-base:

### cased Training
| Epoch | Training Loss | Validation Loss | Accuracy |     F1 | Precision | Recall |
|------:|--------------:|----------------:|---------:|-------:|----------:|-------:|
|     1 |        0.2871 |          0.2729 |   0.9192 | 0.9192 |    0.9257 | 0.9125 |
|     2 |        0.2120 |          0.3598 |   0.9196 | 0.9195 |    0.9490 | 0.8878 |
|     3 |        0.1042 |          0.3997 |   0.9240 | 0.9240 |    0.9464 | 0.8998 |

As we can clearly see the validation loss increases quite a lot after each epoch, triggering early stopping.

### uncased Training
| Epoch | Training Loss | Validation Loss | Accuracy |     F1 | Precision |   Recall |
|------:|--------------:|----------------:|---------:|-------:|----------:|---------:|
|     1 |        0.2543 |          0.2632 |   0.9240 | 0.9240 |    0.9420 |   0.9045 |
|     2 |        0.1749 |          0.3009 |   0.9300 | 0.9300 |    0.9363 |   0.9236 |
|     3 |        0.1122 |          0.3677 |   0.9340 | 0.9340 |    0.9410 |   0.9268 |

Again we see aggressive growth in validation loss after each epoch. This clearly suggests to us that the base model is overfitting.

### Testing
Below are our test results for both baselines and bert-base models

|           | Bag-of-Words | Term Frequency-Inverse Document Frequency | bert-base-cased | bert-base-uncased |
|-----------|--------------|-------------------------------------------|-----------------|-------------------|
| Loss      | 0.4338       | 0.3181                                    | 0.2435          | 0.2277            |
| Accuracy  | 0.8618       | 0.8834                                    | 0.9220          | 0.9338            |
| F1        | 0.8606       | 0.8832                                    | 0.9220          | 0.9338            |
| Precision | 0.8683       | 0.8847                                    | 0.9281          | 0.9392            |
| Recall    | 0.8531       | 0.8818                                    | 0.9148          | 0.9278            |
In general our baseline results compares pretty well with the ones presented in the [original paper for the dataset](https://aclanthology.org/P11-1015/) with results around 87-88%.

The bert-base models seem to perform relatively even, with a slight edge to the uncased model. 
Interesting to note, however, is that the test loss in both models are higher than the training loss, but much lower than the validation loss.
Below we show the confusing matrix for bert-base-uncased:

![confussion_matrix_uncased.png](confussion_matrix_uncased.png)

The model predicts almost even on both true positives and true negatives, but incorrectly predicts negative more often than incorrectly predicting positive.

# Discussion: summarise the most important results and lessons learned (what is good, what can be improved)
In general both models improved from the baselines counterparts. They both did however overfit quite a lot, which means that our training usually stopped after just 3 epochs. 
The drastic overfitting on the validation set could suggest that the model learns specific patterns present in the train set, which made us look a bit closer on some example datapoints. 
Here we found that some reviews had a label of 1 (positive), while being sentimentally negative. From the research paper [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/), 
the dataset was constructed such that reviews with a rating of 4 or lower would be labeled negative, while reviews with 7 or higher would be labeled positive - ratings in between where not included in the dataset. 
Based on this we found some concrete examples from the dataset with, what we assume, to have incorrect label. An example of this would be the following review / sample: 

![missClassifiedReview.png](missClassifiedReview.png)

With a label of 1 / positive. 
While we were not able to find the original review on imdb, the criteria for a review being positive is a rating of at least 7, and thus if this review is correct in its rating, then this review is not aligned with the criteria that the authors set for the dataset. With this in mind, the increase in validation loss after 3 epochs could be influenced by this flaw in the dataset.
From our limited research we have not been able to find others discussing this flaw in the dataset, even though [others seems to encounter the same problem of overfitting](https://huggingface.co/lyrisha/distilbert-base-finetuned-sentiment). 

Looking at the data points which contributed most to our validation loss, we saw a general pattern of reviews we would classify as negative or positive having a the opposite label. Concrete examples are shown below:

| Label | Loss   | Text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | 5.7770 | Great movie - especially the music - Etta James - "At Last". This speaks volumes when you have finally found that special someone.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 1     | 6.0665 | This film Evil Breed: The legend of samhain contains very little thought or effort. It is ridiculed with specs of ultra fast "slasher" style death and plain disgusting acts of death. The acting was rated a D as the actors show very little ability, and the stupidity of them in the film is too questionable. The way they portrayed what people their ages act like was incredibly different. The odd split of porn is fit in thought it really doesn't offer much, and any area that is respectable but is quite quickly run down with absolute gut wrenching death. Example is the poor fellow whom is disemboweled from his anus, and the scene lasts for about 5 minutes. It is terribly obvious of how little of a fight the kids put up. This film is a good choice for someone who likes to watch some awful deaths and practically torture. |

This would explain the validation loss trending upwards, since some, in our mind, negative reviews are labelled as 0 / negative, and some negative reviews are labelled as 1 / positive (and likewise for positive reviews).
We would expect the validation loss to stop around 50% (random guessing), given the models are unable to learn specific patterns due to these seemingly random labelled reviews.

Given more time we would like to dig deeper into the dataset, to see how many of these incorrectly labelled datapoints there are, and whether they are equally present in the test set. 
The test loss for both models is also relatively high, but given the 10 times larger size of the dataset, we would suspect the overfitting of the train set to matter less on average.

Lastly we initially set out to compare `bert-base-uncased` with `bert-base-cased`. They performed in general pretty similar, with a slight edge towards `bert-base-uncased` by 1-2% across the different metrics.
We would have liked to make more experiments with both, specifically trying to fine-tune the two different models, but ended up using most of our time trying to figure out why both of them overfit so much.


## References

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).  
*Learning Word Vectors for Sentiment Analysis*.  
Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).  
https://aclanthology.org/P11-1015/

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018).  
*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.  
arXiv preprint arXiv:1810.04805.  
https://arxiv.org/abs/1810.04805

Wolf, T., Debut, L., Sanh, V., Chaumond, J., et al. (2020).  
*Transformers: State-of-the-Art Natural Language Processing*.  
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.  
https://arxiv.org/abs/1910.03771

Hugging Face. (2024).  
*stanfordnlp/imdb Dataset*.  
https://huggingface.co/datasets/stanfordnlp/imdb

Hugging Face. (2024).  
*Transformers Documentation: Trainer API*.  
https://huggingface.co/docs/transformers/main_classes/trainer