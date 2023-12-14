# <Scholar Wiki-Generator>

## Overview

This moduel has 2 components:

1. A Django backend that is responsible for initial prediction of Scholar prediction, data updating, and model retraining.

2. A jupyter notebook section that is responsible for classification model testing using the current dataset.

## Setup

List the steps needed to install your module's dependencies: 

1. The code is developed on Python version 3.10.1 and pip version 22.3.1

2. 
```
pip install -r requirements.txt 
```

3. Django backend setup is liked to a online sql database storage @https://bit.io/taigez/forward-scholar. Data can be viewed in csv format under /model_testing

4. Django backend can be hosted locally using 
```
py django_backend/mysite/manage.py runserver [ip port]
```

## Overall breakdown

```
taige-zhang-scholar-wiki/
    - requirements.txt
    - django_backend/               # django_backend that work collaboratively with Jayden's Google extension
        - mysite/
            - classifier/
                -- views.py         # function handling for frontend-backend interactions
                -- classify.py      # classification functions such as prediction and retraining
    - model_testing/                # testing XLNet's performance on existing datasets
        -- awd_data.csv             # labelled dataset for academic awards
        -- edu_data.csv             # labelled dataset for educational background
        -- int_data.csv             # labelled dataset for research interest
        -- cs.csv                   # testing dataset that is independent of training data
        -- Awards.ipynb             # XLNet performance evaluation for academic awards
        -- Education.ipynb          # XLNet performance evaluation for educational background
        -- Interests.ipynb          # XLNet performance evaluation for research interest
        -- performance.ipynb        # BERT-Cosine performance evaluation
```    



## Functional Design (Usage)

In classify.py:

* Takes as input a list of strings, each representing a unique sentence from a webpage url, and return a dictionary with the sentences that matches our interested areas.
```python
    def process_paragraph(text: list[str]):
        ... 
        return (
            { 'background': initial_prediction_educational_background, 'interest': initial_prediction_research_interest 'awards': initial_prediction_academic_awards },
            total_parsed_sentences
        )
```

* 3 retraing functions that creates a balanced dataset after saving wrongly predicted sentences to the main databse. Wrongly predicted sentences were given a heavier weight.
```python
    def train_awd():
        # save dataset
        # create balanced dataset
        # retrain
        ...
        return
```

```python
    def train_awd():
        # save dataset
        # create balanced dataset
        # retrain
        ...
        return 
```

```python
    def train_awd():
        # save dataset
        # create balanced dataset
        # retrain
        ...
        
```

## Demo video
Github does not support video of size larger than 10Mb
https://youtu.be/QLqekHmDl94

## Algorithmic Design 
The frontend-backend system creates a feedback loop that helps the model to learn from its mistakes and create a much more robust dataset.

First, all sentences from a webpage is parsed through the chorme extention and forwarded to the backend in dictionary form. The backend then process these sentences one by one to product a set of initial prediction. These predictions are returned to the frontend for user to verify and correct. Mislabeled sentences are stored within the local sql lite database before pushing to the main one with a heavier weight. This allows us to generate an expanding dataset that corrects overfitting and inaccarcy through human-validations.

With a few performance evaluation of our original BERT classification model, it came to our attention that the single layer BERT classifier using cosine simularity does not produce the best result. XLNet has been tested on the current dataset and has yeild a much promising result. More improvements can be made to the strucutre of the prediction, such that one sentence will only belong to one category, and prediction will be made based on the condition given by other category as well.





## Issues and Future Work

In this section, please list all know issues, limitations, and possible areas for future improvement. For example:

* High false positive rate for research interest. 
* Inaccurate webpage text extraction.
* Replace cosine similarity classifier with XLNet
* Replace current binary classifier with multi-class classifier.
* Implement trafilatura to aid with information extraction.


## Change log

Use this section to list the _major_ changes made to the module if this is not the first iteration of the module. Include an entry for each semester and name of person working on the module. For example 



## References 
include links related to datasets and papers describing any of the methodologies models you used. E.g. 


* BERT paper: Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
* XLNet paper: Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding.

