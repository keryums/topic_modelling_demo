# Topic Modelling Example
A sample workflow using LDA, CorEx, Anchors, and Word Embeddings. If you want to get to the heart of the matter, see [topic_model_example](topic_model_example.ipynb), but the devil is certainly in the details. 

This repository includes the following files: 
* [preprocessing_example](preprocessing_example.ipynb): morphological processing and phrase detection via collocations
* [preprocessing_word2vec](preprocessing_word2vec.ipynb): trains a Word2Vec model on the grocery reviews corpus 
* [topic_model_example](topic_model_example.ipynb): topic modelling using Latent Dirichlet Allocation (LDA), basic Correlation Explanation (CorEx), anchored CorEx, and topic enrichment using word embeddings, and topic aggregations
* [helpers/helper_base](helpers/helper_base.py): helper functions for reading and rendering data
* [helpers/helper_prep](helpers/helper_prep.py): helper functions for text preprocessing
* [helpers/helper_model](helpers/helper_model.py) helper functions for topic modelling
