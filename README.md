# Advanced Machine Learning course

This is my github repo for the AML specialisation offered by Yandex and HSE on coursera. 

## Modules
1. [Intro to Deep Learning](https://www.coursera.org/learn/intro-to-deep-learning/) . The course covers the fundamentals of Deep Learning, from the basic ideas of overfitting and underfitting to state of the art CNN and RNN.
	
	- During the course, I coded a neural network in numpy which 	helped me understand how backprop really works. 
		
	- The assignments are open ended, encouraging experimentation and trial and error, as it would be the case in a real world application.
		
	- The assignments have an interesting blend of numpy, keras, and tensorflow. This helps to think of these modules as tools in the same toolbox instead of isolated tools.
		
	- The final project is designing a **captioning neural network**, featuring both a CNN for feature extraction (Pretrained InceptionV3) and an RNN. It is trained on a set of (images, captions) and the network learns to caption any image (that resembles the training set, that is)

2. [Competitive Data Science](https://www.coursera.org/learn/competitive-data-science). The course covers exploratory data analysis, feature generation, and feature tuning and model validation, all taught by expert kaggle competition winners.
	- The course involved **participating in an actual competition**, and I ranked in the top 10% (Out of ~ 300 participants)
	- My final model was designed, trained, and ran in an Amazon AWS instance, and it included lagged features, mean encoded features, and features derived from item descriptions using PCA
	- The assignments involved a range of tasks, but were mostly to build understanding of the actual goals involved in the competition
3. [Bayesian Methods for Machine Learning](https://www.coursera.org/learn/bayesian-methods-in-machine-learning). The course builds upon preexisting understanding of ML methods, and places them in the context of Bayesian statistics.
	- Many key concepts are covered: conjugate priors, latent variable modes, gaussian mixtures, expectation maximisation, variational inference, latent dirichlet allocation, MCMC with Gibbs and Metropolis-Hastings sampling, variational autoencoders, and bayesian optimization
	- The final project involved designing an **algorithm to help a user generate faces with certain properties from a variational autoencoder**: Initially I show the user different faces, then the user is progressively shown faces and asking to rate them. Using these values and GPyOpt, the code adjusts the latent variables of the VAE to approach the face the user wants.

4. Ǹatural Language Processing](https://www.coursera.org/learn/language-processing). The course covers a variety of NLP approaches and concepts, including the basics such as lemmatising or bag of words, to word embeddings, and then in terms of modelling it covers Hidden Markov Models and finally neural network based models.
	- The final project was to design a simple chatbot that could either answer technical questions (by replying with a 	relevant answer from stack overflow, or could just chit chat. The network itself was  implemented in Tensorflow 		following a character level seq2seq approach.
