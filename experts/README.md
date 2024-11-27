# experts, all contents of this folder have been done by me (Ishaq)
# The method has been engineered from scratch, built on just theory due to lack of legacy code.

The jupyter notebook experts_notebook contains a working implementation and explanation of methods in markdown of multiplicative weights method for giving buy/sell/hold stock verdict, a test is conducted on the stock SUZLON.NS

The files named with any variant of "integration" are failed attempts at integrating the model with the sentiment scores. The goal is to use these scores as experts, but the sentiment analysis is not at par with what it is expected to do, and the database changes at the last moment made it impractical to rework the format.
The regret (loss) of this model can be made to influence the sentiment analysis model. This is to be done in the future.

The LSTM example file has a working implementation, tested on wipro, due to poor performance so far, an attempt to integrate was not made.

The database files contain the code required to train the model, they contain the sentiment scores across various aspects.
