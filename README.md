# CSE-601-Data_Mining
Data Mining projects on bioinformatic topic (implmented in Python)
##
## 1) Project - PCA & Appriori Algorithm
**Problem:** This project contains 2 parts. First part is to implement the PCA (Principle Components Analysis) algorithm, project the high-dimensional data to 2 dimensions, and plot the 2-dimensional data points. Second part is on implementing Apriori
algorithm and rule generation algorithm

**Approach:**
The project constructs a Prolog bigram language model using small [DA_Corpus.text](bigram-sentence-evaluator/DA_Corpus.txt) corpus.

Steps taken ([bigram_model.pl](bigram-sentence-evaluator/bigram_model.pl)):

1. The [DA_Corpus.text](bigram-sentense-evaluator/DA_Corpus.txt) corpus is normalized using [unix](bigram-sentence-evaluator/unix_commands.txt) commands.
2. Created a prolog readable [unigram.pl](bigram-sentence-evaluator/unigrams.pl) and [bigram.pl](bigram-sentence-evaluator/bigrams.pl) database from normalized corpus.
3. In the final step, implemented [bigram_model.pl](bigram-sentence-evaluator/bigram_model.pl) which computes the probability of any word sequence, of any size, via a predicate called **calc_prob/2**. The predicate calc_prob/2 works in log space and applies laplace smoothing on fly to compute the probability of given sentence.

**Sample outputs:** 
As shown in the output below, sentence like "the book fell" will have better value than "i fell on the book"

![output1](bigram-sentence-evaluator/output/output1.png)

Similarly the sentence like "the book that he wanted fell on my feet" will have better value than "book the that he wanted fell on my feet"

![output2](bigram-sentence-evaluator/output/output2.png)
