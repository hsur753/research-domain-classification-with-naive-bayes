Note: The algorithm was implemented and tested using the dataset found in the ```data``` directory; slight modifications may be required to support custom datasets.

A multinominal Naive Bayes classifier (NBC) using inverse document frequency was implemented
to classify the data. The training dataset consisted of 4000 instances and three attributes,
including the class. Since we are dealing with text-based data, preprocessing was performed
to clean the data. Any commonly used words (stop words) were dropped from the data as
they would appear too frequently in the text to be of use in classification. The texts were
further cleaned by removing any non-alphabetic characters. The processed data was saved to
a file for ease of access when training the data and avoid processing time.

Since we have extended the classic Naive Bayes classifier to a multinominal NBC with
inverse document frequency, dictionary data structures were implemented to accommodate
these changes. During the training of data, we maintain dictionaries to store both the
frequency of words for corresponding classes and the frequency of documents for all unique
words used in training. These values will be needed to calculate the weighted likelihood of
words in the dataset as we have adapted the inverse document frequency extension in our
classifier. Furthermore, all probabilities were logged as to avoid underflow when performing
calculations.

The performance of our model was evaluated using 10-fold cross-validation, where the mean
accuracy based on the number of correctly classified instances was used as the performance
metric. The implementation of our model performs preprocessing (data cleaning) on the test
data as well. The performance of our model against a standard Naive Bayes classifier and the
null model can be seen as follows (10-fold cross-validation and the same preprocessed
dataset with the cleaned data were used in all cases)

| Classifier   |      Accuracy      |
|----------|:-------------:|------:|
| Null model (ZeroR - WEKA) |  53.6% |
| NaiveBayes (WEKA) |    80.1%   |
| Multinominal NBC with inverse document frequency | 95. 8 % |
| | |

The null model and Naive Bayes classifier of WEKA were used as the benchmark to evaluate
the performance of our model. The null model, which classified all instances as the majority
class (class E), did significantly worse in terms of accuracy. The standard NBC of WEKA
achieved good accuracy; however, our extended model performed much better. The increase
in accuracy for the same evaluation method and dataset suggest that the extensions
implemented to the standard NBC significantly improved the performance in the task of text
classification.