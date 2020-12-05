from collections import defaultdict
import math
import string

"""
Hussel Suriyaarachchi

Implementation of the Multinominal Naive Bayes Classifier with inverse document frequency.

"""

class NaiveBayes:

    def __init__(self,unique_classes):

        self.classes=list(set(unique_classes)) #list of unique class labels

        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'without', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'also', 'among', 'whereas' ,'upon']

    """
    Preprocessing of test data, cleaning the text before inference

    returns: cleaned abstract
    """
    def clean_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
        text_tokens = text.split() #tokenize the next by whitespace
        tokens_without_sw = [word for word in text_tokens if not word in self.stop_words and word.isalpha()]  #remove stopwords and and chars that are non-alphabetic
        filtered_sentence = (" ").join(tokens_without_sw) #create filtered sentence
        return filtered_sentence


    """
    Stores frequency of words for corresponding classes
    Stores frequency of documents for all unique words
    """
    def tokenize(self, data, index):
        a_list = [None] *len(data.split())
        i=0
        for word in data.split(): #iterate through words in abstract
            self.class_words_dict[index][word]+=1 #upate frequency of words for corresponding classes

            if (word not in a_list):
                a_list[i] = word
                self.class_word_count[word]+=1 #update frequency of documents for all unique words
                i+=1

    """
    Train classifier

    self.dataset: list of training data
    self.labels: list of class labels
    self.class_words_dict: list of dictionaries with frequency of words for each class
    self.class_word_count: dictionary with frequency of documents for words
    self.weighted_count: list of dictionaries with weighted count of words in each class
    """
    def train(self, data, classes):
        self.dataset = data
        self.labels = classes
        self.class_words_dict = [defaultdict(lambda:0) for index in range(len(self.classes))]
        self.class_word_count = defaultdict(lambda:0)
        self.weighted_counts = [defaultdict(lambda:0) for index in range(len(self.classes))]

        for class_index,label in enumerate(self.classes, 0):
            class_abstracts = list()
            index = 0
            for class_label in self.labels: #collect all abstracts of corresponding class from dataset
                if label == class_label:
                    class_abstracts.append(self.dataset[index])
                index += 1
            for abstract in class_abstracts: #iterate through each abstract
                self.tokenize(abstract, class_index) #call tokenize function to count frequencies

        class_probability = [None] * len(self.classes) #list of priors
        words_list = []

        for class_index, label in enumerate(self.classes):

            #calculate prior
            counter = 0
            for class_label in self.labels:
                if label == class_label: counter += 1
            class_probability[class_index] = counter/len(self.labels)

            #inverse document frequency
            for key in self.class_words_dict[class_index].keys():
                 weighted_count= self.class_words_dict[class_index][key] * (math.log (len(self.labels) / self.class_word_count[key]))  #  frequency of word in class * log( number of abstratcs / number of abstratcs with the word )
                 self.weighted_counts[class_index][key] = weighted_count

            words_list+=self.class_words_dict[class_index].keys() #store all words in dataset

        weighted_sum_list = [None] * len(self.classes)

        #weighted count sum for each class
        for class_index, label in enumerate(self.classes):

            weighted_sum = sum(self.weighted_counts[class_index].values()) #add all weighted counts of class
            weighted_sum_list[class_index] = weighted_sum

        V = list(set(words_list)) #get unique words
        unique_word_count = len(V) #number of unique words in dataset |V|

        denominators = [None] * len(self.classes) #denominator of the weighted likelihood calculation
        for class_index, label in enumerate(self.classes):
            denominators[class_index] = weighted_sum_list[class_index] + unique_word_count #weighted_count(c) + |V|

        self.model_weight_tuple = [None] * len(self.classes) #list of tuples for each class
        for class_index, label in enumerate(self.classes):
            self.model_weight_tuple[class_index] = (self.weighted_counts[class_index], class_probability[class_index], denominators[class_index]) #store all weight counts, priors and denominators of each class in a tuple


    """
    Run predictions on test dataset

    returns: list of predicted class labels
    """
    def test(self,test_set):

        predictions=[] #list of predicted class labels
        for test in test_set: #iterate through each test abstract
            likelihood = [0] * len(self.classes) #likelihood for each class
            test = self.clean_text(test) #clean text of test abstract

            for class_index, label in enumerate(self.classes):
                for word in test.split():

                    numerator = self.model_weight_tuple[class_index][0].get(word,0)+1 # weighted_count(i,c) + 1
                    probability = numerator/(self.model_weight_tuple[class_index][2]) # ( weighted_count(i,c) + 1 ) / ( weighted_count(c) + |V| )
                    likelihood[class_index] += math.log(probability) # sum of logged probabilities (instead of multiplying raw probabilities)

            #choosing class
            class_result_probability = [None] * len(self.classes)
            for class_index, label in enumerate(self.classes):
                class_result_probability[class_index] = likelihood[class_index] + math.log(self.model_weight_tuple[class_index][1]) # add logged prior (instead of multipying raw prior)

            result = self.classes[class_result_probability.index(max(class_result_probability))] #get argmax
            predictions.append(result) #store prediction

        return predictions
