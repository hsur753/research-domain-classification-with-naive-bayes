import csv
from statistics import mean

from naive_bayes import NaiveBayes

"""
Hussel Suriyaarachchi

Input file: processed.csv
Train the classifier on training data
Perform 10-fold cross-validation to evaluate model performance
Run inference on test set
"""

#count number of rows in data
with open("processed.csv","rt") as source:
    rdr= csv.reader( source )
    row_count = sum(1 for row in rdr)

class_list = [None] * row_count
abstract_list = [None] * row_count

#store abstracts and classes of dataset in corrersponding lists
with open("processed.csv","rt") as source:
    rdr= csv.reader( source )
    i=0
    for r in rdr:
        class_list[i] = r[0]
        abstract_list[i] = r[1]
        i+=1

#remove csv header from lists
abstract_list.pop(0)
class_list.pop(0)

print ("*********************Training*********************")

#10-fold cross-validation
num_folds = 10
accuracy_list = list()
subset_size = len(abstract_list)//num_folds
for i in range(num_folds):
    nb=NaiveBayes(class_list) #initialize Naive Bayes class
    test_data = abstract_list[i*subset_size: (i+1) * subset_size] #subset for validation
    test_classes = class_list[i*subset_size: (i+1) * subset_size]
    training_data= abstract_list[:i*subset_size] + abstract_list[(i+1)*subset_size:] #subset for training
    training_classes = class_list[:i*subset_size] + class_list[(i+1)*subset_size:]

    nb.train(training_data, training_classes) #call train function

    pclasses=nb.test(test_data) #store predictions of test data

    #count correctly classified instances
    count = 0
    for i, predicted_class in enumerate(pclasses):
        if (predicted_class == test_classes[i]): count += 1

    accuracy = count/len(test_classes)
    accuracy_list.append(accuracy)

print ("*********************Complete*********************")

print ("10-Fold Cross-Validation Mean Accuracy: ", mean(accuracy_list)*100,"%") #mean accuracy of cross-validation
print()
print("Running model on test set...")

#inference on test csv
with open("tst.csv","rt") as source:
    rdr= csv.reader( source )
    row_count = sum(1 for row in rdr)

text_list = [None] * row_count

#create list of abstracts
with open("tst.csv", "rt") as source:
    rdr = csv.reader( source )
    i=0
    for r in rdr:
        text_list[i] = r[1]
        i+=1

#remove csv header
text_list.pop(0)

nb=NaiveBayes(class_list) #initialize Naive Bayes class
nb.train(abstract_list, class_list) #train classifier on training data
pred = nb.test(text_list) #get predictions

#store predictions in new csv file
with open("tst.csv","rt") as source:
    rdr = csv.reader(source)
    c = -1
    with open("hsur753.csv","wt", newline='') as result:
        wtr = csv.writer( result )
        for r in rdr:
            if (c==-1): wtr.writerow( (r[0], "class") ) #add csv header
            else: wtr.writerow( (r[0], pred[c]) )
            c+=1

print("Finished.")
