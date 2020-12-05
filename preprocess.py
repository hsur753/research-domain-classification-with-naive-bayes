import csv
import string

"""
Hussel Suriyaarachchi

Clean training data by removing all stop words and non-alphabetic characters
Output file: processed.csv used for training
"""

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'without', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'also', 'among', 'whereas' ,'upon']

print ("*********************Pre-Processing*********************")

#count number of rows in data
with open("trg.csv","rt") as source:
    rdr= csv.reader( source )
    row_count = sum(1 for row in rdr)

with open("trg.csv","rt") as source:
    rdr = csv.reader(source)
    text_list = [None] * row_count #store list of fitered sentence/abstracts
    c = 0

    for r in rdr:
        text = r[2].translate(str.maketrans('', '', string.punctuation)) #remove punctuation
        text_tokens = text.split() #tokenize the next by whitespace
        tokens_without_sw = [word for word in text_tokens if not word in stop_words and word.isalpha()] #remove stopwords and and chars that are non-alphabetic
        filtered_sentence = (" ").join(tokens_without_sw) #create filtered sentence

        text_list[c] = filtered_sentence
        c+=1

with open("trg.csv","rt") as source:
    rdr = csv.reader(source)
    c = 0
    with open("processed.csv","wt", newline='') as result:
        wtr = csv.writer( result )
        for r in rdr:
            wtr.writerow( (r[1], text_list[c]) ) #write class and filtered abstract to row of new csv file
            c+=1

print ("************************Complete************************")
print()
