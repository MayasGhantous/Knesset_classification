import pandas as pd 
import numpy as np 
import sklearn
import random
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.model_selection import train_test_split



#will be deleted
from sklearn.decomposition import TruncatedSVD
import heapq
import time

def process(group):
    #first make the general row
    chunk_size = 5
    #row = {'protocol_name': [group.iloc[0]['protocol_name']] , 'knesset_number': [group.iloc[0]['knesset_number']] ,'protocol_type': [group.iloc[0]['protocol_type']] ,'speaker_name': [group.iloc[0]['speaker_name']] }
    
    sentences = group['sentence_text'].tolist()
    #create the data
    row={}
    #row['protocol_name'] = [group.iloc[0]['protocol_name'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    row['knesset_number'] = [group.iloc[0]['knesset_number'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    row['protocol_type']= [group.iloc[0]['protocol_type'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    #row['speaker_name'] = [group.iloc[0]['speaker_name'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]

    #compine the each 5 sentences
    combined = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    row['sentence_text'] = combined

    #avarage_length = [((len(sentences[i])+len(sentences[i+1])+len(sentences[i+2])+len(sentences[i+3])+len(sentences[i+4]))/5.0) for i in range(0, len(sentences)-4, 5)]
    #row['avarage_length'] = avarage_length

    word_list = ['חוק','הצעה','תודה','חברי','ראש','אדוני','חבר',]
    for word in word_list:
        word_list = [1 if word in combined[i] else 0 for i in range(len(combined))]
        #word_list = [combined[i].count(word) for i in range(len(combined))]

        row[word] = word_list


    #convert to a data frame
    data_frame = pd.DataFrame(row)
    return data_frame

def make_chunks(data):
    #devide the data into the right groups (according to protocol_name  and speaker_name) and apply procces for each group
    #the func in DataFrameGroupBy.apply(func), func takes a data frame and can return a data frame and 
    #thats why this code works because process return a data frame
    result_df = data.groupby(['protocol_type','protocol_name'],dropna = True).apply(process).reset_index(drop = True)

    return result_df


def down_sample(data,N):
    # if we want to down sample non positive number then dont do any thing
    if N<=0:
        return data
    number_list = random.sample(range(len(data)),k=N)
    return data.drop(number_list).reset_index(drop = True)
    


if __name__ == '__main__':
    start_time = time.time()
    # change the sead
    random.seed(42)
    np.random.seed(42)
    # part 1,2 
    df = pd.read_csv('knesset_corpus.csv',index_col=None)
    df=make_chunks(df)

    #we need the indexes in down sample
    committee_data = df.loc[df['protocol_type'] == 'committee'].reset_index(drop=True)
    plenary_data = df.loc[df['protocol_type'] == 'plenary'].reset_index(drop=True)

    #part 3
    committee_data = down_sample(committee_data,len(committee_data)-len(plenary_data))
    plenary_data = down_sample(plenary_data,len(plenary_data)-len(committee_data))

    #connect the 2 types with randomness 
    data = pd.concat([committee_data,plenary_data])
    data = data.sample(frac=1,random_state=42).reset_index(drop = True)


    
    #part 4.1
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['sentence_text'])
    features = vectorizer.transform(data['sentence_text'])

   
    #part 4.2
    #our feature vector is the 100 tokens that have most occurences
###########################
    
    '''
    
    #get the most uncommn words accros the to data type

    p_Counter = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    c_Counter = CountVectorizer(vocabulary=vectorizer.vocabulary_)

    P = p_Counter.fit_transform(plenary_data['sentence_text'])
    C = c_Counter.fit_transform(committee_data['sentence_text'])

    dic = {word: C[:,vectorizer.vocabulary_.get(word)].sum() /P[:,vectorizer.vocabulary_.get(word,1)].sum() if C[:,vectorizer.vocabulary_.get(word)].sum()>2000 else 0  for word_i,word in enumerate(vectorizer.vocabulary_.keys())}
    big_list = heapq.nlargest(30, dic, key=dic.get)
    for word in big_list:

        print(f'{word}:  {str(dic[word])} = {C[:,vectorizer.vocabulary_.get(word)].sum()} - {P[:,vectorizer.vocabulary_.get(word)].sum()}')
    
    '''

#########################

    top_vectorize = TfidfVectorizer(max_features=10,ngram_range=(1,1))
    top_vectorize.fit(data['sentence_text'])
    #our_feature_vector = top_vectorize.transform(data['sentence_text']).toarray().tolist()
    knesset_numbers = data['knesset_number']
    
    #avarage_lengths = data['avarage_length']
    our_feature_vector = [[] for _ in range(len(knesset_numbers))]
    word_list = ['חוק','הצעה','חברי','ראש','אדוני','חבר']

    for i in range(len(our_feature_vector)):
        our_feature_vector[i].extend([knesset_numbers[i]])
        for word in word_list:
           our_feature_vector[i].extend([data.iloc[i][word]])


    #part 5.1
    jobs = -1
    labels = data['protocol_type']
    print('BoW train validation')
    KNN = KNeighborsClassifier(10)
    SVM = svm.SVC(kernel='linear')
    print(f'KNN with corss validation: ')
    KNN_cross_validation = cross_val_predict(KNN,features,labels,cv=10,n_jobs=jobs)
    print(sklearn.metrics.classification_report(labels, KNN_cross_validation))
    #print ("score -"+str(cross_val_score(KNN,features,data['protocol_type'],cv=10,n_jobs=-1).mean()))


    print(f'SVM with corss validation: ')
    SVM_cross_validation = cross_val_predict(SVM,features,labels,cv=10,n_jobs=jobs)

    print(sklearn.metrics.classification_report(labels, SVM_cross_validation))
    #print (cross_val_score(SVM,features,data['protocol_type'],cv=10,verbose=2).mean())


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42,stratify=labels)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)
    last_model = SVM

    y_pred = KNN.predict(X_test)
    print(f'KNN with split: ')
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    y_pred = SVM.predict(X_test)
    print(f'SVM with split: ')
    print(sklearn.metrics.classification_report(y_test, y_pred))
    

    #part 5.2
    print('our vecotr test validation')
    KNN = KNeighborsClassifier(10)
    SVM = svm.SVC(kernel='linear',)
    print(f'Our KNN with corss validation: ')
    KNN_cross_validation = cross_val_predict(KNN,our_feature_vector,labels,cv=10,n_jobs=jobs)
    print(sklearn.metrics.classification_report(labels, KNN_cross_validation))

    print(f'our SVM with corss validation: ')
    SVM_cross_validation = cross_val_predict(SVM,our_feature_vector,labels,cv=10,n_jobs=jobs)
    print(sklearn.metrics.classification_report(labels, SVM_cross_validation))


    X_train, X_test, y_train, y_test = train_test_split(our_feature_vector, labels, test_size=0.1, random_state=42,stratify=labels)
    print(f'our KNN with split:')
    KNN.fit(X_train,y_train)
    y_pred = KNN.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))

    print(f'our SVM with split: ')
    SVM.fit(X_train,y_train)
    y_pred = SVM.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    end_time = time.time()
    print('took')
    print(end_time - start_time)



    #part 6
    with open('knesset_text_chunks.txt', 'r',encoding='utf-8') as file:
        sentences = file.readlines()
        predictions = last_model.predict(vectorizer.transform(sentences))
        text=''
        for i,prediction in enumerate(predictions): 
            text+= prediction + '\n'
        with open('classification_results.txt','w') as write_file:
            write_file.write(text)