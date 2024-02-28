import pandas as pd 
import numpy as np 
import sklearn
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.model_selection import train_test_split

def process(group):
    chunk_size = 5
    
    sentences = group['sentence_text'].tolist()
    #create the data
    row={}
    row['knesset_number'] = [group.iloc[0]['knesset_number'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    row['protocol_type']= [group.iloc[0]['protocol_type'] for _ in range(0, len(sentences)-(chunk_size-1), chunk_size)]

    #compine the each 5 sentences
    combined = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences)-(chunk_size-1), chunk_size)]
    row['sentence_text'] = combined

    #avarage_length = [((len(sentences[i])+len(sentences[i+1])+len(sentences[i+2])+len(sentences[i+3])+len(sentences[i+4]))/5.0) for i in range(0, len(sentences)-4, 5)]
    #row['avarage_length'] = avarage_length

    word_list = ['חוק','תודה','חברי','ראש','אדוני','חבר','הכנסת','הצעת']

    for word in word_list:
        word_list = [1 if word in combined[i] else 0 for i in range(len(combined))]

        row[word] = word_list


    #convert to a data frame
    data_frame = pd.DataFrame(row)
    return data_frame

def make_chunks(data):
    #devide the data into the right groups (according to protocol_name and the type) and apply procces for each group
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

    ###########################
    '''
    p_Counter = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    c_Counter = CountVectorizer(vocabulary=vectorizer.vocabulary_)

    P = p_Counter.fit_transform(plenary_data['sentence_text'])
    C = c_Counter.fit_transform(committee_data['sentence_text'])

    dic = {word: P[:,vectorizer.vocabulary_.get(word)].sum() /C[:,vectorizer.vocabulary_.get(word,1)].sum() if P[:,vectorizer.vocabulary_.get(word)].sum()>1000 else 0  for word_i,word in enumerate(vectorizer.vocabulary_.keys())}
    big_list = heapq.nlargest(30, dic, key=dic.get)
    for word in big_list:

        print(f'{word}:  {str(dic[word])} = {P[:,vectorizer.vocabulary_.get(word)].sum()} / {C[:,vectorizer.vocabulary_.get(word)].sum()}')
    '''
    #########################

    knesset_numbers = data['knesset_number']
    
    #avarage_lengths = data['avarage_length']
    our_feature_vector = [[] for _ in range(len(knesset_numbers))]

    word_list = ['חוק','תודה','חברי','ראש','אדוני','חבר','הכנסת','הצעת']
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


    print(f'SVM with corss validation: ')
    SVM_cross_validation = cross_val_predict(SVM,features,labels,cv=10,n_jobs=jobs)
    print(sklearn.metrics.classification_report(labels, SVM_cross_validation))


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42,stratify=labels)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)
    last_model = SVM

    print(f'KNN with split: ')
    y_pred = KNN.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    print(f'SVM with split: ')
    y_pred = SVM.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    

    #part 5.2
    print('our vecotr test validation')
    KNN = KNeighborsClassifier(10)
    SVM = svm.SVC(kernel='rbf')


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



    #part 6
    with open('knesset_text_chunks.txt', 'r',encoding='utf-8') as file:
        sentences = file.readlines()
        predictions = last_model.predict(vectorizer.transform(sentences))
        text=''
        for i,prediction in enumerate(predictions): 
            text+= prediction + '\n'
        with open('classification_results.txt','w') as write_file:
            write_file.write(text)