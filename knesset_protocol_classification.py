import pandas as pd 
import numpy as np 
import sklearn
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.model_selection import train_test_split

import time

def process(group):
    #first make the general row
    row = {'protocol_name': [group.iloc[0]['protocol_name']] , 'knesset_number': [group.iloc[0]['knesset_number']] ,'protocol_type': [group.iloc[0]['protocol_type']] ,'speaker_name': [group.iloc[0]['speaker_name']] }
    sentences = group['sentence_text'].tolist()
    #compine the each 5 sentences
    combined = [f'{sentences[i]} {sentences[i+1]} {sentences[i+2]} {sentences[i+3]} {sentences[i+4]}' for i in range(0, len(sentences)-4, 5)]
    row['sentence_text'] = combined

    #avarage_length = [((len(sentences[i])+len(sentences[i+1])+len(sentences[i+2])+len(sentences[i+3])+len(sentences[i+4]))/5.0) for i in range(0, len(sentences)-4, 5)]
    #row['avarage_length'] = avarage_length
    #series = pd.Series(combined, index=np.arange(len(combined)))

    #fill the right data for all the chuncks
    data_frame = pd.DataFrame.from_dict(row,orient='index').transpose().ffill()
    return data_frame

def make_chunks2(data):
    #devide the data into the right groups (according to protocol_name  and speaker_name) and apply procces for each group 
    result_df = data.groupby(['protocol_name', 'speaker_name']).apply(process).dropna().reset_index(drop = True)

    return result_df




'''def make_chunks(data):
    # if we have diffrent speakers then dont put then in the same chunck also if there are not in the same protocol do not put them in the same chunck

    #if we sort the data then we can make assienmet more efficient
    sorted_data = data.sort_values(by = ['protocol_name','speaker_name'])
    i = 0 
    new_data = pd.DataFrame(columns = sorted_data.columns)
    while i < len(sorted_data):

        #the current row
        protocol_type = sorted_data.iloc[i]['protocol_type']
        cuurent_number =sorted_data.iloc[i]['knesset_number'] 
        protocol_name =sorted_data.iloc[i]['protocol_name']
        sentence = sorted_data.iloc[i]['sentence_text']
        speaker = sorted_data.iloc[i]['speaker_name']
        i+=1
        j = 1
        while j<5: #max 5 sentences
            if i >= len(sorted_data):
                break
            #take the sentnce
            currnet_protocol_name =sorted_data.iloc[i]['protocol_name']
            sentence += " " +sorted_data.iloc[i]['sentence_text']
            current_speaker = sorted_data.iloc[i]['speaker_name']
            #if we have somthinf wrong (not the same speaker or not the same protocol) then change what we are searching for
            if currnet_protocol_name != protocol_name or current_speaker != speaker:
                j = 1
                cuurent_number =sorted_data.iloc[i]['knesset_number']
                protocol_name = currnet_protocol_name
                speaker = current_speaker
                sentence = sorted_data.iloc[i]['sentence_text']
                i+=1
            else:
                i+=1
                j+=1
        if j != 5:# if we dont have enough sentence then dont save the results
            continue
        new_row = pd.DataFrame({'protocol_name':[protocol_name],'knesset_number':[cuurent_number],'protocol_type':[protocol_type],'speaker_name':[speaker],'sentence_text':[sentence]})
        new_data = pd.concat([new_data,new_row],ignore_index=True)
    return new_data'''

def down_sample(data,N):
    # if we want to down sample non positive number then dont do any thing
    if N<=0:
        return data
    number_list = random.sample(range(len(data)),k=N)
    return data.drop(number_list)
    


if __name__ == '__main__':
    start_time = time.time()
    # change the sead
    random.seed(42)
    np.random.seed(42)
    # part 1
    df = pd.read_csv('knesset_corpus.csv',index_col=None)
    committee_data = df.loc[df['protocol_type'] == 'committee']
    plenary_data = df.loc[df['protocol_type'] == 'plenary']

    #part 2
    committee_data = make_chunks2(committee_data)
    plenary_data = make_chunks2(plenary_data)
    
    #part 3
    committee_data = down_sample(committee_data,len(committee_data)-len(plenary_data))
    plenary_data = down_sample(plenary_data,len(plenary_data)-len(committee_data))

    #connect the 2 types with randomness 
    data = pd.concat([committee_data,plenary_data])
    data = data.sample(frac=1,random_state = 42)
    
    #part 4.1
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['sentence_text'])
    features = vectorizer.transform(data['sentence_text'])
   
    #part 4.2
    #our feature vector is the 100 tokens that have most occurences 
    our_feature_vector = pd.DataFrame()
    '''unique_strings = {string: index for index, string in enumerate(set(data['speaker_name']))}

    # Assign a number to each string in the list
    numbered_strings = [unique_strings[string] for string in data['speaker_name']]

    our_feature_vector['speaker_name'] = numbered_strings'''
    top_vectorize = TfidfVectorizer(max_features=100)
    top_vectorize.fit(data['sentence_text'])
    our_feature_vector = top_vectorize.transform(data['sentence_text'])
    #avarage_length = data['avarage_length'].to_numpy().reshape(len(data['avarage_length']),1)
    #our_feature_vector = np.concatenate((our_feature_vector,avarage_length) ,axis=1)

    #part 5.1
    jobs = -1
    print('BoW train validation')
    KNN = KNeighborsClassifier(5)
    SVM = svm.SVC()
    print(f'KNN with corss validation: ')
    labels = data['protocol_type']
    KNN_cross_validation = cross_val_predict(KNN,features,labels,cv=10,n_jobs=jobs)
    print(sklearn.metrics.classification_report(labels, KNN_cross_validation))
    #print ("score -"+str(cross_val_score(KNN,features,data['protocol_type'],cv=10,n_jobs=-1).mean()))


    print(f'SVM with corss validation: ')
    SVM_cross_validation = cross_val_predict(SVM,features,labels,cv=10,n_jobs=jobs)

    print(sklearn.metrics.classification_report(labels, SVM_cross_validation))
    #print (cross_val_score(SVM,features,data['protocol_type'],cv=10,verbose=2).mean())


    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
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
    KNN = KNeighborsClassifier(3)
    SVM = svm.SVC()
    print(f'Our KNN with corss validation: ')
    KNN_cross_validation = cross_val_predict(KNN,our_feature_vector,labels,cv=10,n_jobs=jobs)
    #print(sklearn.metrics.classification_report(labels, KNN_cross_validation))

    print(f'our SVM with corss validation: ')
    SVM_cross_validation = cross_val_predict(SVM,our_feature_vector,labels,cv=10,n_jobs=jobs)
    #print(sklearn.metrics.classification_report(labels, SVM_cross_validation))


    X_train, X_test, y_train, y_test = train_test_split(our_feature_vector, labels, test_size=0.1, random_state=42)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)

    y_pred = KNN.predict(X_test)
    print(f'our KNN with split:')
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    y_pred = SVM.predict(X_test)
    print(f'our SVM with split: ')
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