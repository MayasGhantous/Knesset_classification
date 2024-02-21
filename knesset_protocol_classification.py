import pandas as pd 
import numpy as np 
import sklearn
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import train_test_split

def make_chunks(data):
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


    return new_data

def down_sample(data,N):
    # if we want to down sample non positive number then dont do any thing
    if N<=0:
        return data
    number_list = random.sample(range(len(data)),k=N)
    return data.drop(number_list)
    


if __name__ == '__main__':
    # change the sead
    random.seed(42)
    np.random.seed(42)
    # part 1
    df = pd.read_csv('knesset_corpus.csv',index_col=None)
    committee_data = df.loc[df['protocol_type'] == 'committee']
    plenary_data = df.loc[df['protocol_type'] == 'plenary']

    #part 2
    '''committee_data = make_chunks(committee_data)
    committee_data.to_csv('committee_data.csv',index=False)
    plenary_data = make_chunks(plenary_data)
    plenary_data.to_csv('plenary_data.csv',index=False)'''

    committee_data = pd.read_csv('committee_data.csv')
    plenary_data = pd.read_csv('plenary_data.csv')
    


    #part 3
    committee_data = down_sample(committee_data,len(committee_data)-len(plenary_data))
    plenary_data = down_sample(plenary_data,len(plenary_data)-len(committee_data))

    data = pd.concat([committee_data,plenary_data])

    #part 4.1
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data['sentence_text'])

    #part 4.2
    our_feature_vector = pd.DataFrame()
    unique_strings = {string: index for index, string in enumerate(set(data['speaker_name']))}

# Assign a number to each string in the list
    numbered_strings = [unique_strings[string] for string in data['speaker_name']]

    our_feature_vector['speaker_name'] = numbered_strings
    


    #part 5.1
    print('BoW train validation')
    KNN = KNeighborsClassifier(100)
    SVM = svm.SVC()
    print (cross_val_score(KNN,features,data['protocol_type'],cv=10).mean())
    print (cross_val_score(SVM,features,data['protocol_type'],cv=10).mean())


    X_train, X_test, y_train, y_test = train_test_split(features, data['protocol_type'], test_size=0.1, random_state=42)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)
    last_model = SVM

    y_pred = KNN.predict(X_test)
    print(f'KNN with split {sklearn.metrics.classification_report(y_test, y_pred)}')
    
    y_pred = SVM.predict(X_test)
    print(f'SVM with split {sklearn.metrics.classification_report(y_test, y_pred)}')


    #part 5.2
    print('our vecotr test validation')
    KNN = KNeighborsClassifier(1)
    SVM = svm.SVC()
    print (cross_val_score(KNN,our_feature_vector,data['protocol_type'],cv=10).mean())
    print (cross_val_score(SVM,our_feature_vector,data['protocol_type'],cv=10).mean())


    X_train, X_test, y_train, y_test = train_test_split(our_feature_vector, data['protocol_type'], test_size=0.1, random_state=42)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)

    y_pred = KNN.predict(X_test)
    print(f'KNN with split {sklearn.metrics.classification_report(y_test, y_pred)}')
    
    y_pred = SVM.predict(X_test)
    print(f'SVM with split {sklearn.metrics.classification_report(y_test, y_pred)}')


    #part 6
    with open('knesset_text_chunks.txt', 'r',encoding='utf-8') as file:
        sentences = file.readlines()
        predictions = last_model.predict(sentences)
        text=''
        for prediction in predictions: 
            text+=prediction + '\n'
        with open('classification_results.txt','w') as write_file:
            write_file.write(text)

        

