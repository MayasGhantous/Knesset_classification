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
    sorted_data = data.sort_values(by = ['protocol_name','speaker_name'])
    i = 0 
    new_data = pd.DataFrame(columns = sorted_data.columns)
    while i < len(sorted_data):
        protocol_type = sorted_data.iloc[i]['protocol_type']
        cuurent_number =sorted_data.iloc[i]['knesset_number'] 
        protocol_name =sorted_data.iloc[i]['protocol_name']
        sentence = sorted_data.iloc[i]['sentence_text']
        speaker = sorted_data.iloc[i]['speaker_name']
        i+=1
        j = 1
        while j<5:
            if i >= len(sorted_data):
                break
            currnet_protocol_name =sorted_data.iloc[i]['protocol_name']
            sentence += " " +sorted_data.iloc[i]['sentence_text']
            current_speaker = sorted_data.iloc[i]['speaker_name']
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
        if j != 5:
            continue
        new_row = pd.DataFrame({'protocol_name':[protocol_name],'knesset_number':[cuurent_number],'protocol_type':[protocol_type],'speaker_name':[speaker],'sentence_text':[sentence]})
        new_data = pd.concat([new_data,new_row],ignore_index=True)


    return new_data

def down_sample(data,N):
    if N<=0:
        return data
    number_list = random.sample(range(len(data)),k=N)
    return data.drop(number_list)
    


if __name__ == '__main__':
    # part 1
    df = pd.read_csv('knesset_corpus.csv',index_col=None)
    committee_data = df.loc[df['protocol_type'] == 'committee']
    plenary_data = df.loc[df['protocol_type'] == 'plenary']

    #part 2
    committee_data = make_chunks(committee_data)
    plenary_data = make_chunks(plenary_data)

    #part 3
    committee_data = down_sample(committee_data,len(committee_data)-len(plenary_data))
    plenary_data = down_sample(plenary_data,len(plenary_data)-len(committee_data))

    data = pd.concat([committee_data,plenary_data])

    #part 4
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data['sentence_text'])

    #part 5
    KNN = KNeighborsClassifier(1000)
    SVM = svm.SVC()
    print (cross_val_score(KNN,features,data['protocol_type'],cv=10))
    print (cross_val_score(SVM,features,data['protocol_type'],cv=10))


    X_train, X_test, y_train, y_test = train_test_split(features, data['protocol_type'], test_size=0.1, random_state=42)
    KNN.fit(X_train,y_train)
    SVM.fit(X_train,y_train)

    y_pred = KNN.predict(X_test)
    print(f'KNN with split {sklearn.metrics.classification_report(y_test, y_pred)}')
    
    y_pred = SVM.predict(X_test)
    print(f'SVM with split {sklearn.metrics.classification_report(y_test, y_pred)}')

    #part 6
    with open('knesset_text_chucks.txt', 'r') as file:
        sentences = file.read().split('\n')
        prediction_porb = 0.7*SVM.predict_proba(vectorizer.fit_transform(sentences)) + 0.3 *KNN.predict_proba(vectorizer.fit_transform(sentences))
        print(prediction_porb)
        


    
    
