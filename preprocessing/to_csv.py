import csv
import pandas as pd
# converting words_label.txt into a csv file that's labels.csv and adding two extra columns to it for words having more than one division of labels like MP S sub-label1 = MP sub-label2 is S...
with open('words_label.txt', 'r') as input_file:
    stripped = (line.strip() for line in input_file)
    lines = (line.split() for line in stripped if line)
    with open('labels.csv', 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(('word-id','result','gray_level','x','y','w','h','grammar_tag','label','extra_1','extra_2'))
        writer.writerows(lines)

'''
    creating a csv file from labels.csv that deletes the erroneous data and keeps the clean data into labels2.csv
    labels2.csv has only two columns the word ids and the label of the words.
'''
df = pd.read_csv('labels.csv', sep=',')
df = df[df.result == "ok"]
df = df[df.extra_1.isnull() & df.extra_2.isnull()]
df = df[['word-id', 'label']]
'''we cant do count here coz when we do we dont get count of the labels but it counts the labels, ie
    1 is move .... till the last label...'''
df.to_csv('labels2.csv', sep=',', index=False)


''' Counting the number of occurences of all the labels. First converting into pandas df and then counting
    input labels2.csv output measure.csv
    measure.csv contains two columns the label and the number of words of that label.
    '''
mdf = pd.read_csv('labels2.csv', sep=',')
count = mdf['label'].value_counts()
'''count is a pandas series frame, converting it into a pandas data frame'''
count = pd.DataFrame({'label':count.index, 'count':count.values})
'''getting all the words having length 3 or more'''
count = count[count.label.apply(len) >= 3]
count = count.head(225)
count.to_csv('measure.csv') 



'''mdf = pd.read_csv('labels2.csv', sep=',')
count = pd.read_csv('measure.csv')
words_list = count['label'].tolist()
#min_occur = count['count'].min()
#print (min_occur)
#print (count)
#print (words_list)
ids = {}
for word in words_list:
    temp_df = pd.DataFrame()
    temp_df = mdf.loc[mdf['label'] == word]
    temp_list = []
    temp_list = temp_df['word-id'].tolist()
    temp_list = temp_list[:25]
    ids.update({word : temp_list})
    train_word_dir = os.path.join("db","train",word)
    if not os.path.exists(train_word_dir):
        os.makedirs(train_word_dir)
    for image_id in temp_list:
        image_id_list = image_id.split('-')
        dir_name = image_id_list[0]
        sub_dir_name = image_id_list[0]+'-'+image_id_list[1]
        image_name = image_id+'.png'
        image_path = os.path.join("words",dir_name,sub_dir_name,image_name)
        shutil.copy(image_path,train_word_dir)


#print (ids)'''

# labels.csv converting .txt to csv with the given labels
#labels2.csv taking only 2 columns that is wor-d and labels
#measure.csv getting word frequency of words having length of 3 or more
