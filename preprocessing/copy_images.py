import csv
import pandas as pd
import os
import shutil
from PIL import Image, ImageOps
import cv2

def find_max_min_size(filenames):
    sizes = [Image.open(f, 'r').size for f in filenames]
    return max(sizes), min(sizes)

def to_gray_scale(image_paths):
    for im_pth in image_paths:
        print(im_pth)
        im = Image.open(im_pth).convert('L')
        os.remove(im_pth)
        im.save(im_pth)

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

'''
def copy_images(image_list,dest_dir): # ex dest_dir = db/train/the
    for image_id in image_list:
        image_id_list = image_id.split('-')
        dir_name = image_id_list[0]
        sub_dir_name = image_id_list[0]+'-'+image_id_list[1]
        image_name = image_id+'.png'
        image_path = os.path.join("words",dir_name,sub_dir_name,image_name)
        shutil.copy(image_path,dest_dir)

mdf = pd.read_csv('labels2.csv', sep=',')
count = pd.read_csv('measure.csv')
words_list = count['label'].tolist()
words = words_list[:51]+words_list[150:]
#print (words_list)
words.remove("West")
print (len(words))
print (words)
#exit()
for word in words:
    temp_df = pd.DataFrame()
    temp_df = mdf.loc[mdf['label'] == word]

    temp_list = []
    temp_list = temp_df['word-id'].tolist()
    
    temp_train_list = temp_list[:150]
    #temp_val_list = temp_list[100:110]
    #temp_test_list = temp_list[110:115]
    
    train_word_dir = os.path.join("db",word) #train in between word and db
    val_word_dir = os.path.join("db","validate",word)
    test_word_dir = os.path.join("db","test",word)
    
    make_dirs(train_word_dir)
    #make_dirs(val_word_dir)
    #make_dirs(test_word_dir)
    
    copy_images(temp_train_list,train_word_dir)
    #copy_images(temp_val_list,val_word_dir)
    #copy_images(temp_test_list,test_word_dir)

image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk("db") for f in filenames if os.path.splitext(f)[1] == '.png']

max_size, min_size = find_max_min_size(image_paths)
desired_size = max(max_size)
#resizing the images by padding white spaces and saving them as their original names
for im_pth in image_paths:
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    os.remove(im_pth) #removing the old image
    new_im = Image.new("RGB", (desired_size, desired_size), color=16777215) #color value is white to pad images with white background
    new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
    new_im.save(im_pth) #saving the new image with the original name

#reducing the image size
for im_pth in image_paths:
    img = cv2.imread(im_pth,0)
    resized = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    os.remove(im_pth)
    cv2.imwrite(im_pth,resized)

to_gray_scale(image_paths)
'''

#before running this edited measure.csv such that repeated words like "the" , "The" and "you" , "You" were reomved and only one of these instances have been retained randomly. this was done manually. some words dont have enough testing data. see how that can be handled. next step would be to resize the image and build a model and train on it.

#took 100 words into 150 samples and padding white space to it and resizing it. Next have to convert the images into grayscale image Then have to divide them into train and test. can be done directly in line 71 using mode L instead of RGB. folders which contains every word's subfolder. train contains 100 samples of that word and test 50 samples.

#converted to grayscale using the function

def copy_images2(image_list,dest_dir,src_dir): # ex dest_dir = db/train/the
    for image in image_list:
        image_path = os.path.join("db",src_dir,image)
        shutil.copy(image_path,dest_dir)

words = os.listdir("db")
words.remove('.DS_Store')
# creating folders in the database folder for training and testing folders. creating empty subfolders whose names are the names of the words
for word in words:
    train_word_dir = os.path.join("database","train",word)
    test_word_dir = os.path.join("database","test",word)
    make_dirs(train_word_dir)
    make_dirs(test_word_dir)
# copying imags into the empty directories
dirs = [ dir for dir in os.listdir("db") if os.path.isdir(os.path.join("db", dir)) ]
for dir in dirs:
    files = [f for f in os.listdir(os.path.join("db",dir)) if f.endswith(".png")]
    train_list = files[:100]
    test_list = files[100:150]
    train_dir = os.path.join("database","train",dir)
    test_dir = os.path.join("database","test",dir)
    copy_images2(train_list,train_dir,dir)
    copy_images2(test_list,test_dir,dir)

'''
    Find the number of images in the corresponding word directoriess
    
    >>> lenlist = []
    >>> for r,d,f in os.walk("db"):
    ...     for dir in d:
    ...             files = os.listdir(os.path.join("db",dir))
    ...             length = len(files)
    ...             lenlist.append(length)
    '''
