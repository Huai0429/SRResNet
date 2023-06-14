import os
def create_list(path):
    training_files = [os.path.join(path+'LR', file) for file in os.listdir(path+'LR')]
    testing_files = [os.path.join(path+'HR', file) for file in os.listdir(path+'HR')]
    print('train set length :', len(training_files))
    print('test set length :', len(testing_files))
    f=open('training.txt','w')
    for i,img in enumerate(training_files):
        img_path="./Data/LR/"+str(i)+".png"
        f.write(img_path+'\n')
    f.close()

    f=open('testing.txt','w')
    for i,img in enumerate(testing_files):
        img_path="./Data/HR/"+str(i)+".png"
        f.write(img_path+'\n')
    f.close()