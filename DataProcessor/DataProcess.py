data_Path = '../data'
test_Path = '/test'
train_Path = '/train'
valid_Path = '/valid'

def test_data_process():
    append_path = valid_Path
    test_in_path = data_Path+append_path+'/sum'
    new_test_in_path = data_Path+append_path+'/new_sum'
    count = 0
    f_write = open(new_test_in_path,'a')
    with open(test_in_path) as file:
        new_str = ''
        for line in file.readlines():
            if count%2 == 0:
                new_str = ''
                new_str = new_str + line.replace('\n','') + ' . '
            else:
                new_str+=line
                f_write.write(new_str)
            count+=1
    f_write.close()

if __name__ == '__main__':
    test_data_process()