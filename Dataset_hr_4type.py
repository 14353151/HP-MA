import scipy.sparse as sp
import numpy as np
import random
class Dataset(object):
   
    def __init__(self, path):
       
        self.types = {'u' : 1, 'l' : 2}
        
        self.trainMatrix ,nt= self.load_rating_file_as_matrix(path + "user_artist.dat",path + "user_artist.dat")
        self.num_users, self.num_items = self.trainMatrix.shape[0], self.trainMatrix.shape[1]
        self.user_item_map, self.item_user_map, self.train,self.testRatings, self.testNegatives = self.load_rating_file_as_map(path + "user_artist.dat",nt)
        
        
        #self.testRatings, self.testNegatives= self.load_rating_file_as_list(path + "test_tensor.txt",nt)
        #self.testNegatives = self.load_negative_file(path + "test_negative.txt",self.testRatings)
        
        self.user_feature, self.item_feature = self.load_feature_as_map(path+'FM_bpr_train_user_4.txt', path+'FM_bpr_train_item_4.txt')
        self.fea_size = len(self.user_feature[1])
        
        
        
        self.path_ulul, self.ulul_path_num, self.ulul_jump = self.load_path_as_map(path + 'fm_train_ulul_limit10_bpr2500_knn_hr_3sim_3_top5.txt')
        self.path_ulll, self.ulll_path_num, self.ulll_jump = self.load_path_as_map(path + 'fm_train_ulll_limit10_bpr2500_knn_hr_3sim_3_top5.txt')
        self.path_uuul, self.uuul_path_num, self.uuul_jump = self.load_path_as_map(path + 'fm_train_uuul_limit10_bpr2500_knn_hr_3sim_3_top5.txt')
        self.path_uull, self.uull_path_num, self.uull_jump = self.load_path_as_map(path + 'fm_train_uull_limit10_bpr2500_knn_hr_3sim_3_top5.txt')
        '''
        
        self.path_ulul, self.ulul_path_num, self.ulul_jump = self.load_path_as_map(path + 'ulul_final_max.txt')
        self.path_ulll, self.ulll_path_num, self.ulll_jump = self.load_path_as_map(path + 'ulll_final_max.txt')
        self.path_uuul, self.uuul_path_num, self.uuul_jump = self.load_path_as_map(path + 'uuul_final_max.txt')
        
        self.path_ulul1, self.ulul1_path_num, self.ulul1_jump = self.load_path_as_map(path + 'ulul_final22.txt')
        self.path_ulll1, self.ulll1_path_num, self.ulll1_jump = self.load_path_as_map(path + 'ulll_final22.txt')
        self.path_uuul1, self.uuul1_path_num, self.uuul1_jump = self.load_path_as_map(path + 'uuul_final22.txt')
        '''
    def load_rating_file_as_list(self, filename,trainM):
        ratingList = []
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            tmp = np.zeros((self.num_users+1, self.num_items+1))
            while line != None and line != "":
                arr = line.split("\t")
                tmp[int(arr[0])][int(arr[1])] = 1
                line = f.readline()
            for i in range (1,self.num_users):
                tmp_list = []
                tmp_list1 = []
                tmp_list.append(i)
                for j in range(1,self.num_items):
                    if tmp[i][j] == 1 and trainM[i][j] == 0:
                        tmp_list.append(j)  
                    if trainM[i][j] == 0 and tmp[i][j] == 0:
                        tmp_list1.append(j)
                ratingList.append(tmp_list)
                negativeList.append(tmp_list1)
        return ratingList,negativeList
    
    def load_rating_file_as_map(self, filename,trainM):
        user_item_map = {}
        item_user_map = {}
        train = []
        popularity_dict = {}
        max_i = 0
        testRatings = []
        testNegatives = []
        for i in range (1,self.num_users):
            tmp_c = 100
            tmp_n = []
            tmp_p = []
            tmp_p.append(i)
            count = 0
            for j in range(1,self.num_items):
                if trainM[i][j] == 1 and count == 0:
                    count = 1
                    tmp_p.append(j)
                    tmp_n.append(j)
                elif trainM[i][j] == 1 and count == 1:
                    train.append([i, j])
                    max_i = max(max_i, i)
                    if i not in user_item_map:
                        user_item_map[i] = {}
                    if j not in item_user_map:
                        item_user_map[j] = {}
                    if j not in popularity_dict:
                        popularity_dict[j] = 0
                    user_item_map[i][j] = 1.0
                    item_user_map[j][i] = 1.0
                    popularity_dict[j] += 1
            while tmp_c > 0:
                tmp_c -= 1
                k = random.randint(1, self.num_items-1)
                while trainM[i][k] == 1:
                    k = random.randint(1, self.num_items-1)
                tmp_n.append(k)
            testNegatives.append(tmp_n)
            testRatings.append(tmp_p)
        item_popularity = [0] * max_i
        
        return user_item_map, item_user_map, train,testRatings,testNegatives

    def load_rating_file_as_matrix(self, filename, filename1):
      
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        nt = np.zeros((num_users+1, num_items+1))
        train_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                train_list.append([user, item])
                mat[user, item] = 1.0
                nt[user][item] = 1
                line = f.readline()    
                
        with open(filename1, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                train_list.append([user, item])
                mat[user, item] = 1.0
                nt[user][item] = 1
                line = f.readline()    
                
        return mat ,nt

    def load_feature_as_map(self, user_fea_file, item_fea_file):
        user_feature = np.zeros((self.num_users, 8))
        item_feature = np.zeros((self.num_items, 8))
        
        with open(user_fea_file) as infile:
            t = 1
            for line in infile.readlines():
                arr = line.split('\t')
                if t <= 2321:
                    for j in range(len(arr)-1):
                        user_feature[t][j] = float(arr[j])
                    t += 1    
            
            
        with open(item_fea_file) as infile:
            t = 1
            for line in infile.readlines():
                arr = line.split('\t')
                #item_feature[i] = list()
                
                for j in range(len(arr)-1):
                    item_feature[t][j] = float(arr[j])
                t += 1    
       
        return user_feature, item_feature
        
    def load_path_as_map1(self, filename):
        print (filename) 
        path_dict = {}
        path_dict_num = {}
        path_num = 5
        jump = 4
        length = 2
        ctn = 0
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                line = infile.readline()
                ctn += 1
        print (ctn, path_num, jump, length)
        with open(filename) as infile:
            line = infile.readline()

            while line != None and line != "":
                u_index = 0
                l_index = 0
                tmp_list1 = line.split(',')
                ul_list = tmp_list1[0].split('\t')
                u_index = int(ul_list[0])
                l_index = int(ul_list[1])
                tmp_list = tmp_list1[1].split('\t')
                path_dict[(u_index, l_index)] = []
                path_dict_num[(u_index, l_index)] = []
                path_dict_num[(u_index, l_index)].append(len(tmp_list)-1)
                for tt in range(0, len(tmp_list)-1):
                    tmp = tmp_list[tt].split('-')
                    node_list = []
                    t = 0
                    for node in tmp:
                        
                        index = int(node[1:])
                        node_list.append([self.types[node[0]], index])
                        t += 1
                    path_dict[(u_index, l_index)].append(node_list)
                line = infile.readline()        
        return path_dict, path_num, jump,path_dict_num
    
    def load_path_as_map(self, filename):
        print (filename) 
        path_dict = {}
        path_dict_num = {}
        path_num = 5
        jump = 4
        length = 2
        ctn = 0
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                line = infile.readline()
                ctn += 1
        print (ctn, path_num, jump, length)
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                
                u_index = 0
                l_index = 0
               
                tmp_list = line.split('\t')
                for tt in range(0, len(tmp_list)-1):
                    tmp = tmp_list[tt].split('-')
                    node = tmp[0]
                    u_index = int(node[1:])
                    node = tmp[3]
                    l_index = int(node[1:])
                path_dict[(u_index, l_index)] = []
                path_dict_num[(u_index, l_index)] = []
                #path_dict_num[(u_index, l_index)].append(len(tmp_list)-1)
                for tt in range(0, len(tmp_list)-1):
                    tmp = tmp_list[tt].split('-')
                    node_list = []
                    t = 0
                    for node in tmp:
                        index = int(node[1:])
                        node_list.append([self.types[node[0]], index])
                        t += 1
                    path_dict[(u_index, l_index)].append(node_list)
                line = infile.readline()       
        return path_dict, path_num, jump
	

if __name__ == '__main__':
    dataset = Dataset('data/')
    

