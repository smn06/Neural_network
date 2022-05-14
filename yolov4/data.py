import numpy as np
import random
import xml
import cv2
import os


def read(n):

    if not os.path.isfile(n):
        return None
    result = []
    with open(n, 'r') as f:
        for line in f.readlines():
            
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            result.append(line)
    return result


def id(file_n):

    iddic = {}
    contents = read(file_n)
    for i in range(len(contents)):
        iddic[str(contents[i])] = i
    return iddic


def parse_voc_xml(n, names_dict):

    
    result = []
    if not os.path.isfile(n):
        return None
    doc = xml.dom.minidom.parse(n)
    root = doc.documentElement
    size = root.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)

    objs = root.getElementsByTagName('object')
    for obj in objs:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        name_id = names_dict[name]

        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        x = (xmax + xmin) / 2.0 / width
        w = (xmax - xmin) / width
        y = (ymax + ymin) / 2.0 / height
        h = (ymax - ymin) / height

        result.append([name_id, x, y, w, h])
    return result


class Data:
    def __init__(self, vdir, vdirlist, vname, classn, batch, val, scale=True, width=608, height=608):
        self.data_dirs = [os.path.join(os.path.join(vdir, voc_dir), "JPEGImages") for voc_dir in vdirlist]  
        self.classn = classn  
        self.batch = batch
        self.val = np.asarray(val).astype(np.float32).reshape([-1, 2]) / [width, height]  
        self.scale = scale  

        self.imgs_path = []
        self.labpath = []

        self.num_batch = 0      
        self.num_imgs = 0       

        self.width = width
        self.height = height

        self.names_dict = id(vname)    

        
        self.__init_args()
    
    
    def __init_args(self):

        
        for voc_dir in self.data_dirs:
            for img_name in os.listdir(voc_dir):
                img_path = os.path.join(voc_dir, img_name)
                label_path = img_path.replace("JPEGImages", "Annotations")
                label_path = label_path.replace(img_name.split('.')[-1], "xml")
                if not os.path.isfile(img_path):
                    continue
                if not os.path.isfile(label_path):
                    continue
                self.imgs_path.append(img_path)
                self.labpath.append(label_path)
                self.num_imgs += 1        
        
        if self.num_imgs <= 0:
            raise ValueError("exception")
        
        return
        
    
    def read_img(self, img_file):

        if not os.path.exists(img_file):
            return None
        img = cv2.imread(img_file)
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        return img
    
    
    def read_label(self, label_file, names_dict):

        contents = parse_voc_xml(label_file, names_dict)  
        if not contents:
            return None, None, None

        label_y1 = np.zeros((self.height // 32, self.width // 32, 3, 5 + self.classn), np.float32)
        label_y2 = np.zeros((self.height // 16, self.width // 16, 3, 5 + self.classn), np.float32)
        label_y3 = np.zeros((self.height // 8, self.width // 8, 3, 5 + self.classn), np.float32)

        yt = [label_y3, label_y2, label_y1]
        ratio = {0: 8, 1: 16, 2: 32}

        for label in contents:
            label_id = int(label[0])
            box = np.asarray(label[1: 5]).astype(np.float32)   

            best_giou = 0
            bi = 0
            for i in range(len(self.val)):
                min_wh = np.minimum(box[2:4], self.val[i])
                max_wh = np.maximum(box[2:4], self.val[i])
                giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
                if giou > best_giou:
                    best_giou = giou
                    bi = i
            
            
            x = int(np.floor(box[0] * self.width / ratio[bi // 3]))
            y = int(np.floor(box[1] * self.height / ratio[bi // 3]))
            k = bi % 3

            yt[bi // 3][y, x, k, 0:4] = box
            yt[bi // 3][y, x, k, 4:5] = 1.0
            yt[bi // 3][y, x, k, 5 + label_id] = 1.0
        
        return label_y1, label_y2, label_y3

    
    def __get_data(self):
        if self.scale and (self.num_batch % 10 == 0):
            random_size = random.randint(10, 19) * 32
            self.width = self.height = random_size
        
        imgs = []
        laby1, laby2, laby3 = [], [], []

        count = 0
        while count < self.batch:
            curr_index = random.randint(0, self.num_imgs - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labpath[curr_index]

            img = self.read_img(img_name)
            label_y1, label_y2, label_y3 = self.read_label(label_name, self.names_dict)
            if img is None:
                continue
            if label_y1 is None:
                continue
            imgs.append(img)
            laby1.append(label_y1)
            laby2.append(label_y2)
            laby3.append(label_y3)

            count += 1

        self.num_batch += 1
        imgs = np.asarray(imgs)
        laby1 = np.asarray(laby1)
        laby2 = np.asarray(laby2)
        laby3 = np.asarray(laby3)
        
        return imgs, laby1, laby2, laby3

    
    def __next__(self):
        return self.__get_data()
