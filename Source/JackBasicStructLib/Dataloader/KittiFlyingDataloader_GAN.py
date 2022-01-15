# -*- coding: utf-8 -*-
from JackBasicStructLib.ImgProc.ImgHandler import *
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.ImgProc.DataAugmentation import *
import matplotlib.pyplot as plt
import cv2


# output file setting
DEPTH_DIVIDING = 255.0


class KittiFlyingDataloader(object):
    def __init__(self):
        super(KittiFlyingDataloader, self).__init__()
        self.imgL = None
        self.imgR = None
        pass

    def SaveKITTITestData(self, args, saveFormat, img, num):
        path = self.__GenerateOutImgPath(args.resultImgDir, saveFormat, args.imgType, num)
        img = self.__DepthToImgArray(img)
        self.__SavePngImg(path, img)

    def CropTestImg(self, img, top_pad, left_pad):
        if top_pad > 0 and left_pad > 0:
            img = img[top_pad:, : -left_pad, :]
        elif top_pad > 0:
            img = img[top_pad:, :, :]
        elif left_pad > 0:
            img = img[:, :-left_pad, :]
        return img

    def GetBatchImage(self, args, randomlist, num, isVal=False):
        for i in range(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]
            RGB_IMG_NUM = 788 # kitti dataset
            rgb_num = idNum % RGB_IMG_NUM
            INPUT_IMG_NUM = 4400 # driving dataset
            input_num = idNum % INPUT_IMG_NUM + 394

            if isVal == False:
                edge_imgL, edge_imgR, intput_rgb_imgL, intput_rgb_imgR, rgb_imgL, rgb_imgR, input_edge_y_L, input_edge_y_R, label_rgb_img = self.__RandomCropRawImage(
                    args, input_num, rgb_num)       # get img
            else:
                edge_imgL, edge_imgR, intput_rgb_imgL, intput_rgb_imgR, rgb_imgR, input_edge_y_R, label_rgb_img = self.__ValRandomCropRawImage(
                    args, input_num, rgb_num)       # get img
            
            if i == 0:
                edge_imgLs = edge_imgL
                edge_imgRs = edge_imgR
                intput_rgb_imgLs = intput_rgb_imgL
                intput_rgb_imgRs = intput_rgb_imgR
                rgb_imgLs = rgb_imgL
                rgb_imgRs = rgb_imgR
                input_edge_y_Ls = input_edge_y_L
                input_edge_y_Rs = input_edge_y_R
                label_rgb_imgs = label_rgb_img
            else:
                edge_imgLs = np.concatenate((edge_imgLs, edge_imgL), axis=0) 
                edge_imgRs = np.concatenate((edge_imgRs, edge_imgR), axis=0)
                intput_rgb_imgLs = np.concatenate((intput_rgb_imgLs, intput_rgb_imgL), axis=0)
                intput_rgb_imgRs = np.concatenate((intput_rgb_imgRs, intput_rgb_imgR), axis=0)
                rgb_imgLs = np.concatenate((rgb_imgLs, rgb_imgL), axis=0)
                rgb_imgRs = np.concatenate((rgb_imgRs, rgb_imgR), axis=0)
                input_edge_y_Ls = np.concatenate((input_edge_y_Ls, input_edge_y_L), axis=0)
                input_edge_y_Rs = np.concatenate((input_edge_y_Rs, input_edge_y_R), axis=0)
                label_rgb_imgs = np.concatenate((label_rgb_imgs, label_rgb_img), axis=0)  

        return edge_imgLs, edge_imgRs, intput_rgb_imgLs, intput_rgb_imgRs, rgb_imgLs, rgb_imgRs, input_edge_y_Ls, input_edge_y_Rs, label_rgb_imgs

    def GetBatchTestImage(self, args, randomlist, num, isVal=False):
        top_pads = []
        left_pads = []
        names = []
        for i in range(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]

            edge_imgL, edge_imgR, imgL, imgR, rgb_imgL, rgb_imgR, edge_img_y_L, edge_img_y_R, top_pad, left_pad, name = self.__GetPadingTestData(
                args, idNum)       # get img

            top_pads.append(top_pad)
            left_pads.append(left_pad)
            names.append(name)
            if i == 0:
                edge_imgLs = edge_imgL
                edge_imgRs = edge_imgR
                imgLs = imgL
                imgRs = imgR
                rgb_imgLs = rgb_imgL
                rgb_imgRs = rgb_imgR
                edge_img_y_Ls = edge_img_y_L
                edge_img_y_Rs = edge_img_y_R
            else:
                edge_imgLs = np.concatenate((edge_imgLs, edge_imgL), axis=0)
                edge_imgRs = np.concatenate((edge_imgRs, edge_imgR), axis=0)
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)
                rgb_imgLs = np.concatenate((rgb_imgLs, rgb_imgL), axis=0)
                rgb_imgRs = np.concatenate((rgb_imgRs, rgb_imgR), axis=0)
                edge_img_y_Ls = np.concatenate((edge_img_y_Ls, edge_img_y_L), axis=0)
                edge_img_y_Rs = np.concatenate((edge_img_y_Rs, edge_img_y_R), axis=0)

        return edge_imgLs, edge_imgRs, imgLs, imgRs, rgb_imgLs, rgb_imgRs, edge_img_y_Ls, edge_img_y_Rs, top_pads, left_pads, names

    def __GenerateOutImgPath(self, dirPath, filenameFormat, imgType, num):
        path = dirPath + filenameFormat % num + imgType
        return path

    def __DepthToImgArray(self, img):
        img = np.array(img)
        img = ((img + 1) * float(DEPTH_DIVIDING) / 2).astype(np.uint8)
        return img

        # save the png file
    def __SavePngImg(self, path, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    def __ReadRandomPfmGroundTrue(self, path, x, y, w, h):
        # flying thing groundtrue
        imgGround, _ = ReadPFM(path)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        return imgGround

    def __ReadRandomGroundTrue(self, path, x, y, w, h):
        # kitti groundtrue
        img = Image.open(path)
        imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        return imgGround

    def __ReadData(self, args, pathL, pathR, label_path, rgb_path_L, rgb_path_R):
        # Flying Things and Kitti
        w = args.corpedImgWidth
        h = args.corpedImgHeight

        # get the img, the random crop
        imgL = ReadImg(pathL) # source left images
        imgR = ReadImg(pathR) # source right images
        rgb_imgL = ReadImg(rgb_path_L) # target left images
        rgb_imgR = ReadImg(rgb_path_R) # target right images

        # random crop
        x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
        imgL = ImgSlice(imgL, x, y, w, h) 
        imgR = ImgSlice(imgR, x, y, w, h)
        x1, y1 = RandomOrg(rgb_imgL.shape[1], rgb_imgL.shape[0], w, h)
        rgb_imgL = ImgSlice(rgb_imgL, x1, y1, w, h)
        rgb_imgR = ImgSlice(rgb_imgR, x1, y1, w, h)
        
        # get edge
        edge_imgL = Sobel_Edge(imgL)
        # nomalize
        edge_imgL = Standardization(edge_imgL)
        edge_imgR = Sobel_Edge(imgR)
        edge_imgR = Standardization(edge_imgR)
        edge_img_y_L = Sobel_Edge(rgb_imgL)
        edge_img_y_L = Standardization(edge_img_y_L)
        edge_img_y_R = Sobel_Edge(rgb_imgR)
        edge_img_y_R = Standardization(edge_img_y_R)

        imgL = imgL / 127.5 - 1
        imgR = imgR / 127.5 - 1
        rgb_imgL = rgb_imgL / 127.5 -1
        rgb_imgR = rgb_imgR / 127.5 -1

        file_type = os.path.splitext(label_path)[-1]

        # get groundtrue
        if file_type == ".png":
            imgGround = self.__ReadRandomGroundTrue(label_path, x, y, w, h)
        else:   
            imgGround = self.__ReadRandomPfmGroundTrue(label_path, x, y, w, h)

        edge_imgL = np.expand_dims(edge_imgL, axis=0) # source edge images
        edge_imgL = np.expand_dims(edge_imgL, axis=3)
        edge_imgR = np.expand_dims(edge_imgR, axis=0) 
        edge_imgR = np.expand_dims(edge_imgR, axis=3)
        imgL = np.expand_dims(imgL, axis=0) # source rgb images
        imgR = np.expand_dims(imgR, axis=0)
        imgGround = np.expand_dims(imgGround, axis=0) # source disparity gt
        rgb_imgL = np.expand_dims(rgb_imgL, axis=0) # target rgb images
        rgb_imgR = np.expand_dims(rgb_imgR, axis=0)
        edge_img_y_L = np.expand_dims(edge_img_y_L, axis=0) # target edge images
        edge_img_y_L = np.expand_dims(edge_img_y_L, axis=3)
        edge_img_y_R = np.expand_dims(edge_img_y_R, axis=0)
        edge_img_y_R = np.expand_dims(edge_img_y_R, axis=3)
        

        return edge_imgL, edge_imgR, imgL, imgR, rgb_imgL, rgb_imgR, edge_img_y_L, edge_img_y_R, imgGround

    def __RandomCropRawImage(self, args, input_num, rgb_num):
        # Get path
        pathL = GetPath(args.trainListPath, 2*input_num+1)
        pathR = GetPath(args.trainListPath, 2*(input_num + 1))
        pathGround = GetPath(args.trainLabelListPath, input_num + 1)
        rgb_path_L = GetPath(args.trainListPath, 2*rgb_num + 1)
        rgb_path_R = GetPath(args.trainListPath, 2*(rgb_num + 1))
        
        edge_imgL, edge_imgR, input_rgb_imgL, input_rgb_imgR, rgb_imgL, rgb_imgR, input_edge_y_L, input_edge_y_R, label_rgb_img = self.__ReadData(
                                                            args, pathL, pathR, pathGround, rgb_path_L, rgb_path_R)
   
        return edge_imgL, edge_imgR, input_rgb_imgL, input_rgb_imgR, rgb_imgL, rgb_imgR, input_edge_y_L, input_edge_y_R, label_rgb_img



    # Val Flying Things and Kitti
    def __ValRandomCropRawImage(self, args, input_num, label_num):
        # Get path
        img_path = GetPath(args.valListPath, input_num+1)
        label_num = GetPath(args.valLabelListPath, input_num+1)
        edge_img, intput_rgb_img, label_rgb_img = self.__ReadData(args, pathL, pathR, label_path)

        return edge_img, intput_rgb_img, label_rgb_img

   # Padding Img, used in testing
    def __GetPadingTestData(self, args, input_num):
        pathL = GetPath(args.testListPath, 2*input_num+1)
        pathR = GetPath(args.testListPath, 2*(input_num + 1))

        imgL = ReadImg(pathL)
        imgR = ReadImg(pathR)

        edge_imgL = Sobel_Edge(imgL)
        edge_imgL = Standardization(edge_imgL)
        edge_imgR = Sobel_Edge(imgR)
        edge_imgR = Standardization(edge_imgR)

        imgL = imgL / 127.5 - 1
        imgR = imgR / 127.5 - 1

        edge_imgL = np.expand_dims(edge_imgL, axis=0)
        edge_imgL = np.expand_dims(edge_imgL, axis=3)
        edge_imgR = np.expand_dims(edge_imgR, axis=0)
        edge_imgR = np.expand_dims(edge_imgR, axis=3)
        imgL = np.expand_dims(imgL, axis=0)
        imgR = np.expand_dims(imgR, axis=0)

        # pading size
        top_pad = args.padedImgHeight - edge_imgL.shape[1]
        left_pad = args.padedImgWidth - edge_imgL.shape[2]


        # pading
        edge_imgL = np.lib.pad(edge_imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                                         (0, 0)), mode='constant', constant_values=0)
        edge_imgR = np.lib.pad(edge_imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                                         (0, 0)), mode='constant', constant_values=0)
        imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                                       (0, 0)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                                       (0, 0)), mode='constant', constant_values=0)

        name = None

        return edge_imgL, edge_imgR, imgL, imgR, imgL, imgR, edge_imgL, edge_imgR, top_pad, left_pad, name
