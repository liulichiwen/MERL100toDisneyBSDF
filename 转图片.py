#-*- coding:utf-8 -*-
from PIL import Image
import re

x = 1024 #x坐标  通过对txt里的行数进行整数分解
y = 1024 #y坐标  x*y = 行数
o = 2000

im = Image.new("RGB",(x,y))#创建图片
file = open('pink-jasper-color.txt') #打开rbg值文件

#通过一个个rgb点生成图片
for i in range(0,x):
    for j in range(0,y):
        line = file.readline().replace('[','').replace(']','')#获取一行
        rgb = line.split(",")#分离rgb
        #print(type(rgb[0]))
        im.putpixel((j,i),(int(float(rgb[0])*o),int(float(rgb[1])*o),int(float(rgb[2])*o)))#rgb转化为像素
im.show()
