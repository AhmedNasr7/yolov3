import glob
import argparse
import os
from PIL import ImageTk , Image
parser = argparse.ArgumentParser(description='script to convert labels from retina form to darknet form')
parser.add_argument('--txt_file', type=str, help='type of datasets', default="./annotations/annotations.txt")
parser.add_argument('--imgs_dir', type=str, help='type of datasets', default="")
parser.add_argument('--new_dir', type=str, help='type of datasets', default="./labels/")
args = parser.parse_args();

txt_file=args.txt_file;
direc=args.new_dir;
imgs_dir=args.imgs_dir;

if(imgs_dir!="" and imgs_dir[-1]!='/'):
    imgs_dir+='/';
if(direc[-1]!='/'):
    direc+='/';


files = glob.glob(direc+'*.txt')
for f in files:
    os.remove(f)

file=open(txt_file,'r');
content=""
for line in file:
    content+=line+'\n';
file.close();
content=content.split('\n');
for line in content:
    line.strip();
    tokens=line.split(' ');
    if(len(tokens)<6):
        continue;
    
    name=tokens[0];
    name=name.split('/')[-1];
    name=name[:-4]+'.txt'
    img_name=name[:-4]+'.jpg'
    img=Image.open(imgs_dir+img_name)
    img_w,img_h=img.size
        
    xmin=float(tokens[1]);
    ymin=float(tokens[2]);
    xmax=float(tokens[3]);
    ymax=float(tokens[4]);
    classNum=tokens[5];
    

    x=str((xmin+xmax)/2/img_w);
    y=str((ymin+ymax)/2/img_h);
#    width=str(abs(ymin-ymax)/img_h);            ## wrong one , but we trained on it
#    height=str(abs(xmin-xmax)/img_w);           ## wrong one , but we trained on it
    width=str(abs(xmax-xmin)/img_w);           ## correct one , but we haven't trained on it
    height=str(abs(ymax-ymin)/img_h);          ## correct one , but we haven't trained on it
    

    new_file=open(direc+name,'a')
    new_file.write(classNum+' '+x+' '+y+' '+width+' '+height+'\n')
    new_file.flush();
    new_file.close();

