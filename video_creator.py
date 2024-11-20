import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil

# Directory zip files 
base = "D:/NST/panda"
secondary = base+'/images'

def extract_all(last_img=''):
    """
    Extract all files in the same directory in images folder

    param: 
      last_img: Last image if it is not zipped
      
    """
    for zipped in os.listdir(base):
      if zipped[-3:] == "zip":
        with zipfile.ZipFile(base+"/"+zipped, 'r') as zip_ref:
          zip_ref.extractall(secondary)
    if last_img != '':
      shutil.copy(last_img,secondary)

def reset():
  """
  Delete all files in a images folder directory in the base
  """
  for i in os.listdir(secondary):
      os.remove(secondary + '/' + i)
  os.rmdir(secondary)


def create_video(video_name = 'result.mp4'):
  """
  Create a video from images folder in the base
  """

  def extractOrder(imgPath):
    return imgPath[:imgPath[6:].find('-')+6]
  images = os.listdir(f'{base}/images')
  image_order = list(map(extractOrder, images))


  sample = cv2.imread(f'{secondary}/{images[0]}')

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(base + '/' + video_name, fourcc, 20, (sample.shape[1], sample.shape[0]))
  for i in range(0,50000,500):
    image = images[image_order.index(f'image-{i}')]
    img = cv2.imread(f'{secondary}/'+image)
    out.write(img)

  for order in range(50000,100001,500):
    image = images[image_order.index(f'image-{order}')]
    img = cv2.imread(f'{secondary}/'+image)
    out.write(img)
  out.release()




# if __name__ == '__main__':
#   extract_all()
#   create_video()
#   reset()