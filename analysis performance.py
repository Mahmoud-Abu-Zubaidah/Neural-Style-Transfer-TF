import os
import zipfile
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
base = "D:/NST/G3"
secondary = base+'/images'

def extract_all():
    """
    Extract all files in the same directory in images folder

    param: 
      last_img: Last image if it is not zipped
      
    """
    for zipped in os.listdir(base):
      if zipped[-3:] == "zip":
        with zipfile.ZipFile(base+"/"+zipped, 'r') as zip_ref:
          zip_ref.extractall(secondary)


def reset():
  """
  Delete all files in a images folder directory in the base
  """
  for i in os.listdir(secondary):
      os.remove(secondary + '/' + i)
  os.rmdir(secondary)

def extract_cost(name:str):
   name = name[:-4].split('-')
   epoch = int(name[1])
   cost = round(float(name[2][5:]),3)
   return [epoch, cost]


def create_plot():
    epochs = []
    costs = []

    images = os.listdir(secondary)
    # Take every N-th image
    for i in range(10, len(images), 500):
        img = images[i]
        epoch, cost = extract_cost(img)
        epochs.append(epoch)
        costs.append(cost)

    result = pd.DataFrame({'epoch': epochs, 'cost': costs})

    plt.plot('epoch', 'cost',data=result,c= '#FC4C02',ls = '--',)
    plt.xlabel('Epochs',c='gray')
    plt.ylabel('Costs',c='red')
    plt.savefig('epoch_cost_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()

create_plot()