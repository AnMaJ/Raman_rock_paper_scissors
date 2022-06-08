# the game is played as follows:
# on running the programme, your camera will prompt for an input
# provide the required input after pressing the 's' key. 
# Once the 's' key would be pressed, the frame name will change to input for 10 seconds
# Show your move inside the blue bounding box as any one of the possible inpts namely- rock, paper and scissors
# After the 10 seconds are over, another window would pop out showing the outcome of the game(whether you win or lose)
# To play another game press 's' again.
# You can play 3 games in one execution of the programme.

#importing necessary libraries
import time
from serial import Serial
import tensorflow as tf
import numpy as np
import platform
import datetime
import os
import math
import random
import cv2 as cv
import numpy as np
from tensorflow import keras

#importing the pre trained model for rock(0), paper(1), scissors(2) detection
model = keras.models.load_model('./model.h5')

#.......................................................
#arduino connection

# model.summary()

# arduino = Serial(port='COM5', baudrate=115200, timeout=.1)
# def write_read(x):
#     arduino.write(bytes(x, 'utf-8'))
#     time.sleep(0.05)
#     data = arduino.readline()
#     return data

#........................................................

#function to print the outcome of the game
def outcome(computer, me):
      
      if(computer==0 and me==1):
            return 1
            
      if(computer==me):
            print("Draw")
            return 0
        
      if(computer==1 and me==2):
            return 1
            
      if(computer==2 and me==0):
            return 1
      else:
            return 2


#functio to print the preditoion of the interpretation that the model makes
def print_pred(pred):
  if(pred==0):
    print("rock")
  if(pred==1):
    print("paper")
  if(pred==2):
    print("scissor")

#function to find the overall outcome of the the game in which we are taking multipke inputs( predictions made by the model in a given time duration and then predicting the max out of all the predictions to get the most accurate one) as snapshots.
def find_max(result):
      rock=0
      paper=0
      scissor=0
      for i in range(len(result)):
            if(result[i]==0):
                  rock=rock+1
            if(result[i]==2):
                  scissor=scissor+1
            if(result[i]==1):
                  paper=paper+1
      if(rock>=scissor and rock>=paper):
            return 'rock'
      if(scissor>=rock and scissor>=paper):
            return 'scissor'
      if(paper>=scissor and paper>=rock):
            return 'paper'



#using open cv to do real time detection by taking continious snapshots
cap=cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)       
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

def game():
  comp=np.random.randint(2)

  while True:
    _, img0 = cap.read()
    img0=cv.rectangle(img0, (280,0), (1000,720), (255,0,0), 10)
    win = cv.imshow("input", img0)
    
    if cv.waitKey(200) & 0xFF==ord('s'):
      print('game started')
      cv.destroyAllWindows()
      break
  
  result=[]
  start_tm = time.time()
  
  while True:

      _,img0=cap.read()
      #converting the img and resizing it
      img = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
      img=np.asarray(img).astype(np.float32)
      #drawing the bounding box
      img0=cv.rectangle(img0, (280,0), (1000,720), (255,0,0), 10)
      img= tf.image.central_crop(img, 9/16)
      img=tf.image.resize(img,(150,150))
      img = tf.expand_dims(img, axis=0) /255
      
      #predicting the hand gesture(input) using the model
      pred = model.predict(img) 
      pred = np.argmax(pred)
      result.append(pred)
      
      #showing the input frame
      cv.imshow("playing",img0)
      
      #creating the window for showing the game outcome
      my_img_1 = np.zeros((512, 512, 1), dtype = "uint8")
      
      comp1='rock'
      if(comp==0):
            comp1='rock'
      if(comp==1):
            comp1='paper'
      if(comp==2):
            comp1='scissor'
      cv.putText(my_img_1, comp1, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
      
      if cv.waitKey(300) & ((time.time() - start_tm) >= 10.0):
          cv.destroyAllWindows()
          
          ans=find_max(result)
      
          ans1=0
          if(ans=='rock'):
                ans1=0
          if(ans=='paper'):
                ans1=1
          if(ans=='scissor'):
                ans1=2
          print("computer is playing: ",comp)
          print("MY input: ",ans1)
          final=outcome(comp,ans1)
          
          #displaying the final output based onf the outcome
          if(final==0):
                cv.imshow('result',cv.imread('/Users/mansi/Downloads/rock_paper_scissor/draw1.jpeg'))
          if(final==1):
                cv.imshow('result',cv.imread('/Users/mansi/Downloads/rock_paper_scissor/win.jpeg'))
          if(final==0):
                cv.imshow('result',cv.imread('/Users/mansi/Downloads/rock_paper_scissor/lose1.jpeg'))
          while True:
            
            if cv.waitKey(200) & 0xFF==ord('s'):
                  print('game ended')
                  cv.destroyAllWindows()
                  break
          break

#playing the game 3 times
game()
game()
game()
cap.release()
