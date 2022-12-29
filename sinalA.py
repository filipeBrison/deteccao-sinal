import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

PontaIndicador = 8
PontaMedio = 12
PontaPolegar = 4
PontaAnelar = 16
PontaMinimo = 20
Palma = 0

cont = 0
simbolo = 1
fimTeste = 0
tempoA = []
tempoL = []
tempoM = []
tempoV = []
tempo1 = []
tempo2 = []
tempo3 = []

st = 'Bem-vindo'

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image.flags.writeable = True
    h, w, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        #time.sleep(0.1)
    
        for hand_landmarks in results.multi_hand_landmarks:
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append(lm)
               # print(id,":", int(lm.x*w), int(lm.y*h))

            inicio = time.time()
            if (lm_list[PontaIndicador-2].y < lm_list[PontaIndicador-1].y < lm_list[PontaIndicador].y < 
                lm_list[Palma].y) and (lm_list[PontaMedio-2].y < lm_list[PontaMedio-1].y < 
                lm_list[PontaMedio].y < lm_list[Palma].y) and (lm_list[PontaAnelar-2].y < 
                lm_list[PontaAnelar-1].y < lm_list[PontaAnelar].y < lm_list[Palma].y) and (
                lm_list[PontaMinimo-2].y < lm_list[PontaMinimo-1].y < lm_list[PontaMinimo].y < 
                lm_list[Palma].y) and (lm_list[PontaPolegar].y < lm_list[PontaPolegar-1].y < 
                lm_list[PontaPolegar-2].y < lm_list[PontaPolegar-3].y < lm_list[Palma].y) and (
                lm_list[PontaMinimo-2].x > lm_list[PontaAnelar-2].x > lm_list[PontaMedio-2].x > 
                lm_list[PontaIndicador-2].x > lm_list[PontaPolegar].x) and (lm_list[PontaPolegar].y < 
                lm_list[PontaIndicador-2].y) and (lm_list[Palma].x > lm_list[PontaIndicador].x) and (
                lm_list[PontaPolegar-2].y > lm_list[PontaMedio-2].y) and simbolo == 1:

                fim = time.time() 
                print(inicio)
                print(fim)
                tempo = ((fim - inicio)* 10**6)
                tempoA.append(tempo)
                cont = cont + 1
                print("A ", cont)
                if cont == 10:
                    simbolo = 2
                    cont = 0
                time.sleep(2)
                st = 'A'

                         
            cv2.putText(image, st, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    cv2.imshow('Camera', image)
    if cv2.waitKey(5) & 0xFF == 27:
    #if cv2.waitKey(5) & fimTeste == 1:
      break
cap.release()
print(tempoA)
print("Resultado do simbolo A:")
print("Simbolo | Tempo médio de reconhecimento | Desvio padrão")
print("A : ", np.mean(tempoA)," | ",np.std(tempoA))

fig = plt.figure(figsize =(10, 7))
plt.boxplot(tempoA)
plt.show()