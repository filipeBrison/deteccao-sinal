import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

PontaIndicador = 8
PontaMedio = 12
PontaPolegar = 4
PontaAnelar = 16
PontaMinimo = 20
Palma = 0

A = 0
L = 0
ligar1 = 1
ligar2 = 1
ligar3 = 1
ligarM = 1
ligarV = 1
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
                lm_list[PontaPolegar-2].y > lm_list[PontaMedio-2].y) and A == 0:
                    
                print("Alexa")
                st = 'Alexa!'
                A = 1
                L = 0

            if (lm_list[PontaIndicador].y < lm_list[PontaIndicador-1].y < lm_list[PontaIndicador-2].y < 
                lm_list[PontaMedio].y ) and (lm_list[PontaPolegar].x < lm_list[PontaPolegar-1].x < 
                lm_list[PontaPolegar-2].x) and (lm_list[PontaMedio].y > lm_list[PontaMedio-1].y > 
                lm_list[PontaMedio-2].y) and (lm_list[PontaAnelar].y > lm_list[PontaAnelar-1].y > 
                lm_list[PontaAnelar-2].y) and (lm_list[PontaMinimo].y > lm_list[PontaMinimo-1].y > 
                lm_list[PontaMinimo-2].y) and (lm_list[PontaMedio].x < lm_list[PontaAnelar].x < 
                lm_list[PontaMinimo].x) and (lm_list[PontaMedio-1].x < lm_list[PontaAnelar-1].x < 
                lm_list[PontaMinimo-1].x) and (lm_list[PontaMedio-2].x < lm_list[PontaAnelar-2].x < 
                lm_list[PontaMinimo-2].x) and A == 1 and L == 0:

                print("L")  
                st = 'Qual uma das lampadas (1, 2 ou 3)?'
                L = 1
                A = 0
                
            if (lm_list[PontaIndicador].y > lm_list[PontaIndicador-1].y > lm_list[PontaIndicador-2].y
                ) and (lm_list[PontaMedio].y > lm_list[PontaMedio-1].y > lm_list[PontaMedio-2].y
                ) and (lm_list[PontaAnelar].y > lm_list[PontaAnelar-1].y > lm_list[PontaAnelar-2].y
                ) and (lm_list[Palma].y < lm_list[PontaMedio].y) and A == 1:

                print("M")  
                if ligarM == 1:
                    st = 'Voce liga motor'
                    ligarM = 0
                elif ligarM == 0:
                    st = 'Voce desliga motor'
                    ligarM = 1
                A = 0

            if (lm_list[PontaPolegar].y < lm_list[PontaPolegar-1].y < lm_list[PontaPolegar-2].y < 
                lm_list[PontaPolegar-3].y < lm_list[Palma].y) and (lm_list[PontaPolegar-2].y < 
                lm_list[PontaMedio-2].y) and (lm_list[PontaIndicador].x > 
                lm_list[PontaIndicador-1].x) and L == 1:
  
                print("L1")
                if ligar1 == 1:
                    st = 'Voce liga lampada L1'
                    ligar1 = 0
                elif ligar1 == 0:
                    st = 'Voce desliga lampada L1'
                    ligar1 = 1
                L = 0
                A = 0

            if (lm_list[PontaPolegar].y < lm_list[PontaPolegar-1].y < lm_list[PontaPolegar-2].y) and (
                lm_list[PontaIndicador].x < lm_list[PontaIndicador-1].x < lm_list[PontaIndicador-2].x < 
                lm_list[PontaIndicador-3].x) and (lm_list[PontaMedio].x > lm_list[PontaMedio-1].x > 
                lm_list[PontaMedio-2].x) and (lm_list[PontaAnelar].x > lm_list[PontaAnelar-1].x > 
                lm_list[PontaAnelar-2].x) and ( lm_list[PontaMinimo].x > lm_list[PontaMinimo-1].x > 
                lm_list[PontaMinimo-2].x) and (lm_list[Palma].x > lm_list[PontaIndicador].x) and (
                lm_list[PontaMedio].y < lm_list[PontaAnelar].y < lm_list[PontaMinimo].y) and (
                lm_list[PontaMedio-1].y < lm_list[PontaAnelar-1].y < lm_list[PontaMinimo-1].y) and (
                lm_list[PontaMedio-2].y < lm_list[PontaAnelar-2].y < lm_list[PontaMinimo-2].y) and L == 1:

                print("L2")   
                if ligar2 == 1:
                    st = 'Voce liga lampada L2'
                    ligar2 = 0
                elif ligar2 == 0:
                    st = 'Voce desliga lampada L2'
                    ligar2 = 1
                L = 0
                A = 0

            if (lm_list[PontaIndicador].x < lm_list[PontaIndicador-1].x < lm_list[PontaIndicador-2].x < lm_list[PontaIndicador-3].x) and (lm_list[PontaMedio].x < 
                lm_list[PontaMedio-1].x < lm_list[PontaMedio-2].x < lm_list[PontaMedio-3].x) and (lm_list[PontaAnelar].x <
                lm_list[PontaAnelar-1].x < lm_list[PontaAnelar-2].x < lm_list[PontaAnelar-3].x) and (lm_list[PontaAnelar-2].x < lm_list[PontaMinimo].x) and (lm_list[Palma].y >
                lm_list[PontaPolegar-3].y) and (lm_list[PontaIndicador-3].y < lm_list[PontaMedio-3].y < 
                lm_list[PontaAnelar-3].y < lm_list[PontaMinimo-3].y) and L == 1:

                print("L3")  
                if ligar3 == 1:
                    st = 'Voce liga lampada L3'
                    ligar3 = 0
                elif ligar3 == 0:
                    st = 'Voce desliga lampada L3'
                    ligar3 = 1
                L = 0
                A = 0

            if (lm_list[PontaIndicador].y < lm_list[PontaIndicador-1].y < lm_list[PontaIndicador-2].y < 
                lm_list[PontaIndicador-3].y) and (lm_list[PontaMedio].y < lm_list[PontaMedio-1].y < 
                lm_list[PontaMedio-2].y < lm_list[PontaMedio-3].y) and (lm_list[PontaAnelar].y > 
                lm_list[PontaAnelar-1].y > lm_list[PontaAnelar-2].y) and (lm_list[PontaMinimo].y > 
                lm_list[PontaMinimo-1].y > lm_list[PontaMinimo-2].y) and A == 1:

                print("V")  
                if ligarV == 1:
                    st = 'Voce liga ventilador'
                    ligarV = 0
                elif ligarV == 0:
                    st = 'Voce desliga ventilador'
                    ligarV = 1
                A = 0

            cv2.putText(image, st, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    cv2.imshow('Camera', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()