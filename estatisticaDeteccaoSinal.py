import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Pontas de cada dedo
PontaIndicador = 8
PontaMedio = 12
PontaPolegar = 4
PontaAnelar = 16
PontaMinimo = 20
Palma = 0

cont = 1
simbolo = 1
fimTeste = 0
t = 0
deteccao = 0
acertoA = 0
acertoL = 0
acertoM = 0
acertoV = 0
acerto1 = 0
acerto2 = 0
acerto3 = 0
erro = 0
tempoA = []
tempoL = []
tempoM = []
tempoV = []
tempo1 = []
tempo2 = []
tempo3 = []
seqA = []
seqL = []
seqM = []
seqV = []
seq1 = []
seq2 = []
seq3 = []
st = 'Bem-vindo'
st1 = ['A','L','M','1','2','3','V','Fim']
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
        if t == 0:
            inicio = time.time()
            t = 1
        for hand_landmarks in results.multi_hand_landmarks:
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append(lm)
               # print(id,":", int(lm.x*w), int(lm.y*h))

            # Simbolo A detectado
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
                lm_list[PontaPolegar-2].y > lm_list[PontaMedio-2].y) and deteccao == 0:

                fim = time.time() 
                tempo = fim - inicio
                if simbolo == 1:
                    acertoA = acertoA + 1
                    print(st1[simbolo-1],"= A em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= A em", cont,":","{0:.3f}".format(tempo),"Erro")
                time.sleep(3)
                st = 'A'
                t = 0
                deteccao = 1

            # Simbolo L detectado
            if (lm_list[PontaIndicador].y < lm_list[PontaIndicador-1].y < lm_list[PontaIndicador-2].y < 
                lm_list[PontaMedio].y ) and (lm_list[PontaPolegar].x < lm_list[PontaPolegar-1].x < 
                lm_list[PontaPolegar-2].x) and (lm_list[PontaMedio].y > lm_list[PontaMedio-1].y > 
                lm_list[PontaMedio-2].y) and (lm_list[PontaAnelar].y > lm_list[PontaAnelar-1].y > 
                lm_list[PontaAnelar-2].y) and (lm_list[PontaMinimo].y > lm_list[PontaMinimo-1].y > 
                lm_list[PontaMinimo-2].y) and (lm_list[PontaMedio].x < lm_list[PontaAnelar].x < 
                lm_list[PontaMinimo].x) and (lm_list[PontaMedio-1].x < lm_list[PontaAnelar-1].x < 
                lm_list[PontaMinimo-1].x) and (lm_list[PontaMedio-2].x < lm_list[PontaAnelar-2].x < 
                lm_list[PontaMinimo-2].x) and deteccao == 0:

                fim = time.time()
                tempo = fim - inicio
                if simbolo == 2:
                    acertoL = acertoL + 1
                    print(st1[simbolo-1],"= L em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= L em", cont,":","{0:.3f}".format(tempo),"Erro") 
                time.sleep(3)
                st = 'L'
                t = 0
                deteccao = 1

            # Simbolo M detectado
            if (lm_list[PontaIndicador].y > lm_list[PontaIndicador-1].y > lm_list[PontaIndicador-2].y
                ) and (lm_list[PontaMedio].y > lm_list[PontaMedio-1].y > lm_list[PontaMedio-2].y
                ) and (lm_list[PontaAnelar].y > lm_list[PontaAnelar-1].y > lm_list[PontaAnelar-2].y
                ) and (lm_list[Palma].y < lm_list[PontaMedio].y) and deteccao == 0:

                fim = time.time()
                tempo = fim - inicio
                if simbolo == 3:
                    acertoM = acertoM + 1
                    print(st1[simbolo-1],"= M em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= M em", cont,":","{0:.3f}".format(tempo),"Erro")
                time.sleep(3)
                st = 'M'
                t = 0
                deteccao = 1

            # Simbolo 1 detectado
            if (lm_list[PontaPolegar].y < lm_list[PontaPolegar-1].y < lm_list[PontaPolegar-2].y < 
                lm_list[PontaPolegar-3].y < lm_list[Palma].y) and (lm_list[PontaPolegar-2].y < 
                lm_list[PontaMedio-2].y) and (lm_list[PontaIndicador].x > lm_list[PontaIndicador-1].x) and deteccao == 0:
                
                fim = time.time()
                tempo = fim - inicio
                if simbolo == 4:
                    acerto1 = acerto1 + 1
                    print(st1[simbolo-1],"= 1 em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= 1 em", cont,":","{0:.3f}".format(tempo),"Erro")
                time.sleep(3)
                st = '1'
                t = 0
                deteccao = 1

            # Simbolo 2 detectado
            if (lm_list[PontaPolegar].y < lm_list[PontaPolegar-1].y < lm_list[PontaPolegar-2].y) and (
                lm_list[PontaIndicador].x < lm_list[PontaIndicador-1].x < lm_list[PontaIndicador-2].x < 
                lm_list[PontaIndicador-3].x) and (lm_list[PontaMedio].x > lm_list[PontaMedio-1].x > 
                lm_list[PontaMedio-2].x) and (lm_list[PontaAnelar].x > lm_list[PontaAnelar-1].x > 
                lm_list[PontaAnelar-2].x) and ( lm_list[PontaMinimo].x > lm_list[PontaMinimo-1].x > 
                lm_list[PontaMinimo-2].x) and (lm_list[Palma].x > lm_list[PontaIndicador].x) and (
                lm_list[PontaMedio].y < lm_list[PontaAnelar].y < lm_list[PontaMinimo].y) and (
                lm_list[PontaMedio-1].y < lm_list[PontaAnelar-1].y < lm_list[PontaMinimo-1].y) and (
                lm_list[PontaMedio-2].y < lm_list[PontaAnelar-2].y < lm_list[PontaMinimo-2].y) and deteccao == 0:
                
                fim = time.time()
                tempo = fim - inicio
                if simbolo == 5:
                    acerto2 = acerto2 + 1
                    print(st1[simbolo-1],"= 2 em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= 2 em", cont,":","{0:.3f}".format(tempo),"Erro") 
                time.sleep(3)
                st = '2'
                t = 0
                deteccao = 1

            # Simbolo 3 detectado
            if (lm_list[PontaIndicador].x < lm_list[PontaIndicador-1].x < lm_list[PontaIndicador-2].x < lm_list[PontaIndicador-3].x) and (lm_list[PontaMedio].x < 
                lm_list[PontaMedio-1].x < lm_list[PontaMedio-2].x < lm_list[PontaMedio-3].x) and (lm_list[PontaAnelar].x <
                lm_list[PontaAnelar-1].x < lm_list[PontaAnelar-2].x < lm_list[PontaAnelar-3].x) and (lm_list[PontaAnelar-2].x < lm_list[PontaMinimo].x) and (lm_list[Palma].y >
                lm_list[PontaPolegar-3].y) and (lm_list[PontaIndicador-3].y < lm_list[PontaMedio-3].y < 
                lm_list[PontaAnelar-3].y < lm_list[PontaMinimo-3].y) and deteccao == 0:

                fim = time.time()
                tempo = fim - inicio
                if simbolo == 6:
                    acerto3 = acerto3 + 1
                    print(st1[simbolo-1],"= 3 em", cont,":","{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= 3 em", cont,":","{0:.3f}".format(tempo),"Erro")
                time.sleep(3)
                st = '3'
                t = 0
                deteccao = 1

            # Simbolo V detectado
            if (lm_list[PontaIndicador].y < lm_list[PontaIndicador-1].y < lm_list[PontaIndicador-2].y < 
                lm_list[PontaIndicador-3].y) and (lm_list[PontaMedio].y < lm_list[PontaMedio-1].y < 
                lm_list[PontaMedio-2].y < lm_list[PontaMedio-3].y) and (lm_list[PontaAnelar].y > 
                lm_list[PontaAnelar-1].y > lm_list[PontaAnelar-2].y) and (lm_list[PontaMinimo].y > 
                lm_list[PontaMinimo-1].y > lm_list[PontaMinimo-2].y) and deteccao == 0:

                fim = time.time()
                tempo = fim - inicio 
                if simbolo == 7:
                    acertoV = acertoV + 1
                    print(st1[simbolo-1],"= V em", cont,":", "{0:.3f}".format(tempo),"Acerto")
                else:
                    print(st1[simbolo-1],"= V em", cont,":", "{0:.3f}".format(tempo),"Erro")
                time.sleep(3)
                st = 'V'
                t = 0
                deteccao = 1

            # Durante detecção, o tempo é colocado em tempo de cada sinal e letra colocada na sequencia
            if deteccao == 1:
                if simbolo == 1:
                    tempoA.append(tempo)
                    seqA.append(st)
                elif simbolo == 2:
                    tempoL.append(tempo)
                    seqL.append(st)
                elif simbolo == 3:
                    tempoM.append(tempo)
                    seqM.append(st)
                elif simbolo == 4:
                    tempo1.append(tempo)
                    seq1.append(st)
                elif simbolo == 5:
                    tempo2.append(tempo)
                    seq2.append(st)
                elif simbolo == 6:
                    tempo3.append(tempo)
                    seq3.append(st)
                elif simbolo == 7:
                    tempoV.append(tempo)
                    seqV.append(st)
                # Mudança de simbolo ou não
                if cont == 10:
                    simbolo = simbolo + 1
                    cont = 0
                cont = cont + 1

                deteccao = 0
             
            cv2.putText(image, st, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    # Texto aparece na camera
    cv2.putText(image, 'Diga ', (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 252), 3)
    cv2.putText(image, st1[simbolo-1], (560, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 252), 3)
    cv2.imshow('Camera', image)
    if cv2.waitKey(5) & 0xFF == 27:
    #if fimTeste == 1:
      break
cap.release()

# Resultado estatistico
print("Resultados de cada simbolo:")
print("Simbolo | Tempo médio de reconhecimento | Desvio padrão | Acerto (%) | sequencia")
print("A       | ", "{0:.3f}".format(np.mean(tempoA))," | ","{0:.3f}".format(np.std(tempoA))," | ",((acertoA)/10)," | ",seqA)
print("L       | ", "{0:.3f}".format(np.mean(tempoL))," | ","{0:.3f}".format(np.std(tempoL))," | ",((acertoL)/10)," | ",seqL)
print("M       | ", "{0:.3f}".format(np.mean(tempoM))," | ","{0:.3f}".format(np.std(tempoM))," | ",((acertoM)/10)," | ",seqM)
print("V       | ", "{0:.3f}".format(np.mean(tempoV))," | ","{0:.3f}".format(np.std(tempoV))," | ",((acertoV)/10)," | ",seqV)
print("1       | ", "{0:.3f}".format(np.mean(tempo1))," | ","{0:.3f}".format(np.std(tempo1))," | ",((acerto1)/10)," | ",seq1)
print("2       | ", "{0:.3f}".format(np.mean(tempo2))," | ","{0:.3f}".format(np.std(tempo2))," | ",((acerto2)/10)," | ",seq2)
print("3       | ", "{0:.3f}".format(np.mean(tempo3))," | ","{0:.3f}".format(np.std(tempo3))," | ",((acerto3)/10)," | ",seq3)

dado = [tempoA, tempoL, tempoM, tempoV, tempo1, tempo2, tempo3]
fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(dado)
#ax.set_xticklabels(['A','L','M','V','1','2','3'])
plt.show()