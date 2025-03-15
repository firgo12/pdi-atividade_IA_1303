import cv2
import numpy as np

# Carregar o vídeo
cap = cv2.VideoCapture("./q1/q1B.mp4")

if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo.")
    exit()

# Variável para verificar se a colisão já ocorreu
colisao_ocorreu = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Vídeo finalizado ou erro na leitura do frame.")
        break

    # Redimensionar para performance
    altura, largura = frame.shape[:2]
    nova_largura = 640
    nova_altura = int(altura * nova_largura / largura)
    frame_resized = cv2.resize(frame, (nova_largura, nova_altura))

    # Converter para HSV
    img_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # Faixas de cores ajustadas
    lower_blue = np.array([90, 50, 50])  # Azul pastel
    upper_blue = np.array([130, 255, 255])
    lower_orange = np.array([10, 100, 100])  # Laranja pastel
    upper_orange = np.array([25, 255, 255])

    # Criar máscaras
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)

    # Encontrar contornos
    contornos_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variável para verificar colisão
    passou_barreira = False

    # Identificar os bounding boxes
    if contornos_blue:
        maior_contorno_azul = max(contornos_blue, key=cv2.contourArea)
        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(maior_contorno_azul)
        
        # Desenhar o retângulo verde no bounding box do azul
        cv2.rectangle(frame_resized, (x_blue, y_blue), (x_blue + w_blue, y_blue + h_blue), (0, 255, 0), 2)  # Retângulo verde

        if contornos_orange:
            for contorno_orange in contornos_orange:
                x_orange, y_orange, w_orange, h_orange = cv2.boundingRect(contorno_orange)
                
                # Desenhar o retângulo laranja
                cv2.rectangle(frame_resized, (x_orange, y_orange), (x_orange + w_orange, y_orange + h_orange), (0, 165, 255), 2)  # Retângulo laranja

                # Verificar colisão entre os bounding boxes
                if (x_blue < x_orange + w_orange and x_blue + w_blue > x_orange and
                    y_blue < y_orange + h_orange and y_blue + h_blue > y_orange):
                    colisao_ocorreu = True  # Colisão detectada

                # Verificar se o laranja passou a barreira após a colisão (vertical ou lateral)
                if colisao_ocorreu and (
                    y_orange + h_orange < y_blue or y_orange > y_blue + h_blue or  # Ultrapassagem vertical
                    x_orange + w_orange < x_blue or x_orange > x_blue + w_blue  # Ultrapassagem lateral
                ):
                    passou_barreira = True
                    break  

    # Exibir mensagem de colisão se ocorreu
    if colisao_ocorreu and not passou_barreira:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_resized, "COLISAO DETECTADA", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir mensagem "PASSOU BARREIRA" se ultrapassagem ocorreu
    if passou_barreira:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_resized, "PASSOU BARREIRA", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar vídeo processado
    cv2.imshow("Video Processado", frame_resized)

    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
