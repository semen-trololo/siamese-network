# Пример кода для веб-камеры
import cv2
import random
import os

# Прописываем классификатор для лица
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Прописываем классификатор для глаз
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Путь куда сохраняем картинки
PATCH_SAVE = 's0/'
# Метка
METKA = 'I_'
PATCH = '480.mp4'
cap = cv2.VideoCapture(PATCH)
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Ищем лица
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(100, 100)
    )
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Вырезаем область с лицами
        # Ищем глаза в области с лицом для увеличения точности обноружения.
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(10, 10),
        )
        if len(eyes) > 0:
            # Генерируем имя картинки
            name = METKA + str(random.randint(1, 1000000)) + '.jpg'
            size = (100, 100)
        # Изменяем размер картинки на 100х100. Нам не нужно заботиться о пропорциях,

            output = cv2.resize(roi_gray, size, interpolation=cv2.INTER_AREA)
# Сохраняем лицо
            cv2.imwrite(os.path.join(PATCH_SAVE, name), output)
# Рисуем контур лица на видео для удобства.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 27:  # Клавиша Esc
        break
cap.release()
cv2.destroyAllWindows()
