# Alternative-task

# Ссылка на Google Colab
  Перейдя по ссылке, можно проверить работоспособность используемых метрик:
https://colab.research.google.com/drive/174bxpcxs4Vi2ndHpPRrp1LDYbSz6btx3?hl=ru#scrollTo=cLmtwDOQSFNY
(единственный минус - загрузка изображений возможна исключительно вручную с использованием возмжностей Google Colab)
  В процессе работы с изображениями в Google Colab, который не поддерживает графический интерфейс, было решено использовать библиотеку Matplotlib для визуализации. При показе изображений с помощью plt.imshow(image) цвета отображались некорректно. Это связано с тем, что OpenCV по умолчанию использует формат BGR, тогда как Matplotlib ожидает формат RGB. В связи с этим в код программы были добавлены следующие две строки с использованием функции cv2.cvtColor, благодаря которой стало возможно преобразовать цветовое пространство из BGR в RGB:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Метрики
  - Для вычисления значений метрик: MSE, SSIM, MS-SSIM, RMSE, PSNR, PSNR-B, UQI, SCC, RASE, SAM, VIFP, ERGAS, D_lambda был использован модуль sewar;
  - Для вычисления значений метрик: IW-SSIM, DSS, HaarPSI, MDSI, SR-SIM, FSIM, VSI, GMSD, MS-GMSD был использован модуль piq;
  - Значения метрик NPCR, UACI были вычислены при помощи написанных вручную функций calculate_npcr() и calculate_uaci(). Формулы для вычисления данных метрик были взяты из прикреплённого источника c именем файла "book.pdf".

# Зависимости в коде написанного приложения
  - PyQt5.QtWidgets, PyQt5.QtGui и PyQt5.QtCore: модули для создания пользовательского интерфейса и для работы с изображениями, графическими элементами и основными функциями приложения;
  - numpy: для работы с многомерными массивами и для вычислений;
  - piq: используется для вычисления метрик;
  - sewar: используется для вычисления метрик;
  - torch;
  - cv2;
  - matplotlib.pyplot;
  - PIL;
  - sys.

# Основная логика программы
  Программа позволяет пользователю загрузить два изображения и вычислить набор метрик для их сравнения.

# Дополнительно
  https://drive.google.com/drive/folders/12BMGVvdR5H6pA3JD2JFENF8LS7YCqY1T - полезная ссылка на найденный датасет в Google Drive.
