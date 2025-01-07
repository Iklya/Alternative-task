# Alternative-task

# Ссылка на Google Colab
  Перейдя по ссылке, можно проверить работоспособность используемых метрик:
https://colab.research.google.com/drive/174bxpcxs4Vi2ndHpPRrp1LDYbSz6btx3?hl=ru#scrollTo=cLmtwDOQSFNY
(единственный минус - загрузка изображений возможна исключительно вручную с использованием возмжностей Google Colab)
  В процессе работы с изображениями в Google Colab, который не поддерживает графический интерфейс, было решено использовать библиотеку Matplotlib для визуализации. При показе изображений с помощью plt.imshow(image) цвета отображались некорректно. Это связано с тем, что OpenCV по умолчанию использует формат BGR, тогда как Matplotlib ожидает формат RGB. В связи с этим в код программы были добавлены следующие две строки с использованием функции cv2.cvtColor, благодаря которой стало возможно преобразовать цветовое пространство из BGR в RGB:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Метрики
  - Для вычисления значений метрик: MSE, SSIM, MS-SSIM, RMSE, PSNR, PSNR-B, UQI, SCC, RASE, SAM, VIFP, ERGAS, D_lambda был использован модуль sewar.
  - Для вычисления значений метрик: IW-SSIM, DSS, HaarPSI, MDSI, SR-SIM, FSIM, VSI, GMSD, MS-GMSD был использован модуль piq.
  - Значения метрик NPCR, UACI были вычислены при помощи написанных вручную функций calculate_npcr() и calculate_uaci(). Формулы для вычисления данных метрик были взяты из прикреплённого источника c именем файла "book.pdf".
