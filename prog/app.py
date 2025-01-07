import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from colab import sewar_funcs, piq_funcs, calculate_npcr, calculate_uaci

class ImageLoader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Основное окно
        self.setWindowTitle("Загрузка изображений и вычисление метрик")
        self.setGeometry(100, 100, 800, 800)

        # Центральный виджет
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: rgb(255, 212, 115);")
        self.setCentralWidget(central_widget)

        # Главный макет программы
        main_layout = QVBoxLayout()
        # Горизонтальный макет (нужен для изображений и кнопок)
        image_layout = QHBoxLayout()

        # Левая часть
        left_layout = QVBoxLayout()
        self.label1 = QLabel("Изображение №1", self)
        self.label1.setStyleSheet("border: 4px solid black; font-size: 14px; background-color: rgb(255, 240, 255);")
        self.label1.setAlignment(Qt.AlignCenter)

        self.button1 = QPushButton("Загрузить первое изображение", self)
        self.button1.setStyleSheet("font-weight: bold; font-size: 12px; background-color: rgb(255, 212, 255);")
        self.button1.clicked.connect(lambda: self.load_image(self.label1, "img1"))

        left_layout.addWidget(self.label1)
        left_layout.addWidget(self.button1)

        # Правая часть
        right_layout = QVBoxLayout()
        self.label2 = QLabel("Изображение №2", self)
        self.label2.setStyleSheet("border: 4px solid black; font-size: 14px; background-color: rgb(255, 240, 255);")
        self.label2.setAlignment(Qt.AlignCenter)

        self.button2 = QPushButton("Загрузить второе изображение", self)
        self.button2.setStyleSheet("font-weight: bold; font-size: 12px; background-color: rgb(255, 212, 255);")
        self.button2.clicked.connect(lambda: self.load_image(self.label2, "img2"))

        right_layout.addWidget(self.label2)
        right_layout.addWidget(self.button2)

        # Добавление левой и правой части в макет для изображений
        image_layout.addLayout(left_layout)
        image_layout.addLayout(right_layout)

        # Кнопка для вычисления метрик
        self.metrics_button = QPushButton("Вычислить метрики", self)
        self.metrics_button.setStyleSheet("font-weight: bold; font-size: 12px; background-color: rgb(255, 212, 255);")
        self.metrics_button.clicked.connect(self.calculate_metrics)

        # Макет для отображения метрик
        self.metrics_layout = QVBoxLayout()
        self.metrics_header = QLabel("Вычисление метрик:")
        self.metrics_header.setAlignment(Qt.AlignCenter)
        self.metrics_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.metrics_layout.addWidget(self.metrics_header)

        self.metrics_grid = QGridLayout()
        self.metrics_layout.addLayout(self.metrics_grid)

        # Список всех метрик и их значений
        self.MSE_label = QLabel("MSE:")
        self.MSE_label.setAlignment(Qt.AlignRight)
        self.MSE_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.MSE_value = QLabel("0.0")
        self.MSE_value.setAlignment(Qt.AlignLeft)
        self.MSE_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.MSE_label, 0, 0)
        self.metrics_grid.addWidget(self.MSE_value, 0, 1)

        self.SSIM_label = QLabel("SSIM:")
        self.SSIM_label.setAlignment(Qt.AlignRight)
        self.SSIM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.SSIM_value = QLabel("0.0")
        self.SSIM_value.setAlignment(Qt.AlignLeft)
        self.SSIM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.SSIM_label, 1, 0)
        self.metrics_grid.addWidget(self.SSIM_value, 1, 1)       

        self.MS_SSIM_label = QLabel("MS-SSIM:")
        self.MS_SSIM_label.setAlignment(Qt.AlignRight)
        self.MS_SSIM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.MS_SSIM_value = QLabel("0.0")
        self.MS_SSIM_value.setAlignment(Qt.AlignLeft)
        self.MS_SSIM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.MS_SSIM_label, 2, 0)
        self.metrics_grid.addWidget(self.MS_SSIM_value, 2, 1)

        self.RMSE_label = QLabel("RMSE:")
        self.RMSE_label.setAlignment(Qt.AlignRight)
        self.RMSE_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.RMSE_value = QLabel("0.0")
        self.RMSE_value.setAlignment(Qt.AlignLeft)
        self.RMSE_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.RMSE_label, 3, 0)
        self.metrics_grid.addWidget(self.RMSE_value, 3, 1)

        self.PSNR_label = QLabel("PSNR:")
        self.PSNR_label.setAlignment(Qt.AlignRight)
        self.PSNR_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.PSNR_value = QLabel("0.0")
        self.PSNR_value.setAlignment(Qt.AlignLeft)
        self.PSNR_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.PSNR_label, 4, 0)
        self.metrics_grid.addWidget(self.PSNR_value, 4, 1)

        self.PSNR_B_label = QLabel("PSNR-B:")
        self.PSNR_B_label.setAlignment(Qt.AlignRight)
        self.PSNR_B_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.PSNR_B_value = QLabel("0.0")
        self.PSNR_B_value.setAlignment(Qt.AlignLeft)
        self.PSNR_B_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.PSNR_B_label, 5, 0)
        self.metrics_grid.addWidget(self.PSNR_B_value, 5, 1)

        self.UQI_label = QLabel("UQI:")
        self.UQI_label.setAlignment(Qt.AlignRight)
        self.UQI_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.UQI_value = QLabel("0.0")
        self.UQI_value.setAlignment(Qt.AlignLeft)
        self.UQI_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.UQI_label, 6, 0)
        self.metrics_grid.addWidget(self.UQI_value, 6, 1)

        self.SCC_label = QLabel("SCC:")
        self.SCC_label.setAlignment(Qt.AlignRight)
        self.SCC_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.SCC_value = QLabel("0.0")
        self.SCC_value.setAlignment(Qt.AlignLeft)
        self.SCC_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.SCC_label, 7, 0)
        self.metrics_grid.addWidget(self.SCC_value, 7, 1)

        self.RASE_label = QLabel("RASE:")
        self.RASE_label.setAlignment(Qt.AlignRight)
        self.RASE_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.RASE_value = QLabel("0.0")
        self.RASE_value.setAlignment(Qt.AlignLeft)
        self.RASE_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.RASE_label, 8, 0)
        self.metrics_grid.addWidget(self.RASE_value, 8, 1)

        self.SAM_label = QLabel("SAM:")
        self.SAM_label.setAlignment(Qt.AlignRight)
        self.SAM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.SAM_value = QLabel("0.0")
        self.SAM_value.setAlignment(Qt.AlignLeft)
        self.SAM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.SAM_label, 9, 0)
        self.metrics_grid.addWidget(self.SAM_value, 9, 1)

        self.VIFP_label = QLabel("VIFP:")
        self.VIFP_label.setAlignment(Qt.AlignRight)
        self.VIFP_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.VIFP_value = QLabel("0.0")
        self.VIFP_value.setAlignment(Qt.AlignLeft)
        self.VIFP_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.VIFP_label, 10, 0)
        self.metrics_grid.addWidget(self.VIFP_value, 10, 1)

        self.ERGAS_label = QLabel("ERGAS:")
        self.ERGAS_label.setAlignment(Qt.AlignRight)
        self.ERGAS_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.ERGAS_value = QLabel("0.0")
        self.ERGAS_value.setAlignment(Qt.AlignLeft)
        self.ERGAS_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.ERGAS_label, 11, 0)
        self.metrics_grid.addWidget(self.ERGAS_value, 11, 1)

        self.D_lambda_label = QLabel("D_lambda:")
        self.D_lambda_label.setAlignment(Qt.AlignRight)
        self.D_lambda_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.D_lambda_value = QLabel("0.0")
        self.D_lambda_value.setAlignment(Qt.AlignLeft)
        self.D_lambda_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.D_lambda_label, 12, 0)
        self.metrics_grid.addWidget(self.D_lambda_value, 12, 1)


        self.IW_SSIM_label = QLabel("IW-SSIM:")
        self.IW_SSIM_label.setAlignment(Qt.AlignRight)
        self.IW_SSIM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.IW_SSIM_value = QLabel("0.0")
        self.IW_SSIM_value.setAlignment(Qt.AlignLeft)
        self.IW_SSIM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.IW_SSIM_label, 0, 2)
        self.metrics_grid.addWidget(self.IW_SSIM_value, 0, 3)

        self.DSS_label = QLabel("DSS:")
        self.DSS_label.setAlignment(Qt.AlignRight)
        self.DSS_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.DSS_value = QLabel("0.0")
        self.DSS_value.setAlignment(Qt.AlignLeft)
        self.DSS_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.DSS_label, 1, 2)
        self.metrics_grid.addWidget(self.DSS_value, 1, 3)

        self.HaarPSI_label = QLabel("HaarPSI:")
        self.HaarPSI_label.setAlignment(Qt.AlignRight)
        self.HaarPSI_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.HaarPSI_value = QLabel("0.0")
        self.HaarPSI_value.setAlignment(Qt.AlignLeft)
        self.HaarPSI_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.HaarPSI_label, 2, 2)
        self.metrics_grid.addWidget(self.HaarPSI_value, 2, 3)

        self.MDSI_label = QLabel("MDSI:")
        self.MDSI_label.setAlignment(Qt.AlignRight)
        self.MDSI_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.MDSI_value = QLabel("0.0")
        self.MDSI_value.setAlignment(Qt.AlignLeft)
        self.MDSI_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.MDSI_label, 3, 2)
        self.metrics_grid.addWidget(self.MDSI_value, 3, 3)

        self.SR_SIM_label = QLabel("SR-SIM:")
        self.SR_SIM_label.setAlignment(Qt.AlignRight)
        self.SR_SIM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.SR_SIM_value = QLabel("0.0")
        self.SR_SIM_value.setAlignment(Qt.AlignLeft)
        self.SR_SIM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.SR_SIM_label, 4, 2)
        self.metrics_grid.addWidget(self.SR_SIM_value, 4, 3)

        self.FSIM_label = QLabel("FSIM:")
        self.FSIM_label.setAlignment(Qt.AlignRight)
        self.FSIM_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.FSIM_value = QLabel("0.0")
        self.FSIM_value.setAlignment(Qt.AlignLeft)
        self.FSIM_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.FSIM_label, 5, 2)
        self.metrics_grid.addWidget(self.FSIM_value, 5, 3)

        self.VSI_label = QLabel("VSI:")
        self.VSI_label.setAlignment(Qt.AlignRight)
        self.VSI_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.VSI_value = QLabel("0.0")
        self.VSI_value.setAlignment(Qt.AlignLeft)
        self.VSI_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.VSI_label, 6, 2)
        self.metrics_grid.addWidget(self.VSI_value, 6, 3)

        self.GMSD_label = QLabel("GMSD:")
        self.GMSD_label.setAlignment(Qt.AlignRight)
        self.GMSD_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.GMSD_value = QLabel("0.0")
        self.GMSD_value.setAlignment(Qt.AlignLeft)
        self.GMSD_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.GMSD_label, 7, 2)
        self.metrics_grid.addWidget(self.GMSD_value, 7, 3)

        self.MS_GMSD_label = QLabel("MS-GMSD:")
        self.MS_GMSD_label.setAlignment(Qt.AlignRight)
        self.MS_GMSD_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.MS_GMSD_value = QLabel("0.0")
        self.MS_GMSD_value.setAlignment(Qt.AlignLeft)
        self.MS_GMSD_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.MS_GMSD_label, 8, 2)
        self.metrics_grid.addWidget(self.MS_GMSD_value, 8, 3)       


        self.NPCR_label = QLabel("NPCR:")
        self.NPCR_label.setAlignment(Qt.AlignRight)
        self.NPCR_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.NPCR_value = QLabel("0.0 %")
        self.NPCR_value.setAlignment(Qt.AlignLeft)
        self.NPCR_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.NPCR_label, 0, 4)
        self.metrics_grid.addWidget(self.NPCR_value, 0, 5)

        self.UACI_label = QLabel("UACI:")
        self.UACI_label.setAlignment(Qt.AlignRight)
        self.UACI_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.UACI_value = QLabel("0.0 %")
        self.UACI_value.setAlignment(Qt.AlignLeft)
        self.UACI_value.setStyleSheet("font-size: 12px;")

        # Добавляем метрику в сетку
        self.metrics_grid.addWidget(self.UACI_label, 1, 4)
        self.metrics_grid.addWidget(self.UACI_value, 1, 5)

        # Добавим все макеты в главный макет
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.metrics_button)
        main_layout.addLayout(self.metrics_layout)

        # Установка главного макета
        central_widget.setLayout(main_layout)

        self.img1 = None
        self.img2 = None

    def load_image(self, label, img_attr):
        # Открыть диалог выбора файла
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите изображение", 
            "", 
            "Изображения (*.png *.jpg *.jpeg *.bmp *.gif)", 
            options=options
        )
        if file_path:
            # Загрузка изображения в метку
            pixmap = QPixmap(file_path)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

            # Загрузка изображения в память
            image = Image.open(file_path).convert("RGB")
            setattr(self, img_attr, np.array(image))

            SV2 = [self.MSE_value, self.SSIM_value, self.MS_SSIM_value, self.RMSE_value, self.PSNR_value, self.PSNR_B_value, self.UQI_value, self.SCC_value, self.RASE_value, 
                self.SAM_value, self.VIFP_value, self.ERGAS_value, self.D_lambda_value]
            for i in range(len(SV2)):
                SV2[i].setText(str(0.0000))

            PV2 = [self.IW_SSIM_value, self.DSS_value, self.HaarPSI_value, self.MDSI_value, self.SR_SIM_value, self.FSIM_value, self.VSI_value, self.GMSD_value, self.MS_GMSD_value]
            for i in range(len(PV2)):
                PV2[i].setText(str(0.0000))
            
            self.NPCR_value.setText(str(0.00)+" %")
            self.UACI_value.setText(str(0.00)+" %")

    def calculate_metrics(self):
        if self.img1 is None and self.img2 is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Невозможно рассчитать метрики!")
            msg.setInformativeText('Похоже, что одно из изображений не было загружено.')
            msg.setWindowTitle("Ошибка!")
            msg.exec_()
            return
        
        if self.img1.shape != self.img2.shape:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Невозможно рассчитать метрики!")
            msg.setInformativeText('Похоже, что размеры изображений не совпадают.')
            msg.setWindowTitle("Ошибка!")
            msg.exec_()
            return

        SV1 = [i[1] for i in sewar_funcs(self.img1, self.img2)]
        SV2 = [self.MSE_value, self.SSIM_value, self.MS_SSIM_value, self.RMSE_value, self.PSNR_value, self.PSNR_B_value, self.UQI_value, self.SCC_value, self.RASE_value, 
               self.SAM_value, self.VIFP_value, self.ERGAS_value, self.D_lambda_value]
        for i in range(len(SV2)):
            print(SV2[i], SV1[i])
            SV2[i].setText(str(SV1[i]))

        PV1 = [i[1] for i in piq_funcs(self.img1, self.img2)]
        PV2 = [self.IW_SSIM_value, self.DSS_value, self.HaarPSI_value, self.MDSI_value, self.SR_SIM_value, self.FSIM_value, self.VSI_value, self.GMSD_value, self.MS_GMSD_value]
        for i in range(len(PV2)):
            print(SV2[i], SV1[i])
            PV2[i].setText(str(PV1[i]))
        
        self.NPCR_value.setText(str(calculate_npcr(self.img1, self.img2))+" %")
        self.UACI_value.setText(str(calculate_uaci(self.img1, self.img2))+" %")

        '''
        # Очистка сетки метрик
        for i in reversed(range(self.metrics_grid.count())):
            self.metrics_grid.itemAt(i).widget().setParent(None)

        # Отображение метрик
        for metric, value in metrics.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].setText(f"{value:.4f}")
        '''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageLoader()
    window.show()
    sys.exit(app.exec_())
