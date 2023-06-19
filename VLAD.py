import customtkinter
import tkinter
import h5py
import tensorflow
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from scipy.ndimage import center_of_mass
import math
import numpy as np
import cv2


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.model = load_model('mnist_1_full.h5')  #Загрузка обученной модели

        self.geometry("1280x720")   #Определение размеров окна 
        self.title("OCR")   #Определение загаловка окна

        customtkinter.set_appearance_mode("dark")   #Определение цветовой темы граф. интерфейса
        customtkinter.set_default_color_theme("blue")

        self.main_frame = customtkinter.CTkFrame(master=self)   #Инициализация главного Frame и Label
        self.main_label = customtkinter.CTkLabel(master=self.main_frame, text="OCR")

        self.main_frame.grid_rowconfigure(0, weight=1)  #Создание сетки внутри главного Frame для позиционирования
        self.main_frame.grid_rowconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.img_frame = customtkinter.CTkFrame(master=self.main_frame) #Инициализация фрейма и метки для изображения
        self.img_label = customtkinter.CTkLabel(master=self.img_frame, text="")

        self.ans_frame = customtkinter.CTkFrame(master=self.main_frame) #Инициализация фрейма и меток для ответа (результата распознавания)
        self.ans_label = customtkinter.CTkLabel(master=self.ans_frame, justify=LEFT, font=("Roboto", 24), text="1. Выбрать файл \n2. Обработать файл")
        self.ans_info_label = customtkinter.CTkLabel(master=self.ans_frame, justify=LEFT, font=("Roboto", 16), text="")

        self.btn_frame = customtkinter.CTkFrame(master=self.main_frame) #Инициализация фрейма и кнопок для открытия и обработки файла
        self.open_file = customtkinter.CTkButton(master=self.btn_frame, command=self.open_file, text="Открыть файл")
        self.process_file = customtkinter.CTkButton(master=self.btn_frame, command=self.process_file, text="Обработать файл")

        self.main_frame.pack(padx=20, pady=20, expand=True, fill=BOTH) #Расположение главного Frame

        self.img_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")   #Расположение фрейма для изображения
        self.img_label.pack(expand=True, fill=BOTH)
        self.img_label.pack_propagate(False)

        self.ans_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")   #Расположение фрейма для ответа
        self.ans_label.pack(expand=True, fill=BOTH)
        self.ans_info_label.pack(expand=True, fill=BOTH)

        self.btn_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew") #Расположение фрейма для кнопок
        self.open_file.pack(padx=10, pady=10)
        self.open_file.pack_propagate(False)
        self.process_file.pack(padx=10, pady=10)
        self.process_file.pack_propagate(False)


    def open_file(self): #Метод для открытия файла, срабатывает по нажатию на кнопку "Открыть файл"
        global filepath #Глобальная переменная, содержащая путь к файлу
        filepath = filedialog.askopenfile()
        if filepath != None:
            img_path = filepath.name
            
        dim = (400, 400)
        image = Image.open(img_path)    #Открытие изображения и изменение размера относительно img_label
        image = image.resize(dim, Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image)   #Необходимо для отображения изображения в img_label

        self.img_label.configure(image=photo)   #Обновление аргумента image в img_label


    def show_error(self):   #Метод для обработки простейших ошибок, срабатывает если путь к файлу пустой
        popup = customtkinter.CTkToplevel() #Инициализация окна с ошибкой
        popup.attributes('-topmost', True)  #Расположение поверх остальных окон
        popup.grab_set()    #Блокировка нажатия мыши на все остальные окна
        popup.title("Ошибка")
        popup.geometry("300x100")

        popup_label = customtkinter.CTkLabel(master=popup, text="Сначала необходимо выбрать файл")
        popup_label.pack(padx=20, pady=20)

        popup_btn = customtkinter.CTkButton(master=popup, text="OK", command=popup.destroy) #Кнопка для закрытия окна с ошибкой
        popup_btn.pack()


    def get_best_shift(self, img):  #Вычисляет центр масс контура цифры, возвращает направления сдвига. Нужен для совмещения центра масс с центром итоговой картинки
        cy,cx = center_of_mass(img) #Метод из библ. SciPy
        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)  #Вычисление направления сдвига
        shifty = np.round(rows/2.0-cy).astype(int)
        return shiftx,shifty
    

    def shift(self, img, sx, sy): #Выполняет сдвиг картинки, возвращает картинку со сдвигом
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]]) #Матрица сдвига
        shifted = cv2.warpAffine(img,M,(cols,rows)) #warpAffine выполняет Афинное преобразование (новые координаты в прежнем базисе)
        return shifted


    def calc_dif_brightness(self, img):
        dif_brightness = np.max(img) / np.min(img)
        return int(dif_brightness)

    def preproc_file(self): #Препроцессинг входного изображения
        width = 300
        height = 300
        dim = (width, height)
        img = cv2.imread(filepath.name, cv2.IMREAD_GRAYSCALE) #Чтение изображения в чб
        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA) #Изменение размеров изображения
        
        i_show = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow("Colored", i_show)

        if self.calc_dif_brightness(img) < 3:
            alpha = 150
        else:
            alpha = 128

        print("BR - ", self.calc_dif_brightness(img))
        print("ALPHA - ", alpha)

        (thresh, gray) = cv2.threshold(img, alpha, 255, cv2.THRESH_BINARY_INV)

        i_show = cv2.resize(gray, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow("B&W", i_show)


        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 350:
                brect = cv2.boundingRect(cnt)
                x,y,w,h = brect
                image = gray[y:y+h, x:x+w]
                cv2.rectangle(gray, brect, (255,255,255), 1)
                i+=1

        i_show = cv2.resize(gray, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow("RECT", i_show)

        i_show = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow(f"{i}CROPPED", i_show)

        rows,cols = image.shape

        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            image = cv2.resize(image, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            image = cv2.resize(image, (cols, rows))
        

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))#Увелечение изображения до формата 28 x 28. Заполнение черным
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')

        image = 255 - image
        image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)
        image = 255 - image

        i_show = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow("EnFINAL", i_show)

        shiftx,shifty = self.get_best_shift(image) #Сдвиг цифры, чтобы центр изображения совпадал с центром масс
        shifted = self.shift(image,shiftx,shifty)
        image = shifted
        
        image = image / 255.0

        i_show = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
        cv2.imshow("FINAL", i_show)

        img = np.array(image).reshape(-1, 28, 28, 1) #Преобразование изображения в вектор для работы с СНС

        prediction = self.model.predict(img) #Результат работы СНС
        ans = np.argmax(prediction) #Итоговый ответ предсказания


        self.ans_label.configure(text="Ответ: " + str(ans)) #Обновление метки ans_label
        self.ans_info_label.configure(text="Дополнительная информация: \n" + str(prediction)) #Обновление метки ans_info_label


    def process_file(self): #Метод для обрабтки файла, срабатывает по кнопке "Обработать файл"
        try:
            filepath    #Проверка существования пути к файлу
        except:
            self.show_error()   #Вывод сообщения об ошибке
        else:
            self.preproc_file()  #Препроцессинг входного изображения

            

if __name__ == "__main__":
    app = App()
    app.mainloop()