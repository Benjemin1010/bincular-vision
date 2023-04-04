import PySimpleGUI as sg  #pip install pysimplegui
import cv2  #pip install opencv-python
import numpy as np #pip install numpy

# 背景色
sg.theme('LightGreen')

# 定义窗口布局
layout = [
    [sg.Image(filename='', key='image')],
    [sg.Radio('None', 'Radio', True, size=(10, 1))],
    [sg.Radio('threshold', 'Radio', size=(10, 1), key='thresh'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(40, 15), key='thresh_slider')],
    [sg.Radio('canny', 'Radio', size=(10, 1), key='canny'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='canny_slider_a'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='canny_slider_b')],
    [sg.Radio('contour', 'Radio', size=(10, 1), key='contour'),
     sg.Slider((0, 255), 128, 1, orientation='h', size=(20, 15), key='contour_slider'),
     sg.Slider((0, 255), 80, 1, orientation='h', size=(20, 15), key='base_slider')],
    [sg.Radio('blur', 'Radio', size=(10, 1), key='blur'),
     sg.Slider((1, 11), 1, 1, orientation='h', size=(40, 15), key='blur_slider')],
    [sg.Radio('hue', 'Radio', size=(10, 1), key='hue'),
     sg.Slider((0, 225), 0, 1, orientation='h', size=(40, 15), key='hue_slider')],
    [sg.Radio('enhance', 'Radio', size=(10, 1), key='enhance'),
     sg.Slider((1, 255), 128, 1, orientation='h', size=(40, 15), key='enhance_slider')],
    [sg.Button('Exit', size=(10, 1))]
]

# 窗口设计
window = sg.Window('OpenCV实时图像处理',
                   layout,
                   location=(800, 400),
                   finalize=True)
# 打开内置摄像头
cap = cv2.VideoCapture(0)
cap.set(3,2560)
cap.set(4,720)
while True:
    event, values = window.read(timeout=0, timeout_key='timeout')

    # 实时读取图像
    ret, frame = cap.read()

    if values['thresh']:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
        frame = cv2.threshold(frame, values['thresh_slider'], 255, cv2.THRESH_BINARY)[1]
    if values['canny']:
        frame = cv2.Canny(frame, values['canny_slider_a'], values['canny_slider_b'])
    if values['contour']:
        hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = cv2.GaussianBlur(hue, (21, 21), 1)
        hue = cv2.inRange(hue, np.array([values['contour_slider'], values['base_slider'], 40]),
                      np.array([values['contour_slider'] + 30, 255, 220]))
        cnts= cv2.findContours(hue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)
    if values['blur']:
        frame = cv2.GaussianBlur(frame, (21, 21), values['blur_slider'])
    if values['hue']:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[:, :, 0] += int(values['hue_slider'])
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    if values['enhance']:
        enh_val = values['enhance_slider'] / 40
        clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #GUI实时更新
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['image'].update(data=imgbytes)

    if event == 'Exit' or event is None:
        break

window.close()