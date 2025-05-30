'''PC'''
import serial
import cv2 as cv
import numpy as np

# 串口发送数据
def send_xy_offset(ser, x, y, flag=0):
    """
    发送带标志位的坐标数据
    参数:
        ser: 串口对象
        x: x轴偏移量
        y: y轴偏移量
        flag: 标志位 (0或1, 默认为0)
    """
    # 转换为带符号16位整数
    x_int = int(x)
    y_int = int(y)
    
    # 拆分为高低字节（大端序）
    x_high, x_low = (x_int >> 8) & 0xFF, x_int & 0xFF
    y_high, y_low = (y_int >> 8) & 0xFF, y_int & 0xFF
    
    # 计算校验和（仅对坐标数据校验）
    checksum = (x_low + y_low) & 0xFF
    
    # 构建数据包（原4字节坐标 + 校验和 + 标志位）
    packet = bytes([x_high, x_low, y_high, y_low, checksum, flag & 0x01])
    
    # 通过串口发送
    ser.write(packet)

# 绿色激光点处理函数
def process_green(img,ifshow = False):
    imgblur = cv.GaussianBlur(img,(5,5)，5)

    # 绿色颜色范围
    lower = np.array([35,43,110])
    upper = np.array([90,255,255])
    # 转HSV
    imgHSV = cv.cvtColor(imgblur,cv.COLOR_BGR2HSV)
    # 创建蒙版
    mask = cv.inRange(imgHSV,lower,upper)
    # 膨胀
    kernel = np.ones((5,5), np.uint8)
    imgdilate = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    if (ifshow):
        cv.imshow("dilate_green",imgdilate)  

    return imgdilate

# 红色激光点预处理函数
def process_red(img, ifshow=False):
    imgblur = cv.GaussianBlur(img, (5,5), 5)
    
    # 扩展红色范围（增加容错）
    lower1 = np.array([0, 43, 120])    
    upper1 = np.array([10, 255, 255])  
    lower2 = np.array([145, 43, 120])  
    upper2 = np.array([179, 255, 255])
    
    imgHSV = cv.cvtColor(imgblur, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(imgHSV, lower1, upper1)
    mask2 = cv.inRange(imgHSV, lower2, upper2)
    mask = cv.bitwise_or(mask1, mask2)  # 合并两个红色范围
    
    # 形态学处理优化
    kernel = np.ones((5,5), np.uint8)
    imgdilate = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)
    imgdilate = cv.dilate(imgdilate,kernel,2)
    
    if ifshow:
        cv.imshow("Red Mask", imgdilate)  
    return imgdilate

# 处理激光点函数
def find_light(img):
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) # 检测外部边缘

    # 检查是否检测到角点
    if contours is not None:
        for contour in contours:
            area = cv.contourArea(contour) # 获取当前轮廓围成的面积
            peri = cv.arcLength(contour,True) # 闭合轮廓周长
            approx = cv.approxPolyDP(contour,0.02 * peri,True) # 轮廓角点坐标
            
            # print(area)

            if area > 5:
                (x, y),r = cv.minEnclosingCircle(contour)
                x = int(x)
                y = int(y)
                r = int(r)
                # 绘制圆形
                cv.circle(imgcontours,(x,y),r,(0,0,255),1) 
                cv.putText(imgcontours,f"{x},{y}",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
                
                # print(f"圆心坐标:{(x,y)},半径:{r}")
        
                return x,y,r
    else:
        print("no corners found")  
        return -1,-1,-1

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

ser = serial.Serial('COM7', baudrate=115200, 
                    bytesize=8, stopbits=1, 
                    parity='N', timeout=1)

# 追踪参数
scale = 0.6
max_dis = 5
max_step = 30  

while True:
    '''读取摄像头帧'''
    success, img = cap.read()
    if not success:
        break
    imgcontours = img.copy()

    '''激光点检测'''
    imgred = process_red(img)
    cir_red = find_light(imgred)
    imggreen = process_green(img)
    cir_green = find_light(imggreen)

    '''追踪控制'''
    if cir_red and not cir_green:
        send_xy_offset(ser, 0, 0, 0)
    elif not cir_red and cir_green:
        send_xy_offset(ser, 0, 0, 0)
    elif cir_red and cir_green:
        dx = cir_red[0] - cir_green[0]
        dy = cir_red[1] - cir_green[1]
        dist = (dx**2 + dy**2)**0.5
        
        # 动态判断接近状态
        if dist < max_dis:
            if_close = 0
            step_x = 0
            step_y = 0
        else:
            if_close = 0
            step_x = int(dx * scale)
            step_y = int(dy * scale)
        
        send_xy_offset(ser, step_x, step_y, if_close)
        print(f"dist:{dist:.1f} Δx:{step_x} Δy:{step_y} close:{if_close}")

    '''图像展示'''
    cv.imshow("Original", img)
    cv.imshow("Contours", imgcontours)

    if cv.waitKey(1) & 0xFF == ord('q'):
        send_xy_offset(ser, 0, 0, 0)
        break
