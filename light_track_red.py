'''PC'''

import serial
import cv2 as cv
import numpy as np

current_segment = 0               # 当前线段索引（0~3）
interp_points = []                # 存储中点插值点列表
current_target_idx = 0            # 当前目标插值点索引

arrival_threshold = 5             # 到达判定阈值5
interp_step_size = 10             # 插值点间距10
max_step = 8                      # 最大步长
scale = 0.8                       # 比例系数

global_contours = []

# 红色激光点预处理函数
def process_red(img, ifshow=False):
    imgblur = cv.GaussianBlur(img, (5,5), 5)
    
    # 红色范围（增加容错）
    lower1 = np.array([0, 43, 130])    
    upper1 = np.array([10, 255, 255])  
    lower2 = np.array([145, 43, 90])  
    upper2 = np.array([179, 255, 255])
    
    imgHSV = cv.cvtColor(imgblur, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(imgHSV, lower1, upper1)
    mask2 = cv.inRange(imgHSV, lower2, upper2)
    mask = cv.bitwise_or(mask1, mask2)  # 合并两个红色范围
    
    # 形态学处理优化
    kernel = np.ones((5,5), np.uint8)
    imgdilate = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=5)
    
    if ifshow:
        cv.imshow("Red Mask", imgdilate)  
    return imgdilate

# 处理红色激光点函数
def find_red(img):
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

# 边款预处理函数
def preprocess_frame(img):
    """提取黑色部分并返回二值化图像（白色=黑色区域，黑色=背景）"""

    # 转换到 HSV 色彩空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 定义黑色的HSV范围
    lower_black = np.array([0, 0, 137])
    upper_black = np.array([179, 134, 255])

    # 生成黑色区域的掩码（白色=黑色，黑色=背景）
    mask = cv.inRange(hsv, lower_black, upper_black)

    # 形态学处理：去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    imgdilate = cv.dilate(mask,kernel,2)
    result = cv.erode(imgdilate,kernel,1)
    
    return result

# 处理矩形边框函数
def find_frame(img,ifshow = True):
    '''
    返回大列表，每个元素为一个小列表，代表一个轮廓
    小列表元素依次为:角点坐标，轮廓面积，轮廓周长
    '''
    contours,hierarchy = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) # 检测外部边缘
    
    finalcontours = [] # [approx,area,peri]
    
    for cnt in contours:
        area = cv.contourArea(cnt) # 获取当前轮廓围成的面积
        peri = cv.arcLength(cnt,True) # 闭合轮廓周长
        approx = cv.approxPolyDP(cnt,0.02 * peri,True) # 轮廓角点坐标

        if area > 500 and area < 300000 and len(approx) == 4:
            bbox = cv.boundingRect(approx) # 角点外接矩形    
            # 多边形拟合（基于周长和角点）
            objCor = len(approx) # 角点个数
            finalcontours.append([approx,area,peri])    
            # 绘制拟合的多边形轮廓（绿色）
            cv.drawContours(imgcontours, [approx], -1, (0, 255, 0), 1)
            # 添加文字
            for i in range(len(approx)):
                x,y = approx[i][0]
                cv.putText(imgcontours,f"{approx[i]}",(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    
    finalcontours.sort(key=lambda x: x[1]) # 按面积升序排序

    if (ifshow):
        for cnt in finalcontours:
            print("周长",cnt[1])
            print("面积",cnt[2])

    return finalcontours
    
# 处理红色激光和边框的位置关系
def red_location(x_frame, y_frame, red_place):
    # 提取圆心坐标和半径
    center_x, center_y, radius = red_place

    # 提取内圈和外圈的角点坐标
    inner_x = x_frame[:4]
    inner_y = y_frame[:4]
    outer_x = x_frame[4:]
    outer_y = y_frame[4:]

    # 计算内边框和外边框的边界
    inner_left = min(inner_x)
    inner_right = max(inner_x)
    inner_top = min(inner_y)
    inner_bottom = max(inner_y)

    outer_left = min(outer_x)
    outer_right = max(outer_x)
    outer_top = min(outer_y)
    outer_bottom = max(outer_y)

    # 判断圆心是否在内边框内
    if (inner_left <= center_x <= inner_right) and (inner_top <= center_y <= inner_bottom):
        cv.putText(imgcontours,"inner",(0,80),cv.FONT_HERSHEY_COMPLEX,3,(255,0,0),1)
    # 判断圆心是否在内外边框之间
    elif (outer_left <= center_x <= outer_right) and (outer_top <= center_y <= outer_bottom):
        cv.putText(imgcontours,"between",(0,80),cv.FONT_HERSHEY_COMPLEX,3,(255,0,0),1)
    # 圆心在外边框之外
    else:
        cv.putText(imgcontours,"outer",(0,80),cv.FONT_HERSHEY_COMPLEX,3,(255,0,0),1)

# 串口发送数据
def send_xy_offset(ser, x, y):
    # 转换为带符号16位整数
    x_int = int(x)
    y_int = int(y)
    
    # 拆分为高低字节（大端序）
    x_high, x_low = (x_int >> 8) & 0xFF, x_int & 0xFF
    y_high, y_low = (y_int >> 8) & 0xFF, y_int & 0xFF
    
    # 计算校验和
    checksum = (x_low + y_low) & 0xFF
    
    # 构建数据包
    packet = bytes([x_high, x_low, y_high, y_low, checksum])
    
    # 通过串口发送
    ser.write(packet)

# 角点重排序
def sort_corners_robust(pts):
    """改进版：对任意旋转的四边形角点按顺时针排序 [左上, 右上, 右下, 左下]"""
    pts = np.array(pts).reshape(4, 2)
    
    # 1. 计算几何中心
    center = np.mean(pts, axis=0)
    
    # 2. 将坐标转换为相对中心的向量
    vecs = pts - center
    
    # 3. 计算极角（atan2 返回 [-π, π]）
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    
    # 4. 按极角排序（顺时针）
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]
    
    # 5. 固定顺序：左上→右上→右下→左下
    # 找到 y 最小的两个点（上边）中最左的作为左上
    top_two = sorted_pts[np.argsort(sorted_pts[:, 1])[:2]]
    tl = top_two[np.argmin(top_two[:, 0])]  # 左上
    tr = top_two[np.argmax(top_two[:, 0])]  # 右上
    
    # 剩余两个点中，x 较大的为右下
    remaining = [p for p in sorted_pts if not (p == tl).all() and not (p == tr).all()]
    br = remaining[np.argmax(remaining, axis=0)[0]]  # 右下
    bl = remaining[np.argmin(remaining, axis=0)[0]]  # 左下
    
    return np.array([tl, tr, br, bl])

# 两点插值
def generate_midpoints_interp(p1, p2):
    """生成两个中点之间的插值点"""
    points = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = max(1, int((dx**2 + dy**2)**0.5))
    steps = max(1, dist // interp_step_size)
    
    for i in range(steps + 1):
        t = i / steps
        x = int(p1[0] + t * dx)
        y = int(p1[1] + t * dy)
        points.append((x, y))
    return points

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

ser = serial.Serial('COM7', baudrate=115200, # 需要改变串口标号
                    bytesize=8, stopbits=1, 
                    parity='N', timeout=1)

while True:
    '''读取摄像头帧'''
    success, img = cap.read()
    if not success:
        break
    imgcontours = img.copy()

    '''边框检测'''
    img_pre_frame = preprocess_frame(img)
    finalcontours = find_frame(img_pre_frame)
    if len(finalcontours) == 2:  # 如果检测到有效边框
        global_contours = finalcontours  # 更新全局变量

    '''角点提取与重排序'''
    all_sorted_corners = []  # 存储所有边框的排序后角点
    main_corners = []        # 存储4个中点坐标(左上，右上，右下，左下)

    for i, cnt in enumerate(global_contours):
        if len(cnt[0]) == 4:  
            # 提取原始角点并重排序
            corners = [point[0] for point in cnt[0]]
            sorted_corners = sort_corners_robust(corners) 
            all_sorted_corners.append(sorted_corners)

            # 标注原始角点顺序
            for j, (x, y) in enumerate(sorted_corners):
                cv.putText(imgcontours, f"{i}-{j}", (x, y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    '''计算中点'''
    if len(all_sorted_corners) == 2:
        for i in range(4):
            # 计算对应角点的中点
            x_mid = int((all_sorted_corners[0][i][0] + all_sorted_corners[1][i][0]) / 2)
            y_mid = int((all_sorted_corners[0][i][1] + all_sorted_corners[1][i][1]) / 2)
            main_corners.append([x_mid, y_mid])

            # 在四个中点标注
            cv.drawMarker(imgcontours, (x_mid, y_mid), (0, 255, 255), 
                         markerType=cv.MARKER_CROSS, markerSize=15, thickness=2)
            cv.putText(imgcontours, str(i), (x_mid+10, y_mid+10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    '''红色激光检测'''
    imgred = process_red(img)
    cir = find_red(imgred)

    ''' 路径规划部分 '''

    if len(main_corners) == 4 and cir is not None:
        # 初始化或重新生成插值点
        if not interp_points:
            prev_idx = (current_segment - 1) % 4
            next_idx = current_segment
            interp_points = generate_midpoints_interp(
                main_corners[prev_idx],  # 上一个中点
                main_corners[next_idx]   # 当前中点
            )
            current_target_idx = 0
        
        # 获取当前目标点
        target_x, target_y = interp_points[current_target_idx]
        
        # 检查是否到达
        distance = ((target_x - cir[0])**2 + (target_y - cir[1])**2)**0.5
        if distance <= arrival_threshold:
            if current_target_idx < len(interp_points) - 1:
                current_target_idx += 1
            else:
                current_segment = (current_segment + 1) % 4
                interp_points = []
        
        # 计算移动指令
        dx = target_x - cir[0]
        dy = target_y - cir[1]
        step_dist = max(1, (dx**2 + dy**2)**0.5)
        step_x = int(dx * min(max_step, step_dist) / step_dist * scale)
        step_y = int(dy * min(max_step, step_dist) / step_dist * scale)
        
        send_xy_offset(ser, step_x, step_y)

        # 可视化
        cv.circle(imgcontours, (target_x, target_y), 4, (0,255,255), -1)  # 黄色：当前目标中点
        cv.arrowedLine(imgcontours, (cir[0], cir[1]), 
                    (cir[0]+step_x, cir[1]+step_y), (0,255,0), 1)  # 绿色：移动方向

    '''图像展示'''
    cv.imshow("Original", img)
    cv.imshow("Contours", imgcontours)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
