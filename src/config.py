# 全局配置参数
class Config:
    # 运动检测参数
    MHI_TAU = 15  # MHI衰减时间（秒）
    BG_HISTORY = 500  # 背景减除器历史帧数
    MIN_CONTOUR_AREA = 100  # 最小轮廓面积
    MAX_CONTOUR_AREA = 10000  # 最大轮廓面积
    
    # 跟踪参数
    TRACKER_MAX_LOST = 10  # 目标丢失阈值
    KLT_WIN_SIZE = (15, 15)  # KLT窗口大小
    KLT_MAX_LEVEL = 2  # KLT金字塔层数
    MIN_FEATURES = 5  # 最小特征点数
    
    # 分类参数
    VEHICLE_MIN_AREA = 2000  # 车辆最小面积
    VEHICLE_ASPECT_RATIO = 1.5  # 车辆宽高比阈值
    
    # 可视化参数
    DRAW_HISTORY = True  # 是否绘制轨迹
    DRAW_FEATURES = True  # 是否绘制特征点
    HISTORY_LENGTH = 100  # 轨迹历史长度