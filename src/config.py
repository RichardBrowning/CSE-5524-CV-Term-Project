# 全局配置参数
class Config:
    # motion tracking parameters
    MHI_TAU = 15  # MHI decay time (seconds)
    BG_HISTORY = 500  # Background subtractor history frames
    MIN_CONTOUR_AREA = 100  # Minimum contour area
    MAX_CONTOUR_AREA = 10000  # Maximum contour area

    # tracking parameters
    TRACKER_MAX_LOST = 10  # Maximum lost frames for tracking
    KLT_WIN_SIZE = (15, 15)  # KLT window size
    KLT_MAX_LEVEL = 2  # KLT pyramid levels
    MIN_FEATURES = 5  # Minimum features for tracking

    # classification parameters
    VEHICLE_MIN_AREA = 2000  # Minimum vehicle area
    VEHICLE_ASPECT_RATIO = 1.5  # Vehicle aspect ratio threshold

    # visualization parameters
    DRAW_HISTORY = True  # Draw trajectory
    DRAW_FEATURES = True  # Draw feature points
    HISTORY_LENGTH = 100  # Trajectory history length