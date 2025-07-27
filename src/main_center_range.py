import cv2
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=True):
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {input_path}")

        # 获取视频属性
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.display_preview = display_preview
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 视频输出设置
        fourcc_writer = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc_writer, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"Error creating the output video: {output_path}")

        # 运动检测变量初始化
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 15  # MHI生命周期

        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

        # 聚类参数
        self.min_cluster_size = 3  # 形成聚类所需的最小矩形框数量
        self.distance_threshold = 100  # 聚类中心点距离阈值(像素)

        # 跟踪数据结构
        self.tracking_targets = []  # 存储当前跟踪目标

    def process_video(self):
        progress = tqdm(total=self.total_frames, desc="Processing Video")

        """主循环开始"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 预处理：灰度化 + 高斯模糊
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # 运动检测
            mhi_visual = self._visualize_mhi()
            processed_frame = frame.copy()
            # MHI叠加到原始帧
            cv2.addWeighted(processed_frame, 0.7, mhi_visual, 0.3, 0, processed_frame)

            # 检测运动物体
            detected_objects = self._detect_motion(frame, gray)

            # 聚类合并运动目标
            clustered_objects = self._cluster_objects(detected_objects)

            # 提取跟踪点和邻域范围
            tracking_points, neighborhood_radii = self._extract_tracking_points(clustered_objects)

            # 绘制检测结果和跟踪点
            for i, (x, y, w, h) in enumerate(clustered_objects):
                # 绘制目标边界框
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 绘制跟踪点
                if i < len(tracking_points):
                    center_x, center_y = tracking_points[i]
                    radius = neighborhood_radii[i] if i < len(neighborhood_radii) else 5

                    # 绘制中心点
                    cv2.circle(processed_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                    # 绘制邻域范围（用于MeanShift）
                    cv2.circle(processed_frame, (int(center_x), int(center_y)), int(radius), (255, 0, 0), 1)

                    # 添加ID标签
                    cv2.putText(processed_frame, f'T{i}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 写入输出
            self.out.write(processed_frame)

            # 预览显示
            if self.display_preview:
                cv2.imshow("Motion Detection Preview", processed_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            progress.update(1)
        """主循环结束"""

        progress.close()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def _detect_motion(self, frame, gray):
        try:
            # 应用背景减除
            foreground_mask = self.bg_subtractor.apply(gray)

            # 形态学操作降噪
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

            # 首帧处理
            if self.prev_gray is None:
                self.prev_gray = gray
                return []

            # 帧间差分
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)

            # 背景掩码处理
            _, bg_mask = cv2.threshold(foreground_mask, 127, 1, cv2.THRESH_BINARY)

            # 结合运动掩码和背景掩码
            combined_mask = cv2.bitwise_and(motion_mask, bg_mask)

            # 更新MHI
            self.mhi = np.where(combined_mask == 1, self.tau, np.maximum(self.mhi - 1, 0))

            # 从MHI提取运动对象掩码
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            final_mask = cv2.bitwise_and(mhi_mask, foreground_mask)

            # 查找轮廓
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 提取检测对象
            detected_objects = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # 过滤小噪点
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_objects.append((x, y, w, h))

            self.prev_gray = gray
            return detected_objects
        except Exception as e:
            print(f"Error during motion detection: {e}")
            return []

    def _cluster_objects(self, detected_objects):
        """使用K-means聚类合并运动目标"""
        if len(detected_objects) < self.min_cluster_size:
            # 对象太少，不需要聚类
            return detected_objects

        # 计算每个矩形的中心点
        centers = []
        for (x, y, w, h) in detected_objects:
            centers.append([x + w/2, y + h/2])

        # 转换为NumPy数组
        centers = np.array(centers)

        # 确定聚类数量（基于对象密度）
        n_clusters = self._determine_cluster_count(centers)

        if n_clusters <= 1:
            # 不需要聚类或只有一个聚类
            return self._merge_all_objects(detected_objects)

        # 应用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(centers)

        # 合并每个聚类的矩形
        clustered_objects = []
        for cluster_id in range(n_clusters):
            cluster_rects = [detected_objects[i] for i in range(len(detected_objects))
                             if labels[i] == cluster_id]

            if cluster_rects:
                merged_rect = self._merge_rectangles(cluster_rects)
                clustered_objects.append(merged_rect)

        return clustered_objects

    def _determine_cluster_count(self, centers):
        """基于点密度确定最佳聚类数量"""
        # 计算点之间的平均距离
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)

        if not distances:
            return 1

        avg_distance = np.mean(distances)

        # 根据平均距离确定聚类数量
        if avg_distance < self.distance_threshold / 3:
            return 1  # 所有点都很近，合并为一个聚类
        elif avg_distance < self.distance_threshold:
            return min(3, len(centers) // 2)  # 中等密度，适度聚类
        else:
            return min(5, len(centers))  # 低密度，更多聚类

    def _merge_rectangles(self, rects):
        """合并一组矩形为一个边界框"""
        x_min = min(r[0] for r in rects)
        y_min = min(r[1] for r in rects)
        x_max = max(r[0] + r[2] for r in rects)
        y_max = max(r[1] + r[3] for r in rects)

        width = x_max - x_min
        height = y_max - y_min

        # 添加边界扩展（10%的尺寸）
        expand_x = int(width * 0.1)
        expand_y = int(height * 0.1)

        # 应用边界扩展
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(self.width, x_max + expand_x)
        y_max = min(self.height, y_max + expand_y)

        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _merge_all_objects(self, rects):
        """当所有对象应该合并时处理"""
        if len(rects) == 0:
            return []

        # 计算所有矩形的平均中心
        avg_center_x = sum(r[0] + r[2]/2 for r in rects) / len(rects)
        avg_center_y = sum(r[1] + r[3]/2 for r in rects) / len(rects)

        # 计算平均尺寸
        avg_width = sum(r[2] for r in rects) / len(rects)
        avg_height = sum(r[3] for r in rects) / len(rects)

        # 创建合并矩形
        x = int(avg_center_x - avg_width/2)
        y = int(avg_center_y - avg_height/2)
        w = int(avg_width * 1.2)  # 稍微扩大一点
        h = int(avg_height * 1.2)

        return [(max(0, x), max(0, y), min(self.width - x, w), min(self.height - y, h))]

    def _extract_tracking_points(self, clustered_objects):
        """从聚类后的目标中提取跟踪点和邻域范围"""
        tracking_points = []  # 跟踪点列表 (x, y)
        neighborhood_radii = []  # 邻域半径列表

        for (x, y, w, h) in clustered_objects:
            # 计算跟踪点 - 使用中心点
            center_x = x + w / 2
            center_y = y + h / 2
            tracking_points.append((center_x, center_y))

            # 计算邻域范围
            # 方法1: 基于目标大小计算半径
            # radius = min(w, h) * 0.5  # 取较小尺寸的一半

            # 方法2: 基于目标对角线计算半径
            # diagonal = np.sqrt(w**2 + h**2)
            # radius = diagonal * 0.4

            # 方法3: 综合考虑尺寸和运动历史
            # 这里使用宽度和高度的平均值，乘以0.6作为半径
            radius = (w + h) / 2 * 0.6

            # 确保半径在合理范围内
            min_radius = 10
            max_radius = min(self.width, self.height) * 0.2
            radius = np.clip(radius, min_radius, max_radius)

            neighborhood_radii.append(radius)

        return tracking_points, neighborhood_radii

    def _visualize_mhi(self):
        """将MHI可视化为彩色图像"""
        mhi_norm = cv2.normalize(self.mhi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(mhi_norm, cv2.COLOR_GRAY2BGR)


def parse_args():
    parser = argparse.ArgumentParser(description='Video motion detection with tracking points extraction')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output video file')
    parser.add_argument('--display_preview', action='store_true', default=True,
                        help='Display preview of the processed video')
    parser.add_argument('--min_cluster', type=int, default=3,
                        help='Minimum number of rectangles to form a cluster')
    parser.add_argument('--cluster_dist', type=int, default=100,
                        help='Distance threshold for clustering (pixels)')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 使用命令行参数
    input_video = args.input
    output_video = args.output
    if not input_video or not output_video:
        raise ValueError("Input and output video paths must be specified.")

    # 创建VideoProcessor实例并处理视频
    processor = VideoProcessor(input_video, output_video,
                               display_preview=args.display_preview)
    processor.min_cluster_size = args.min_cluster
    processor.distance_threshold = args.cluster_dist
    processor.process_video()
    print("Processing completed! Output saved to", output_video)