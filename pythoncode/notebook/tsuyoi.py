import cv2
import torch
import logging
import os
from datetime import datetime
from ultralytics import YOLO, solutions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging
logging.disable(logging.CRITICAL)


def process_video(model_path, video_path, save_dir="../output"):
    """视频处理生成器，返回视频参数和帧数据"""
    # 创建视频专属目录
    video_base = os.path.basename(video_path)
    video_name = os.path.splitext(video_base)[0]
    main_save_path = os.path.join(save_dir, video_name)
    os.makedirs(main_save_path, exist_ok=True)

    model = YOLO(model_path,  verbose=False)
    model.to(device)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logging.error("Error 打开视频错误")
        return

    # 视频参数获取
    fps = capture.get(cv2.CAP_PROP_FPS)
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = h // 2  # 水平线Y坐标

    # 初始化ObjectCounter
    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=[(0, line_y), (w, line_y)],
        names=model.names,
        draw_tracks=True,
        line_thickness=1,
    )

    # 跟踪参数
    seen_ids = set()
    frame_count = 0

    yield (fps, w, h)

    # 视频处理循环
    while capture.isOpened():
        success, frame = capture.read()
        frame_count += 1
        if not success:
            logging.warning("视频帧读取结束")
            break

        # 目标跟踪
        tracks = model.track(frame, persist=True, show=False, device=device, classes=[2, 3, 5, 7])
        # processed_frame = counter.start_counting(frame, tracks)

        # 使用轨迹数据进行跨线检测
        for obj_id, track_history in counter.track_history.items():
            # 至少需要2个轨迹点才能判断移动方向
            if len(track_history) >= 2:
                # 获取最近两个坐标点
                prev_point = track_history[-2]
                current_point = track_history[-1]

                # 解析Y坐标
                prev_y = prev_point[1]
                current_y = current_point[1]

                # 判断是否发生跨线
                if (prev_y > line_y and current_y <= line_y) or \
                        (prev_y < line_y and current_y >= line_y):

                    if obj_id not in seen_ids:
                        # 获取当前框信息
                        for box in tracks[0].boxes:
                            if int(box.id.item()) == obj_id:
                                # 创建对象目录
                                cls_id = int(box.cls.item())
                                class_name = model.names[cls_id]
                                obj_dir = os.path.join(
                                    main_save_path,
                                    f"ID{obj_id}_{class_name}"
                                )
                                os.makedirs(obj_dir, exist_ok=True)

                                # 保存裁剪图像
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                crop = frame[y1:y2, x1:x2]
                                cv2.imwrite(os.path.join(obj_dir, f"ID{obj_id}_{class_name}.jpg"), crop)

                                # 保存完整帧
                                frame_filename = f"frame_{frame_count:06d}.jpg"
                                cv2.imwrite(os.path.join(obj_dir, frame_filename), frame)

                                seen_ids.add(obj_id)
                                break  # 找到对应ID后跳出循环

        processed_frame = counter.start_counting(frame, tracks)  # 传入原始frame进行绘制

        counts = {
            "in": counter.in_counts,
            "out": counter.out_counts,
            "class_wise": counter.class_wise_count.copy()
        }
        yield (processed_frame, counts)

    capture.release()


import time


def display_and_print(generator):
    """显示画面并打印计数信息"""
    try:
        # 获取视频基础参数
        fps, w, h = next(generator)
    except StopIteration:
        return

    # 计算帧间隔，0.5秒对应的帧数
    interval_frames = int(fps * 0.5)
    frame_count = 0
    last_print_time = time.time()

    while True:
        try:
            frame, counts = next(generator)
        except StopIteration:
            break

        # 显示画面
        cv2.imshow('output', frame)

        # 控制打印频率
        frame_count += 1
        current_time = time.time()
        if current_time - last_print_time >= 1:
            # 打印信息
            print(f"\n当前统计 (时间: {current_time:.2f}):")
            print(f"总进入(下到上): {counts['in']} | 总离开(上到下): {counts['out']}")
            for cls_name, cls_counts in counts['class_wise'].items():
                print(f"{cls_name} 进入: {cls_counts.get('IN', 0)} | 离开: {cls_counts.get('OUT', 0)}")

            # 更新上次打印时间
            last_print_time = current_time

        # 退出控制，按q键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def combined_processing(generator, output_path):
    def print_counts(counts):
        """打印当前统计信息"""
        print(f"\n当前统计 (时间: {datetime.now().strftime('%H:%M:%S')}):")
        print(f"总进入(下到上): {counts['in']} | 总离开(上到下): {counts['out']}")
        for cls_name, cls_counts in counts['class_wise'].items():
            print(f"{cls_name} 进入: {cls_counts.get('IN', 0)} | 离开: {cls_counts.get('OUT', 0)}")

    try:
        fps, w, h = next(generator)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not video_writer.isOpened():
            raise IOError("无法初始化视频编码器")

        last_print_time = time.time()
        frame_count = 0

        while True:
            frame, counts = next(generator)

            cv2.imshow('Processed Video', frame)

            video_writer.write(frame)
            frame_count += 1

            current_time = time.time()
            if current_time - last_print_time >= 1:
                print_counts(counts)  # 调用内部函数
                last_print_time = current_time

            # 退出控制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except StopIteration:
        print(f"视频处理完成，共处理 {frame_count} 帧")
    finally:
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    gen = process_video(
        model_path="yolo11n.pt",
        video_path="../video.mp4",
        save_dir="../output"
    )

    combined_processing(gen, "output1.mp4")