# -*- coding: utf_8 -*-

import cv2
import threading
import time

# HOG + SVMによる人間検出器をセットアップ
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
is_processing = False  # 関数が実行中かどうかを管理するフラグ

def on_motion_detected(frame):
    global is_processing

    """動体が検知されたときに呼び出される関数"""
    # 人間検出
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    
    # 人間が検出された場合
    if len(boxes) > 0:
        print("Human detected!")
        # 画像を保存
        cv2.imwrite("output.png", frame)

        # 人間の領域を矩形で囲む
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        pass
        # print("No human detected.")

    is_processing = False  # 処理が終わったらフラグをリセット

# 非同期で関数を実行するためのラッパー関数
def run_async(func, *args):
    thread = threading.Thread(target=func, args=args)
    thread.start()

def main():
    global is_processing  # ここでグローバル変数を宣言

    # 動画キャプチャの開始
    cap = cv2.VideoCapture(0)

    # 背景差分を計算するための背景差分法オブジェクト
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        try:
            # フレームをキャプチャ
            ret, frame = cap.read()

            # フレームが正しくキャプチャされたか確認
            if not ret:
                break

            # 背景差分を計算
            fgmask = fgbg.apply(frame)

            # ノイズを減らすためにモーフィング処理（膨張と収縮）を適用
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            # 輪郭を検出
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False

            # 動体を検出したか確認
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # ノイズを除去するために小さな領域は無視
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 動体が検出され、現在処理が実行中でない場合に関数を実行
            if motion_detected and not is_processing:
                is_processing = True  # 処理中フラグを設定
                run_async(on_motion_detected, frame.copy())

            # 結果を表示
            cv2.imshow('Frame', frame)
            cv2.imshow('FG Mask', fgmask)

            # 'q'キーを押すと終了
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
