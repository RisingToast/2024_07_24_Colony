import pandas as pd


# epoch: 학습의 에포크 번호
# train/box_loss: 훈련 데이터의 박스 손실
# train/cls_loss: 훈련 데이터의 클래스 손실
# train/dfl_loss: 훈련 데이터의 DFL 손실
# metrics/precision(B): B 클래스의 정밀도
# metrics/recall(B): B 클래스의 재현율
# metrics/mAP50(B): B 클래스의 mAP (IoU 임계값 50%)
# metrics/mAP50-95(B): B 클래스의 mAP (IoU 임계값 50-95%)
# val/box_loss: 검증 데이터의 박스 손실
# val/cls_loss: 검증 데이터의 클래스 손실
# val/dfl_loss: 검증 데이터의 DFL 손실
# lr/pg0, lr/pg1, lr/pg2: 각 파라미터 그룹의 학습률

# Load the CSV file
file_path = 'C:/kkt/2024_07_24_Colony/runs/detect/train_100_300_16/results.csv'
results_df = pd.read_csv(file_path)

# Strip whitespace from column names
results_df.columns = results_df.columns.str.strip()

# Calculate average precision, recall, and mAP over all epochs
average_precision = results_df['metrics/precision(B)'].mean()
average_recall = results_df['metrics/recall(B)'].mean()
average_map50 = results_df['metrics/mAP50(B)'].mean()
average_map50_95 = results_df['metrics/mAP50-95(B)'].mean()





# 평균 값 출력
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average mAP@50: {average_map50:.4f}")
print(f"Average mAP@50-95: {average_map50_95:.4f}")
