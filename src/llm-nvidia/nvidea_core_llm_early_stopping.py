# ============================================================
# File: nvidea_core_llm_early_stopping.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   PyTorch 학습 루프에 Early Stopping 기능을 추가한 예제.
#   목표 정확도와 patience 값을 설정하여 불필요한 epoch 반복을 줄이고,
#   GPU 자원 절약 및 과적합 방지를 실무적으로 구현.
#
# Usage: ★★★ 
#   - 단독 실행 가능
#   - 조합 가능 (fashion_mnist.py 같은 학습 스크립트와 함께 사용 권장)
# ============================================================

class EarlyStopping:
    def __init__(self, target_accuracy=0.85, patience=2):
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0

    def check(self, val_acc):
        if val_acc >= self.target_accuracy:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at accuracy {val_acc:.3f}")
                return True
        else:
            self.counter = 0
        return False
####   학습 루프에서 적용 ###### 
early_stopper = EarlyStopping(target_accuracy=0.82, patience=2)

for epoch in range(epochs):
    train_one_epoch(...)
    val_acc = validate(...)
    if early_stopper.check(val_acc):
        break
