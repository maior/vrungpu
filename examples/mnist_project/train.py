"""MNIST 학습 스크립트"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from utils import create_dummy_data, evaluate

# 설정
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001

def main():
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 모델 생성
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 더미 데이터 생성 (실제로는 데이터셋 로드)
    train_data, train_labels = create_dummy_data(1000, device)
    test_data, test_labels = create_dummy_data(200, device)

    print(f"\n=== Training Started ===")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}\n")

    # 학습
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = len(train_data) // BATCH_SIZE

        for i in range(0, len(train_data), BATCH_SIZE):
            batch_x = train_data[i:i+BATCH_SIZE]
            batch_y = train_labels[i:i+BATCH_SIZE]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        accuracy = evaluate(model, test_data, test_labels)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print(f"\n=== Training Completed ===")
    print(f"Final Accuracy: {accuracy:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")


if __name__ == "__main__":
    main()
