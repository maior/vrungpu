"""유틸리티 함수"""
import torch


def create_dummy_data(n_samples, device):
    """더미 MNIST 데이터 생성 (28x28 이미지)"""
    data = torch.randn(n_samples, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (n_samples,)).to(device)
    return data, labels


def evaluate(model, test_data, test_labels):
    """모델 평가"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_labels).sum().item()
        accuracy = 100 * correct / len(test_labels)
    return accuracy
