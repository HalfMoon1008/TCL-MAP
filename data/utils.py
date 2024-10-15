import pickle
import numpy as np
import os

from torch.utils.data import DataLoader

# 데이터 로더 생성 함수
# args: 학습에 필요한 하이퍼파라미터들을 포함하는 인자 (배치 크기, 워커 수 등)
# data: 'train', 'dev', 'test' 세 가지 데이터셋이 포함된 딕셔너리
def get_dataloader(args, data):

    # 학습 데이터 로더 생성
    # - 데이터 섞음(shuffle=True)
    # - 지정된 배치 크기 사용 (args.train_batch_size)
    # - 여러 워커를 사용해 데이터 로딩 (args.num_workers)
    # - 핀 메모리 활성화 (pin_memory=True) -> GPU 메모리로 데이터를 빠르게 전송
    train_dataloader = DataLoader(
        data['train'], 
        shuffle=True, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 검증(dev) 데이터 로더 생성
    # - 데이터는 섞지 않음 (shuffle=False)
    # - 지정된 배치 크기 사용 (args.eval_batch_size)
    # - 여러 워커 사용 (args.num_workers)
    # - 핀 메모리 활성화 (pin_memory=True)
    dev_dataloader = DataLoader(
        data['dev'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 테스트 데이터 로더 생성
    # - 데이터는 섞지 않음 (shuffle=False)
    # - 지정된 배치 크기 사용 (args.eval_batch_size)
    # - 여러 워커 사용 (args.num_workers)
    # - 핀 메모리 활성화 (pin_memory=True)
    test_dataloader = DataLoader(
        data['test'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # 생성한 데이터 로더들을 딕셔너리 형태로 반환
    return {
        'train': train_dataloader,
        'dev': dev_dataloader,
        'test': test_dataloader
    }
