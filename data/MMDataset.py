from torch.utils.data import Dataset
import torch
import numpy as np

# 'MMDataset'만 사용
__all__ = ['MMDataset']


# MMDataset 클래스는 여러 종류의 데이터를 묶어서 다루기 쉽게 해주는 도구
class MMDataset(Dataset):
        
    # 클래스가 처음 만들어질 때 데이터를 넣어줌
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):
        
        # 각 데이터들을 클래스 안에 저장
        self.label_ids = label_ids  # 라벨 (이 데이터가 어떤 종류인지 알려주는 정보)
        self.text_feats = text_feats  # 텍스트 데이터
        self.cons_text_feats = cons_text_feats  # 추가로 준비된 텍스트 데이터
        self.condition_idx = condition_idx  # 조건 위치
        self.video_feats = video_feats  # 비디오 데이터
        self.audio_feats = audio_feats  # 오디오 데이터
        self.size = len(self.text_feats)  # 데이터의 총 개수를 저장함 (여기서는 텍스트 데이터의 개수를 기준으로 함)

    # 데이터셋의 크기를 알려주는 함수
    def __len__(self):
        return self.size  # 데이터의 총 개수를 돌려줌

    # 특정 위치(index)에 있는 데이터를 가져오는 함수
    def __getitem__(self, index):

        # 필요한 데이터들을 딕셔너리에 담아서 반환
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]),  # 라벨을 PyTorch 텐서로 변환해서 저장
            'text_feats': torch.tensor(self.text_feats[index]),  # 텍스트 데이터를 텐서로 변환해서 저장
            'video_feats': torch.tensor(self.video_feats['feats'][index]),  # 비디오 데이터의 특정 위치를 텐서로 저장
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),  # 오디오 데이터의 특정 위치를 텐서로 저장
            'cons_text_feats': torch.tensor(self.cons_text_feats[index]),  # 추가 텍스트 데이터도 텐서로 변환해서 저장
            'condition_idx': torch.tensor(self.condition_idx[index])  # 조건 위치 데이터를 텐서로 변환해서 저장
        } 
        return sample  # 변환된 데이터를 돌려줌
    
'''
__init__ 함수: 클래스가 처음 만들어질 때 데이터를 받아와서 저장, 다른 함수에서 이 데이터를 사용할 수 있음
__len__ 함수: 데이터셋의 크기, 즉 데이터가 몇 개인지 알려주는 함수. self.size를 돌려줌
__getitem__ 함수: 데이터셋에서 특정 위치(index)에 있는 데이터를 가져오는 함수. 데이터들을 텐서로 바꿔서 딕셔너리에 담은 다음 돌려줌
'''