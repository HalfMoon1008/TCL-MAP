import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# CTCModule, AlignSubNet, SimModule 클래스를 외부에서 import 할 수 있도록 정의
__all__ = ['CTCModule', 'AlignSubNet', 'SimModule']

# CTCModule: Connectionist Temporal Classification(CTC) 기반의 정렬 모듈
class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len, args):
        super(CTCModule, self).__init__()
        # LSTM을 사용해 입력 시퀀스를 처리하고, 출력 시퀀스 + 1(blank label)을 예측
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True)
        self.out_seq_len = out_seq_len

    def forward(self, x):
        # 입력 시퀀스를 LSTM에 통과시켜 예측된 출력 시퀀스 + blank 레이블을 얻음
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        # 소프트맥스를 적용하여 확률로 변환, shape: (batch_size, in_seq_len, out_seq_len+1)
        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank)
        
        # blank 레이블(첫 번째 레이블)을 제외하고, 실제 출력 시퀀스 확률만 사용, shape: (batch_size, in_seq_len, out_seq_len)
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:]
        
        # (batch_size, out_seq_len, in_seq_len)로 차원 변경
        prob_pred_output_position = prob_pred_output_position.transpose(1,2)
        
        # 배치별 행렬 곱 연산을 통해 입력 시퀀스를 pseudo-aligned output으로 변환
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # shape: (batch_size, out_seq_len, in_dim)
        
        return pseudo_aligned_out
    
# SimModule: 입력 시퀀스 간의 정렬을 수행하는 모듈
class SimModule(nn.Module):
    def __init__(self, in_dim_x, in_dim_y, shared_dim, out_seq_len, args):
        super(SimModule, self).__init__()
        # CTC 기반 정렬 모듈을 사용하여 A에서 B로 정렬을 예측
        self.ctc = CTCModule(in_dim_x, out_seq_len, args)
        self.eps = args.eps  # 정규화 과정에서 작은 값을 사용해 나눗셈으로 인한 문제가 발생하지 않도록 함
        
        # 학습 가능한 파라미터: 로짓 스케일을 위한 로그 값 (초기화 값은 log(1/0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 입력 특징을 shared_dim 차원으로 변환하는 선형 레이어
        self.proj_x = nn.Linear(in_features=in_dim_x, out_features=shared_dim)
        self.proj_y = nn.Linear(in_features=in_dim_y, out_features=shared_dim)

        # FC 레이어: 입력 특징 길이를 줄이는 역할
        self.fc1 = nn.Linear(in_features=out_seq_len, out_features=round(out_seq_len / 2))
        self.fc2 = nn.Linear(in_features=round(out_seq_len / 2), out_features=out_seq_len)
        
        # 활성화 함수
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x를 CTC 모듈을 통해 정렬된 출력으로 변환
        pseudo_aligned_out = self.ctc(x)

        # x 특징을 shared_dim 차원으로 변환하고 정규화
        x_common = self.proj_x(pseudo_aligned_out)
        x_n = x_common.norm(dim=-1, keepdim=True)
        x_norm = x_common / torch.max(x_n, self.eps * torch.ones_like(x_n))
        
        # y 특징을 shared_dim 차원으로 변환하고 정규화
        y_common = self.proj_y(y)
        y_n = y_common.norm(dim=-1, keepdim=True)
        y_norm = y_common / torch.max(y_n, self.eps * torch.ones_like(y_n))
            
        # 코사인 유사도를 로짓으로 변환
        logit_scale = self.logit_scale.exp()  # 로짓 스케일링
        similarity_matrix = logit_scale * torch.bmm(y_norm, x_norm.permute(0, 2, 1))  # 코사인 유사도를 계산
        
        # 소프트맥스 및 FC 레이어를 통해 유사도를 정제
        logits = similarity_matrix.softmax(dim=-1)
        logits = self.fc1(logits)
        logits = self.relu(logits)
        logits = self.fc2(logits)
        logits = self.sigmoid(logits)
        
        # 정렬된 출력 생성 (logits와 pseudo_aligned_out의 배치 행렬 곱)
        aligned_out = torch.bmm(logits, pseudo_aligned_out)

        return aligned_out

# AlignSubNet: 멀티모달 정렬을 위한 서브 네트워크
class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d', 'sim']  # 지원하는 정렬 모드 확인

        # 입력 차원 설정 (텍스트, 비디오, 오디오)
        in_dim_t, in_dim_v, in_dim_a = args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim
        
        # 시퀀스 길이 설정 (텍스트, 비디오, 오디오)
        seq_len_t, seq_len_v, seq_len_a = args.max_cons_seq_length, args.video_seq_len, args.audio_seq_len
        
        self.dst_len = seq_len_t  # 목표 시퀀스 길이는 텍스트 시퀀스 길이로 맞춤
        self.dst_dim = in_dim_t  # 목표 차원은 텍스트 차원으로 맞춤
        self.mode = mode  # 정렬 모드 (avg_pool, ctc, conv1d, sim 중 선택)

        # 정렬 방식을 함수로 매핑
        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool,  # 평균 풀링 방식
            'ctc': self.__ctc,  # CTC 방식
            'conv1d': self.__conv1d,  # 1D 컨볼루션 방식
            'sim': self.__sim,  # 유사도 기반 정렬 방식
        }

        # 모드에 따른 레이어 설정
        if mode == 'conv1d':
            self.conv1d_t = nn.Conv1d(seq_len_t, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_v = nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_a = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ctc':
            self.ctc_t = CTCModule(in_dim_t, self.dst_len, args)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len, args)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len, args)
        elif mode == 'sim':
            self.shared_dim = args.shared_dim  # 텍스트, 비디오, 오디오 간 공유할 차원
            self.sim_t = SimModule(in_dim_t, self.dst_dim,  self.shared_dim, self.dst_len, args)
            self.sim_v = SimModule(in_dim_v, self.dst_dim, self.shared_dim, self.dst_len, args)
            self.sim_a = SimModule(in_dim_a, self.dst_dim, self.shared_dim, self.dst_len, args)

    def get_seq_len(self):
        # 정렬된 시퀀스 길이를 반환
        return self.dst_len
    
    # CTC 방식으로 텍스트, 비디오, 오디오 정렬
    def __ctc(self, text_x, video_x, audio_x):
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x


    # 평균 풀링 방식으로 텍스트, 비디오, 오디오 정렬
    def __avg_pool(self, text_x, video_x, audio_x):
        def align(x):
            raw_seq_len = x.size(1)  # 원래 시퀀스 길이
            if raw_seq_len == self.dst_len:  # 목표 시퀀스 길이와 같으면 그대로 반환
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        video_x = align(video_x)
        audio_x = align(audio_x)
        return text_x, video_x, audio_x
    
    # 1D 컨볼루션 방식으로 텍스트, 비디오, 오디오 정렬
    def __conv1d(self, text_x, video_x, audio_x):
        text_x = self.conv1d_t(text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.conv1d_v(video_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.conv1d_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x
    
    # 유사도 기반 정렬 방식으로 텍스트, 비디오, 오디오 정렬
    def __sim(self, text_x, video_x, audio_x):
        text_x = self.sim_t(text_x, text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.sim_v(video_x, text_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.sim_a(audio_x, text_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x

    # Forward 함수: 텍스트, 비디오, 오디오 시퀀스를 정렬
    def forward(self, text_x, video_x, audio_x):
        # 이미 시퀀스 길이가 모두 같으면 그대로 반환
        if text_x.size(1) == video_x.size(1) and text_x.size(1) == audio_x.size(1):
            return text_x, video_x, audio_x
        # 선택된 정렬 방식을 통해 시퀀스를 정렬
        return self.ALIGN_WAY[self.mode](text_x, video_x, audio_x)