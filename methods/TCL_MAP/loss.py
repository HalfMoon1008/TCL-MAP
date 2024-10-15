from torch import nn
import torch
import torch.nn.functional as F

class SupConLoss(nn.Module):
    '''
    SimCLR에 있는 unsupervised contrastive loss 을 지원
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    '''
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss,self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    '''
    loss 계산
    label과 mask가 둘 다 없으면 SimCLR 비지도 손실 loss로
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...]
        labels: ground truth of shape [bsz]
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
              has the same class as sample i. Can be asymmetric
    Returns:
        A loss scalar.
    '''
    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3 :
            raise ValueError ('`features` needs to be [bsz, n_views, ...],'
                                'at least 3 dimensions are required')
        
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = F.normalize(features, dim = 2)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask in None :
            mask = torch.eye(batch_size, dtype=torch.foloat32).to(device)
        elif labels is not None:
            labels = labels.contihuous().view(-1,1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
        else:
            mask = mask.float().to(device)

        '''
        contrast_count : features.shape[1]로 설정
                        각 샘플에 대해 몇 개의 대조(contrast) 뷰가 존재하는지를 나타냄 ex:augmentations
                        대조 샘플의 수를 나타내는 변수
                        features의 크기에서 두 번째 차원(n_views)의 값이 됨
                        즉, 각 샘플에 대해 몇 개의 대조 뷰가 존재하는지를 의미
                        ex) 만약 각 샘플에 대해 2개의 뷰(대조 샘플)가 있다면, contrast_count는 2가 됨

        contrast_feature : torch.cat(torch.unbind(features, dim=1), dim=0)으로 정의되며, 이는 각 샘플의 여러 뷰를 하나의 텐서로 결합한 것
                            대조 학습에서 각 샘플에 대해 여러 대조 뷰가 존재하는데, 
                            이를 모두 하나의 텐서로 결합하여 하나의 텐서로 만든 대조 샘플 집합을 contrast_feature라고 부름
                            즉, 각 샘플에 대한 대조 뷰들을 모두 하나로 결합한 것
                            ex) 만약 features의 크기가 [32, 2, 128] (batch_size 32, 대조 뷰 2개, feature_dim 128)인 경우, 
                                contrast_feature의 크기는 [64, 128]로 변환

        anchor_count : 앵커(기준) 샘플의 수를 나타내는 변수로, self.contrast_mode에 따라 결정
                        == 'one'인 경우, 앵커 샘플 하나만 사용하므로 anchor_count = 1
                        == 'all'인 경우, 모든 뷰가 앵커로 사용되므로 anchor_count = contrast_count가 됨
                        앵커 == 기준이 되는 샘플
                        학습 중에 대조군과 비교하여, 같은 클래스에 속하는 샘플들과의 유사도를 최대화하고 다른 클래스에 속하는 샘플들과의 유사도를 최소화
                        anchor_count는 얼마나 많은 앵커를 사용할 것인지를 결정하는 변수
                        특정한 경우 하나의 앵커만 선택할 수도 있고, 대조군에서 모든 뷰를 앵커로 사용할 수도 있음
                        'one'이면 앵커 샘플 하나만 사용하여 학습을 진행 or 'all'이면 모든 샘플과 뷰들이 앵커로 진행

        anchor_feature : 학습 중 앵커로 사용할 특징 벡터들을 저장한 변수
                        'one'인 경우 features[:, 0]을 사용하여 첫 번째 뷰만을 앵커로 사용
                        'all'인 경우 모든 대조 샘플을 앵커로 사용하기 위해 contrast_feature와 동일하게 설정
                        대조 학습에서는 특정 샘플이 앵커(anchor)가 되고, 
                        다른 샘플들과 비교되어 유사도를 계산함. 이때, 앵커로 사용할 샘플들의 특징 벡터들을 anchor_feature로 저장

        anchor_dot_contrast : 앵커 샘플과 대조 샘플 간의 유사도를 나타내는 값
                            anchor_feature가 [batch_size * anchor_count, feature_dim]이고 
                            contrast_feature가 [batch_size * contrast_count, feature_dim]이라면, 
                            내적 결과는 [batch_size * anchor_count, batch_size * contrast_count] 크기의 행렬이 됨
                            즉, 각 값은 앵커 샘플과 대조 샘플 간의 유사도를 나타냄

        요약
            contrast_count: 각 샘플에 대한 대조 샘플(뷰)의 수.
            contrast_feature: 각 샘플의 모든 대조 샘플을 하나로 결합한 텐서.
            anchor_count: 학습에 사용할 앵커의 수.
            anchor_feature: 학습에 사용할 앵커 샘플들의 특징 벡터.
            anchor_dot_contrast: 앵커 샘플과 대조 샘플 간의 유사도를 계산한 결과.
        '''

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1),dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:,0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # logits 계산
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        ### matmul 기반 행렬 곱
        # anchor_feature과 contrast_feature는 현재 (batch_size, feature_size)를 가짐
        # T해줘서 행렬 곱

        # self.temperature는 온도 매개변수 | 대조학습에서 중요한 역할!!
        # 유사도 계산의 결과르 조정하는 역할 | def __init__(self, temperature=0.07, contrast_mode='all'):
        # 내적을 정규화(normalization)하여 모델이 너무 크거나 작은 유사도 값에 과하게 의존하지 않도록 함
        # 보통 작은 값 가짐(0.05 ~ 0.1)
        # 값이 작을수록 분포가 날카로워짐, 큰 값일수록 유사도 분포가 평탄해짐
        # 즉, 값이 클수록 예측된 유사도 값들이 덜 민감하게 반응, 작을수록 더 민감하게 반응
        # 코드에서는 0.07이라는 값을 통해 유사도 차이에 민감하게 반응하도록 설계
        # Contrastive Learning의 특성으로 인해 같은 클래스 벡터는 가깝게, 다른 클래스 벡터는 멀리 떨어뜨리는 효과
    
        # Softmax
        # sigma(z_i) = {exp(z_i)}/sum_i{exp(z_j)}
        # 이때, z_i는 주어진 입력 값
        # temperature F
        # sigma(z_i) = {exp(z_i/T)}/sum_i{exp(z_j/T)}
        # 이때 T가 온도 매개변수

        # 최적의 temperature 값은 보통 실험적 조정을 통해 이뤄짐

        # 수치 안정성
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 마스크 관련
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), #(batch_size * anchor_count, batch_size * contrast_count)인 Tensor가 1로 가득
            1,
            torch.arange(batch_size * anchor_count).view(-1,1).to(device),
            0
        )
        
        # mask.repeat(anchor_count, contrast_count): 마스크를 확장하여 모든 앵커와 대조 샘플의 관계를 나타냄
        # torch.scatter(...): logits_mask를 생성하여 각 샘플이 자기 자신과의 관계는 제외하도록 만듬
        # mask * logits_mask: 최종적으로 자기 자신과의 유사도는 제거하고 다른 샘플들과의 관계만 남는 마스크를 생성

        # torch.arange(batch_size * anchor_count) : 
        # 0부터 batch_size * anchor_count - 1까지의 값을 가진 텐서를 생성

        # view(-1, 1) : (batch_size * anchor_count, 1)

        # torch.scatter(...) :
        # logits_mask의 두 번째 차원(dim=1)에서 torch.arange(batch_size * anchor_count)로 인덱싱된 위치에 0을 할당
        
        # if, batch_size * anchor_count가 4,
        # logits_mask
        # [[0, 1, 1, 1],
        #  [1, 0, 1, 1],
        #  [1, 1, 0, 1],
        #  [1, 1, 1, 0]]
        # 자신과의 유사도는 계산 X, 다른 샘플들과의 유사도만 고려하기 위한 용도

        # if, mask
        # [[1, 1, 0, 0],
        #  [1, 1, 0, 0],
        #  [0, 0, 1, 1],
        #  [0, 0, 1, 1]]
        
        # mask = mask * logits_mask
        # [[0, 1, 0, 0],
        #  [1, 0, 0, 0],
        #  [0, 0, 0, 1],
        #  [0, 0, 1, 0]]
        # => 자신과의 유사도를 제거하고, 다른 샘플들과의 관계만 유지하는 마스크 생성


        mask = mask * logits_mask

        # log_prob 계산
        exp_logits = torch.exp(logits) * logits_mask # 각 샘플 간 유사도 값을 지수 함수로 변환한 후, 자신과의 유사도를 제거한 값을 포함한 텐서 생성
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # 소프트맥스 확률의 로그를 계산

        # 같은 클래스에 속하는 샘플들 간의 평균 로그 확률을 계산
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mask * log_prob :
        # mask는 각 샘플 간 같은 클래스인지 여부를 나타내는 행렬
        # 요걸 log_prob와 곱함으로써 같은 클래스에 속하는 샘플들만 선택하여 로그 확률을 계산

        # mask.sum(1) :
        # 각 샘플에 대해 같은 클래스인 샘플의 개수를 계산

        # (mask * log_prob).sum(1) / mask.sum(1) :
        # 같은 클래스 샘플의 로그 확률 합계를 같은 클래스 샘플의 개수로 나누어 평균 로그 확률을 구함
        # => 같은 클래스에 속하는 샘플들(positive pairs) 간의 평균 로그 확률을 계산하는 과정

        # loss
        loss = - mean_log_prob_pos # 긍정적 쌍의 로그 확률을 최대화하는 것이 목표이므로, 이를 음수로 바꾸어 최소화할 손실로 사용
        loss = loss.view(anchor_count, batch_size).mean()
        # view로 (anchor_count, batch_size)로 변환
        # => 각 앵커 샘플에 대해 배치 내에서 계산된 손실을 구조화하는 작업.mean()

        return loss