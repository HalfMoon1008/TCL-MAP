import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

"""
from torch.nn import Parameter
1. 모델이 학습 가능한 파라미터로 등록
    - nn.Module 내에 정의된 경우, 모델 파라미터로 자동 등록
    - nn.Module이 선언된 Parameter는 모델의 parameters() 메더스에 의해 반환, 자동으로 최적화 대상이 됨
    - 일반적으로 torch.Tensor는 모델의 파라미터 취급 X, 학습이 필요할 때는 반드시 Parameter로 선언
2. 역전파에 포함
    - 학습 중에 변화해야하는 값, 기울기가 계산되고 최적화 과정에서 업데이트
3. 고정되지 않은 가중치
    - W & bias 정의
    - 선형 레이어의 가중치나 컨볼루션 필터 등의 값을 Parameter로 선언



"""


class MultiheadAttention(nn.Module):
    """
    Multi-headed attention을 구현한 클래스.
    자세한 설명은 논문 "Attention Is All You Need"를 참조하기

    Parameters:
    - embed_dim (int): 입력과 출력의 임베딩 차원 크기
    - num_heads (int): 어텐션을 계산할 헤드의 개수
    - attn_dropout (float): 어텐션 가중치에 적용할 드롭아웃 비율
    - bias (bool): 각 어텐션의 Q, K, V에 대해 bias를 적용할지 여부
    - add_bias_kv (bool): key와 value에 대해 bias를 추가할지 여부
    - add_zero_attn (bool): 어텐션을 계산할 때 0으로 채운 어텐션 값을 추가할지 여부
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim  # 입력 및 출력 임베딩 차원 크기
        self.num_heads = num_heads  # 어텐션 헤드의 수
        self.attn_dropout = attn_dropout  # 어텐션 가중치에 적용될 드롭아웃 비율
        self.head_dim = embed_dim // num_heads  # 각 헤드별로 사용할 임베딩 차원
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # 어텐션 스케일링을 위한 값 (임베딩 차원 수의 제곱근의 역수)
        self.scaling = self.head_dim ** -0.5

        # 입력 프로젝션 가중치 (Q, K, V를 위한 가중치가 하나로 합쳐져 있음)
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim)) # 차피 Q,K,V로 나눌거니까 3개로 시작
        self.register_parameter('in_proj_bias', None)  # 입력 프로젝션 bias를 초기화 (None으로 설정)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))  # bias를 사용할 경우 초기화
        # 최종 출력 프로젝션 (임베딩 차원 크기를 유지)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # key와 value에 추가적인 bias를 사용할지 여부
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))  # key에 대한 bias
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))  # value에 대한 bias
        else:
            self.bias_k = self.bias_v = None  # bias를 사용하지 않는 경우 None으로 설정

        self.add_zero_attn = add_zero_attn  # 어텐션에 0으로 채운 행을 추가할지 여부

        # 가중치를 초기화
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform 초기화로 가중치 초기화 (일반적으로 잘 작동하는 초기화 방식)
        nn.init.xavier_uniform_(self.in_proj_weight) # xavier_uniform_는 PyTorch에서 신경망의 가중치를 초기화하는 방법
        nn.init.xavier_uniform_(self.out_proj.weight)
        # bias가 있을 경우 0으로 초기화
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        # key와 value bias가 있을 경우 Xavier normal로 초기화
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        
        ### Q) Softmax인데 왜 Xavier?
        # attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        # Xavier 초기화는 선형 변환 레이어에서, 특히 입력과 출력 크기를 모두 고려해야 할 때 적합한 초기화 방식입니다.
        # Softmax는 어텐션 계산에서 사용되는 정규화 함수일 뿐, 네트워크의 가중치 초기화 방식에 직접적인 영향을 미치지 않습니다.
        # ReLU와 같은 활성화 함수가 있을 때 He 초기화가 종종 사용되지만, 선형 변환이 많은 Transformer 구조에서는 여전히 Xavier 초기화가 적절할 수 있습니다.

        # 실험 ㄱ?
        # ex) He (Kaiming) 초기화 _ ReLU 계열. | transformer에서는 ReLU를 쓰긴하니까!
        # Orthogonal 초기화 (nn.init.orthogonal_) 직교로도 추라이
        # LeCun 초기화 _ Sigmoid 및 Tanh 활성화 함수에 특화된 초기화 방식 | Xavier 초기화와 비슷하지만 다르니까
        # xavier_uniform_ 기준으로 실험해보쟈

    def forward(self, query, key, value, attn_mask=None):
        """
        멀티헤드 어텐션을 계산.
        
        Input shape: Time x Batch x Channel
        Self-attention은 query, key, value가 동일할 경우 수행
        어텐션 마스크(attn_mask)를 통해 특정 시간 단계(T x T) 마스크를 적용할 수 있으며,
        key의 패딩을 마스킹할 수 있음

        Parameters:
        - query (Tensor): [tgt_len x batch_size x embed_dim]
        - key (Tensor): [src_len x batch_size x embed_dim] (query와 동일할 수 있음)
        - value (Tensor): [src_len x batch_size x embed_dim] (query와 동일할 수 있음)
        - attn_mask (Tensor, optional): 어텐션 마스크 (T x T 크기)
        """
        # query, key, value가 모두 동일한 경우 (self-attention)
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        # key와 value가 동일한 경우
        kv_same = key.data_ptr() == value.data_ptr()

        # query의 크기에서 타겟 길이, 배치 크기, 임베딩 차원을 추출
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim  # 입력 임베딩 차원이 지정된 임베딩 차원과 일치하는지 확인
        assert list(query.size()) == [tgt_len, bsz, embed_dim]  # query 크기 확인
        assert key.size() == value.size()  # key와 value의 크기가 동일해야 함

        if qkv_same:
            # self-attention: query, key, value가 동일한 경우 (입력 동일)
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention: key와 value가 동일할 때
            q = self.in_proj_q(query)
            if key is None:
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            # 일반적인 경우: query, key, value가 각각 다를 때
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        # query에 스케일링 적용 (scaled dot-product attention)
        q = q * self.scaling

        # key와 value에 bias가 있을 경우 추가
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])  # key에 bias 추가
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])  # value에 bias 추가
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # query, key, value를 여러 헤드로 나눠서 크기를 조정
        # 연속적으로 만들어서 차원 변환 후 index 0과 1을 transpose
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # contiguous() : 메모리 상에서 연속된 텐서를 보장하는 함수
        # 메모리 상에서 비연속적인 상태가 될 수 있음, 이는 view()와 같은 함수가 메모리 상에서 연속된 상태를 요구하기 때문에 필요
        # 비연속적인 메모리를 가진 Tensor를 연속된 메모리로 변환
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 어텐션 계산에서 사용하는 key의 길이 (src_len)
        src_len = k.size(1)

        # 필요 시 0으로 채운 어텐션 값 추가
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # 어텐션 가중치 계산 (Q * K^T)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # bmm() : batch matrix multiplication
        # 3차원 텐서에서 배치별로 2D 행렬 곱셈을 수행할 때 사용
        # 입력으로 3D 텐서 두 개를 받아, 각 배치마다 2D 행렬 곱셈을 수행

        # [batch_size, n, m]과 [batch_size, m, p]를 가지고 [batch_size, n, p]으로 만듬
        # ex)
        # batch1 = torch.randn(10, 3, 4)  # shape: [10, 3, 4]
        # batch2 = torch.randn(10, 4, 5)  # shape: [10, 4, 5]
        # result = torch.bmm(batch1, batch2)  # shape: [10, 3, 5]

        # 이미 q가 [batch_size * num_heads, tgt_len, head_dim] 크기를 가진 3차원 텐서
        # k.transpose(1, 2): k는 [batch_size * num_heads, src_len, head_dim] 크기를 가진 텐서여서 차원을 transpose
        # [batch_size * num_heads, tgt_len, src_len] 크기의 어텐션 가중치를 생성

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # 어텐션 마스크가 있을 경우 가중치에 마스크를 추가
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
        '''
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        에서 마스크를 사용
        '''

        # softmax로 어텐션 가중치를 계산
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        # 드롭아웃 적용
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # 어텐션 가중치를 이용해 value 값들을 가중 합산 (Q * K^T * V)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # 다시 배치 크기와 임베딩 차원으로 크기를 복원
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)  # 최종 프로젝션을 통해 출력 생성

        # 어텐션 가중치를 각 헤드별로 평균 내서 반환
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights  # 어텐션 결과와 어텐션 가중치를 반환

    # Query, Key, Value를 위한 입력 프로젝션 함수들
    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)  # Q, K, V로 나눔

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)  # K, V로 나눔

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    # 입력 프로젝션 함수: Query, Key, Value에 대해 입력을 프로젝션
    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)  # 선형 변환 적용


# 수학적 정사영(Orthogonal Projection)과 여기에서의 in_proj은 다르다.
# 여기서는 선형변환을 위한 행렬 곱의 의미