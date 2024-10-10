import torch
from torch import nn
import torch.nn.functional as F
from .position_embedding import SinusoidalPositionalEmbedding  # 포지셔널 임베딩 모듈
from .multihead_attention import MultiheadAttention  # 멀티헤드 어텐션 모듈
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder는 여러 개의 encoder layer로 구성.
    각 레이어는 TransformerEncoderLayer 클래스.
    
    Args:
        embed_dim (int): 임베딩 차원 수
        num_heads (int): 어텐션 헤드 수
        layers (int): 레이어의 개수
        attn_dropout (float): 어텐션 가중치에 적용될 드롭아웃 비율
        relu_dropout (float): 잔차 블록의 첫 번째 레이어에 적용될 드롭아웃 비율
        res_dropout (float): 잔차 블록 전체에 적용될 드롭아웃 비율
        embed_dropout (float): 임베딩에 적용될 드롭아웃 비율
        attn_mask (bool): 어텐션 가중치에 마스크를 적용할지 여부
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        # 드롭아웃 설정
        self.dropout = embed_dropout  # 임베딩 드롭아웃
        self.attn_dropout = attn_dropout  # 어텐션 드롭아웃
        self.embed_dim = embed_dim  # 임베딩 차원 크기
        self.embed_scale = math.sqrt(embed_dim)  # 임베딩에 스케일을 적용, 일반적으로 sqrt(embed_dim) 사용
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)  # 위치 임베딩 (사인 함수 기반)

        self.attn_mask = attn_mask  # 어텐션 마스크 여부 설정

        # 여러 층의 Transformer encoder 레이어를 구성
        self.layers = nn.ModuleList([])  # 빈 리스트로 레이어 모듈들을 저장
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)  # 각 레이어의 설정
            self.layers.append(new_layer)  # 레이어를 리스트에 추가

        # 버전 관리용 버퍼 등록
        self.register_buffer('version', torch.Tensor([2]))
        
        # 레이어 정규화 사용 여부 설정
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)  # 임베딩 차원에 대해 레이어 정규화 적용

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): 임베딩된 입력, 크기: `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): 임베딩된 key 입력 (optional), 크기: `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): 임베딩된 value 입력 (optional), 크기: `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): 마지막 encoder 레이어의 출력, 크기: `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): 패딩 요소의 위치, 크기: `(batch, src_len)`
        """
        # 입력을 임베딩 스케일로 조정
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            # 위치 임베딩 추가 (입력 시퀀스에 대해 포지셔널 임베딩 적용)
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        # 드롭아웃 적용
        x = F.dropout(x, p=self.dropout, training=self.training)

        # key와 value가 별도로 제공된 경우에도 동일하게 처리
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                # key와 value에도 포지셔널 임베딩 추가
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            # 드롭아웃 적용
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # 각 레이어를 거치며 입력을 처리
        intermediates = [x]  # 중간 결과를 저장하기 위한 리스트
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                # key와 value가 있는 경우
                x = layer(x, x_k, x_v)
            else:
                # self-attention
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            # 레이어 정규화가 설정된 경우 마지막 출력에 정규화 적용
            x = self.layer_norm(x)

        return x  # 최종 결과 반환

    def max_positions(self):
        """Encoder에서 지원하는 최대 입력 길이."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())  # 임베딩과 소스 포지션 비교 후 최소값 반환

class TransformerEncoderLayer(nn.Module):
    """
    Transformer의 Encoder 레이어 블록. 
    각 레이어는 Multi-Head Attention과 Feed Forward Network(FFN)로 구성됩니다.
    논문에서는 각 연산 후 `dropout -> add residual -> layernorm`을 수행합니다.
    *args.encoder_normalize_before*를 True로 설정하면 preprocessing으로 layernorm을 수행합니다.
    
    Args:
        embed_dim (int): 임베딩 차원 수
        num_heads (int): 어텐션 헤드 수
        attn_dropout (float): 어텐션 드롭아웃 비율
        relu_dropout (float): FFN에서 ReLU 활성화 후 드롭아웃 비율
        res_dropout (float): 잔차 연결에 대한 드롭아웃 비율
        attn_mask (bool): 어텐션에 마스크를 적용할지 여부
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim  # 임베딩 차원 크기
        self.num_heads = num_heads  # 어텐션 헤드 수

        # Multihead Attention 레이어 생성
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask  # 어텐션 마스크 여부

        # 드롭아웃 설정
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True  # 레이어 정규화를 미리 적용할지 여부

        # Feed Forward Network 부분 정의
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)  # 첫 번째 선형 변환
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)  # 두 번째 선형 변환
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])  # 레이어 정규화

    def forward(self, x, x_k=None, x_v=None):
        """
        Transformer Encoder 레이어의 순전파(forward) 함수.
        
        Args:
            x (Tensor): 입력 텐서, 크기: `(seq_len, batch, embed_dim)`
            x_k (Tensor, optional): key로 사용할 입력 텐서 (없으면 self-attention), 크기: `(seq_len, batch, embed_dim)`
            x_v (Tensor, optional): value로 사용할 입력 텐서 (없으면 self-attention), 크기: `(seq_len, batch, embed_dim)`

        Returns:
            Tensor: Transformer 레이어의 출력, 크기: `(seq_len, batch, embed_dim)`
        """
        # Residual connection을 위해 입력 x를 보존 (Residual Block에서 사용)
        residual = x
        
        # Layer normalization 적용 (전처리로서 사용)
        x = self.maybe_layer_norm(0, x, before=True)
        
        # 마스크가 설정된 경우 미래 정보를 차단하는 future mask 생성
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        
        # Self-attention: x_k, x_v가 없을 경우 self-attention을 수행
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            # Cross-attention: x_k, x_v가 있는 경우 cross-attention 수행
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        
        # Dropout 적용 (residual block에서)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        
        # Residual connection 적용 (입력과 현재 값을 더함)
        x = residual + x
        
        # Layer normalization 적용 (후처리로서 사용)
        x = self.maybe_layer_norm(0, x, after=True)

        # Feed-forward network 부분
        residual = x  # 또 다른 Residual connection을 위해 x를 보존
        
        # Layer normalization 적용 (전처리로서 사용)
        x = self.maybe_layer_norm(1, x, before=True)
        
        # Feed-forward 네트워크 첫 번째 레이어 (선형 변환 후 ReLU 활성화)
        x = F.relu(self.fc1(x))
        
        # Dropout 적용 (relu 이후)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        
        # Feed-forward 네트워크 두 번째 레이어 (선형 변환)
        x = self.fc2(x)
        
        # Dropout 적용
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        
        # Residual connection 적용 (입력과 더함)
        x = residual + x
        
        # Layer normalization 적용 (후처리로서 사용)
        x = self.maybe_layer_norm(1, x, after=True)
        
        return x  # 최종 출력 반환

    def maybe_layer_norm(self, i, x, before=False, after=False):
        """
        Layer normalization을 선택적으로 적용하는 함수.
        
        Args:
            i (int): 레이어 인덱스 (0 또는 1)
            x (Tensor): 입력 텐서
            before (bool): 레이어 정규화를 전처리로 적용할지 여부
            after (bool): 레이어 정규화를 후처리로 적용할지 여부
        
        Returns:
            Tensor: 레이어 정규화가 적용된(또는 적용되지 않은) 텐서
        """
        # before와 after가 동시에 True일 수 없도록 보장
        assert before ^ after
        
        # normalize_before 설정에 따라 before 혹은 after에 레이어 정규화를 적용
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)  # 정규화를 적용한 텐서 반환
        else:
            return x  # 정규화가 적용되지 않은 원본 텐서 반환

def fill_with_neg_inf(t):
    """
    텐서를 FP16과 호환되도록 -inf로 채우는 함수.
    
    Args:
        t (Tensor): 입력 텐서
    
    Returns:
        Tensor: -inf로 채워진 텐서
    """
    # 텐서를 float로 변환 후 -inf로 채움
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    """ 
    현재 시점 이후의 정보를 참조하지 않도록 마스크를 적용.
    
    Args:
        tensor (Tensor): Query 또는 Key로 사용할 텐서
        tensor2 (Tensor, optional): 두 번째 텐서 (Key에 해당)
    
    Returns:
        Tensor: 상삼각 마스크가 적용된 future mask
    """
    # Query 길이로 기본 크기 설정
    dim1 = dim2 = tensor.size(0)
    
    # tensor2가 있을 경우 Key 길이로 두 번째 차원 크기 설정
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    
    # 상삼각 행렬을 사용해 미래 정보를 차단 (-inf로 채운 마스크)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    '''
    1. torch.ones(dim1, dim2)
        크기가 dim1 x dim2인 1로 채워진 텐서를 생성 | dim1이 주로 Query의 길이 (seq_len), dim2는 Key의 길이 (src_len)
        Query와 Key가 같은 시퀀스인 경우 dim1과 dim2는 동일
    2. torch.triu(input, diagonal)
        상삼각 행렬(upper triangular matrix)을 반환하는 함수
        => 대각선 위의 값은 유지하고, 대각선 아래의 값은 0으로 채움
        input : 입력 텐서 | diagonal : 대각선 오프셋, diagonal=0이면 주대각선 상의 값을 유지, diagonal=1이면 주대각선 위부터 값을 유지하는 방식
        
        matrix_2 = torch.triu(torch.ones(4, 4), diagonal=2)
        matrix_3 = torch.triu(torch.ones(4, 4), diagonal=3)

        matrix_2
        (tensor([[0., 0., 1., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]),
        matrix_3
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]))
        

    3. 1 + abs(dim2 - dim1)는 주대각선 위의 첫 번째 대각선부터 값을 유지하겠다는 의미

    

    대충 이런 느낌이 됨
    tensor([[   0.,  -inf,  -inf,  -inf],
        [   0.,    0.,  -inf,  -inf],
        [   0.,    0.,    0.,  -inf],
        [   0.,    0.,    0.,    0.]])

    '''
    
    # 텐서가 GPU에 있을 경우, 마스크도 같은 장치로 이동
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    
    # 생성한 future mask 반환
    return future_mask[:dim1, :dim2]

def Linear(in_features, out_features, bias=True):
    """
    선형 변환 레이어를 생성하는 함수.
    가중치는 Xavier 초기화로, 바이어스는 0으로 초기화.
    
    Args:
        in_features (int): 입력 차원
        out_features (int): 출력 차원
        bias (bool): 바이어스 사용 여부
    
    Returns:
        Linear layer: 초기화된 선형 레이어
    """
    # 선형 레이어 생성
    m = nn.Linear(in_features, out_features, bias)
    
    # 가중치를 Xavier uniform으로 초기화
    nn.init.xavier_uniform_(m.weight)
    
    # 바이어스가 있을 경우 0으로 초기화
    if bias:
        nn.init.constant_(m.bias, 0.)
    
    return m  # 초기화된 선형 레이어 반환

def LayerNorm(embedding_dim):
    """
    Layer normalization 레이어 생성.
    
    Args:
        embedding_dim (int): 입력 차원 크기
    
    Returns:
        LayerNorm layer: 생성된 레이어 정규화 레이어
    """
    # 입력 차원 크기에 맞춘 LayerNorm 레이어 반환
    return nn.LayerNorm(embedding_dim)

# 메인 함수: Transformer Encoder의 인스턴스 생성 후, 테스트 입력에 대해 출력 형태 확인
if __name__ == '__main__':
    # 임베딩 차원 300, 4개의 어텐션 헤드, 2개의 레이어를 가진 Transformer Encoder 생성
    encoder = TransformerEncoder(300, 4, 2)
    
    # 임의의 입력 데이터 생성 (크기: 20 x 2 x 300, 즉 20 시퀀스 길이, 배치 크기 2, 임베딩 차원 300)
    x = torch.tensor(torch.rand(20, 2, 300))
    
    # Encoder를 실행하고 결과 텐서의 크기를 출력
    print(encoder(x).shape)