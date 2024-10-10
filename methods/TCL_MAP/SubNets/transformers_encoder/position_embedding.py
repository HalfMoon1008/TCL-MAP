import torch
import torch.nn as nn

import math

def make_positions(tensor, padding_idx, left_pad):

    # max_pos는 padding+1부터 시작 _ tensor의 두 번째 차원의 크기만큼의 번호가 필요
    # padding_idx는 다음 위치 번호부터 최대 텐서 길이까지의 위치 번호가 할당
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device() # CPU OR GPU
    buf_name = f'rang_buf_{device}' # 장치별로 이름 부여

    # hasattr(object, name)
    #   객체 object에 이름이 name인 속성이 있는지 확인하는 함수
    #   객체에 특정 속성이 있는지 여부 확인 후 True or False 반환
    # setattr(object, name, value)
    #   object에 이름이 name인 속성을 추가하거나, 기존 속성을 value 값으로 설정하는 함수
    #   객체에 새로운 속성을 추가하거나 기존 속성의 값 변경
    # getattr(object, name[, default])
    #   객체 object에서 이름이 name인 속성의 값을 반환하는 함수
    #   객체에 특정 속성 값을 가져오거나, 속성이 없을 경우 기본 값을 반환

    # `make_positions` 함수에 해당 이름의 속성이 존재하지 않으면, 새로운 텐서를 생성
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new()) # 그냥 비어있는 tensor 생성?
    # `range_buf_*` 속성을 현재 텐서의 데이터 타입으로 설정
    setattr(make_positions, buf_name, getattr(make_positions,buf_name).type(tensor))

    # `range_buf_*` 텐서에 저장된 값의 수가 필요한 최대 위치 번호보다 작으면, torch.arange로 범위를 생성해 업데이트
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))

    # 텐서에서 패딩이 아닌 위치를 표시하기 위한 마스크를 생성
    # 패딩인 곳은 False, 패딩이 아닌 곳은 True
    mask = tensor.ne(padding_idx)
    # `range_buf_*`에서 현재 텐서의 두 번째 차원의 크기만큼의 범위를 확장하여 `positions` 텐서를 만듬
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)

    # 만약 패딩이 왼쪽에 있으면 (left_pad=True), 패딩의 크기를 반영하여 위치 번호를 조정
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    # 만든 텐서 복사해서 새로운 텐서 만듬
    new_tensor = tensor.clone()

    # `mask`에서 True인 위치(패딩이 아닌 위치)에 대해, 위치 번호를 해당 위치에 덮어씀
    return new_tensor.masked_scatter_(mask,positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    이 모듈은 임의의 길이의 위치 정보에 대한 사인 함수 기반의 위치 임베딩을 생성
    패딩 심볼(즉, `padding_idx`)은 무시되며, 패딩이 왼쪽에 추가되었는지 (left_pad=True)
    오른쪽에 추가되었는지 (left_pad=False) 여부를 지정해야 함.
    
    Parameters:
    - embedding_dim (int): 각 위치에 대해 생성될 임베딩 벡터의 차원 수.
    - padding_idx (int, optional): 패딩을 나타내는 인덱스. 해당 인덱스 위치는 0으로 설정.
    - left_pad (bool, optional): 패딩이 왼쪽에 있는지 여부
    - init_size (int, optional): 초기 크기를 설정. 초기 상태에서 생성될 최대 시퀀스 길이를 정의.
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict() # 각 디바이스에 대한 임베딩을 저장하기 위한 딕셔너리 (nn.DataParallel 사용 시 필요)
        self.register_buffer('_float_tensor', torch.FloatTensor(1)) # 디바이스 타입 관리를 위한 FloatTensor 등록

    @staticmethod #클래스의 메서드를 정적 메서드(static method)로 정의
    # 정적 메서드 : 클래스의 인스턴스나 클래스 자체와 무관하게 동작하는 메서드로 독립적인 작업 or 유틸리티 함수를 수행할 때 적합
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """
        사인 함수 기반 위치 임베딩을 생성
        이 구현은 'Attention Is All You Need' 논문의 3.5절과 약간 다름

        ###Paper###
        PE_(pos,2_i) = sin [ pos / 10000^[{2_i/d_model}]
        PE_(pos,2_i +1) = cos [ pos / 10000^[{2_i/d_model}]

        ###Code###
        논문의 '1 / 10000^[{2_i/d_model}' 대신 'torch.arange(half_dim)'을 통해 i차원의 주기를 조정
        논문에서는 pos마다 주기가 달라지도록 하여 각 위치의 임베딩을 구했지만,
        get_embedding에서는 'emb = math.log(10000) / (half_dim - 1)'로 고정된 로그 변환을 사용하여
        i를 변등하게 변경하고, 그 후 위치 인덱스를 곱해 임베딩을 계산
        즉, 차원 별 주기 변화는 있지만, 논문처럼 미세한 변화보다는 상대적으로 간단하게 주기가 변경되는 구조
        => 경우에 따라서는 좀 더 효율적일 수 있음(계산 복잡도 감소하지만, 실제 위치 정보는 유지)

        Parameters:
        - num_embeddings (int): 생성할 임베딩의 개수 (최대 위치 수)
        - embedding_dim (int): 임베딩 벡터의 차원 크기
        - padding_idx (int, optional): 패딩 인덱스가 있는 경우 해당 위치를 0으로 설정

        Returns:
        - Tensor: 각 위치에 대한 사인 및 코사인 값으로 구성된 임베딩 텐서
        """
        half_dim = embedding_dim // 2 # 임베딩 차원의 절반, 사인과 코사인 파트로 나누기 위함
        emb = math.log(10000) / (half_dim - 1) # 각 차원에서 주기의 변화율을 계산
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb) # 주기의 변화에 따른 지수적 감소 생성
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0) # 위치마다 다른 주기를 가진 임베딩 계산
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1).view(num_embeddings, -1) # 사인과 코사인 값을 합쳐 최종 임베딩 생성
        if embedding_dim % 2 == 1:
            # 임베딩 차원이 홀수일 경우, 마지막에 zero padding을 추가해 차원을 맞춤
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

        # 패딩 인덱스가 있으면 해당 위치의 임베딩은 모두 0으로 설정
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb # 최종적으로 사인, 코사인으로 구성된 임베딩 텐서 반환
    
        """
        Paper 그대로 하려면 아래와 같음

        Parameters:
        - num_embeddings (int): 최대 위치 (임베딩할 시퀀스 길이)
        - embedding_dim (int): 임베딩 차원

        Returns:
        - Tensor: [num_embeddings, embedding_dim] 크기의 위치 임베딩 텐서

        def get_embedding(num_embeddings, embedding_dim):
            # 임베딩의 각 위치(pos)와 차원(i)에 대해 PE(pos, 2i)와 PE(pos, 2i+1)를 계산
            position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)  # 위치 인덱스 [num_embeddings, 1]
            div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * 
                                -(math.log(10000.0) / embedding_dim))  # 주기 변화 계산 [embedding_dim // 2]

            # 짝수 인덱스 차원(2i)에는 사인 함수를 적용
            pe = torch.zeros(num_embeddings, embedding_dim)
            pe[:, 0::2] = torch.sin(position * div_term)  # [num_embeddings, embedding_dim // 2]

            # 홀수 인덱스 차원(2i+1)에는 코사인 함수를 적용
            pe[:, 1::2] = torch.cos(position * div_term)  # [num_embeddings, embedding_dim // 2]

            return pe
        
        maybe?
        """
    
    def forward(self, input):
        """
        주어진 입력에 대해 사인 함수 기반의 위치 임베딩을 생성.
        입력 텐서의 크기는 [batch_size x seq_len] 이어야 함.

        Parameters:
        - input (Tensor): [batch_size x seq_len] 크기의 입력 시퀀스

        Returns:
        - Tensor: [batch_size x seq_len x embedding_dim] 크기의 위치 임베딩
        """
        bsz, seq_len = input.size()  # 배치 크기와 시퀀스 길이 추출
        max_pos = self.padding_idx + 1 + seq_len  # 필요한 최대 위치 인덱스 계산
        device = input.get_device()  # 입력이 위치한 디바이스(CPU/GPU) 확인
        
        # 현재 디바이스에 대해 임베딩이 아직 없거나, 필요한 시퀀스 길이가 기존 임베딩보다 크면 임베딩을 재생성/확장
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,  # 현재 시퀀스 길이에 맞는 최대 위치 수
                self.embedding_dim,  # 임베딩 차원
                self.padding_idx  # 패딩 인덱스
            )
        
        # 생성된 임베딩을 입력의 디바이스로 전송하고, _float_tensor와 같은 데이터 타입으로 변환
        self.weights[device] = self.weights[device].type_as(self._float_tensor).to(input.device)
        
        # 입력 시퀀스에서 각 토큰의 위치를 계산 (make_positions 함수 이용)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        
        # 계산된 위치를 기반으로 임베딩에서 해당 위치의 임베딩 값을 선택하고, 결과를 리턴
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape(bsz, seq_len, -1).detach()

    def max_positions(self):
        """
        지원되는 최대 위치의 개수를 반환합니다.
        """
        return int(1e5)  # 짱 큰 숫자로 설정하여 사실상 제한이 없도록 함