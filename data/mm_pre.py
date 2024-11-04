import pickle
import numpy as np
import os


# 비디오와 오디오 데이터를 가져오는 역할
# `feats_path`라는 폴더에 데이터가 있어야 하는데, 없으면 오류를 발생
def get_v_a_data(data_args, feats_path):
    
    # 만약 `feats_path` 폴더가 존재하지 않으면, "폴더가 비어 있다"는 오류를 내보냄
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')    

    # 데이터가 있으면 `load_feats` 함수로 데이터를 불러옴
    feats = load_feats(data_args, feats_path)

    # `padding_feats` 함수로 데이터를 일정한 길이로 맞춰줌 (짧으면 패딩을 추가)
    data = padding_feats(data_args, feats)
    
    return data  # 길이가 맞춰진 데이터를 돌려줌


# 비디오 피처(feature)를 불러오는 함수
def load_feats(data_args, video_feats_path):

    # `video_feats_path` 파일에서 비디오 데이터를 읽어옴
    with open(video_feats_path, 'rb') as f:
        video_feats = pickle.load(f)

    # 학습용 데이터만 따로 모아서 리스트로 변환함
    train_feats = [np.array(video_feats[x]) for x in data_args['train_data_index']]

    # 검증용 데이터도 따로 모아서 리스트로 변환함
    dev_feats = [np.array(video_feats[x]) for x in data_args['dev_data_index']]

    # 테스트용 데이터도 따로 모아서 리스트로 변환함
    test_feats = [np.array(video_feats[x]) for x in data_args['test_data_index']]
    
    # 세 종류의 데이터를 하나의 딕셔너리로 묶어서 반환함
    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }

    return outputs  # 학습용, 검증용, 테스트용 데이터가 모두 들어있는 딕셔너리를 돌려줌


# 데이터를 일정한 길이로 맞춰주는 패딩(padding)을 해주는 함수
# 필요한 길이보다 짧으면 데이터를 0으로 채우거나 평균값으로 채움
def padding(feat, max_length, padding_mode='zero', padding_loc='end'):
    """
    padding_mode: 'zero' (0으로 채우기) 또는 'normal' (평균값으로 채우기)
    padding_loc: 'start' (앞에 채우기) 또는 'end' (뒤에 채우기)
    """
    assert padding_mode in ['zero', 'normal']  # `padding_mode`는 'zero' 또는 'normal'만 가능
    assert padding_loc in ['start', 'end']     # `padding_loc`는 'start' 또는 'end'만 가능

    length = feat.shape[0]  # 현재 데이터의 길이를 가져옴
    if length > max_length:  # 데이터 길이가 `max_length`보다 크면
        return feat[:max_length, :]  # 필요한 길이만큼 자르고 돌려줌

    # 패딩 모드가 'zero'일 때: 0으로 채움
    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    # 패딩 모드가 'normal'일 때: 데이터의 평균과 표준편차를 사용해서 랜덤한 값을 채움
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()  # 평균과 표준편차 계산
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))
    
    # 패딩 위치가 'start'일 때: 앞쪽에 패딩을 추가
    if padding_loc == 'start':
        feat = np.concatenate((pad, feat), axis=0)
    # 패딩 위치가 'end'일 때: 뒤쪽에 패딩을 추가
    else:
        feat = np.concatenate((feat, pad), axis=0)

    return feat  # 패딩을 추가한 데이터를 돌려줌


# 여러 개의 데이터를 받아서 각 데이터의 길이를 맞춰주는 함수
def padding_feats(data_args, feats):
    
    max_seq_len = data_args['max_seq_len']  # 최대 길이를 가져옴

    p_feats = {}  # 패딩이 적용된 데이터를 담을 빈 딕셔너리 생성

    # `feats` 딕셔너리에 있는 데이터셋 종류별로 반복
    for dataset_type in feats.keys():
        f = feats[dataset_type]  # 'train', 'dev', 'test' 중 하나를 선택

        tmp_list = []  # 길이 맞춘 피처를 담을 리스트
        length_list = []  # 각 피처의 원래 길이를 담을 리스트
        
        # 데이터셋 안의 각 데이터를 순서대로 처리
        for x in f:
            x_f = np.array(x)  # 데이터를 numpy 배열로 변환
            # 데이터의 차원이 3차원이면 첫 번째 차원을 없애줌 (1차원 줄이기)
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f

            length_list.append(len(x_f))  # 데이터의 원래 길이를 저장
            p_feat = padding(x_f, max_seq_len)  # 패딩을 사용해 길이를 맞춘 데이터 생성
            tmp_list.append(p_feat)  # 길이 맞춘 데이터를 리스트에 추가

        # 데이터셋 종류에 따라 길이 맞춘 피처와 원래 길이를 저장
        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats  # 각 데이터셋에 패딩을 적용한 결과를 돌려줌

'''
get_v_a_data: 비디오와 오디오 데이터를 가져오는 함수
load_feats: pickle 파일에서 학습, 검증, 테스트 데이터를 각각 읽어오는 함수
padding: 데이터 길이가 짧을 때 앞이나 뒤를 0이나 평균값으로 채워서 길이를 맞춰주는 함수
padding_feats: 여러 데이터를 padding 함수로 길이를 맞춰주는 함수
'''