import os
import logging
import csv
import copy
import numpy as np

# 필요한 파일에서 클래스나 함수를 가져옴
from .MMDataset import MMDataset  # 여러 데이터(텍스트, 비디오, 오디오 등)를 한꺼번에 다루기 쉽게 해주는 클래스
from .text_pre import get_t_data  # 텍스트 데이터를 가져오는 함수
from .mm_pre import get_v_a_data  # 비디오와 오디오 데이터를 가져오는 함수
from .__init__ import benchmarks   # 벤치마크 데이터를 가져옴 (데이터셋에 관한 정보가 담겨 있음)

# 여기서는 DataManager 클래스만 씀
__all__ = ['DataManager']


# DataManager 클래스는 데이터셋을 불러오고 필요한 준비를 해주는 역할
class DataManager:
    
    def __init__(self, args):
        
        # 로깅 설정: 로거(logger)라는 도구를 사용해 로그를 기록 (프로그램 진행 상황이나 정보를 알려줌)
        self.logger = logging.getLogger(args.logger_name)
        
        # 데이터를 불러오는 함수 호출해서 `mm_data`에 저장
        self.mm_data = get_data(args, self.logger) 


# 데이터 불러오는 함수
def get_data(args, logger):
    
    # 데이터 경로를 만들어 줌 (예: "./data/MIntRec")
    #base_path = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join('/home/gitiresearch/TCL-MAP/'+ args.data_path, args.dataset))
    #print(f"Base path: {base_path}")
    print(f"Data path: {data_path}")


    # 불러올 데이터셋에 대한 정보를 가져옴 (벤치마크에서 데이터셋 정보 가져오기)
    bm = benchmarks[args.dataset]
    

    # 의도(intent) 라벨 목록을 복사해와서 사용
    label_list = copy.deepcopy(bm["intent_labels"])
    logger.info('Lists of intent labels are: %s', str(label_list))  # 의도 라벨 목록을 로그에 기록함
      
    # 의도 라벨 개수와 데이터의 각 차원 설정
    args.num_labels = len(label_list)  # 라벨 개수
    args.text_feat_dim = bm['feat_dims']['text']  # 텍스트 피처 차원
    args.video_feat_dim = bm['feat_dims']['video']  # 비디오 피처 차원
    args.audio_feat_dim = bm['feat_dims']['audio']  # 오디오 피처 차원
    args.label_len = bm['label_len']  # 라벨 길이
    logger.info('In-distribution data preparation...')  # 데이터 준비 중이라는 로그 남기기
    
    # 학습, 검증, 테스트 데이터 인덱스와 라벨을 가져옴
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'train.tsv'), args.data_mode)
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'), args.data_mode)
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'), args.data_mode)
    args.num_train_examples = len(train_data_index)  # 학습용 데이터 개수 설정
    
    # 데이터에 필요한 여러 설정을 모아둠
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
        'bm': bm,
    }
    
    # 텍스트 데이터 길이를 설정하고 가져옴
    data_args['max_seq_len'] = args.text_seq_len = bm['max_seq_lengths']['text']
    text_data, cons_text_feats, condition_idx = get_t_data(args, data_args)
    
    
    # 비디오 피처 데이터 경로와 설정을 모아둔 후, 비디오 데이터를 가져옴
    video_feats_path = os.path.join(data_path, 'video_feats.pkl')
    video_feats_data_args = {
        'data_path': video_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    video_feats_data_args['max_seq_len'] = args.video_seq_len = bm['max_seq_lengths']['video_feats']
    video_feats_data = get_v_a_data(video_feats_data_args, video_feats_path)

    # 오디오 피처 데이터 경로와 설정을 모아둔 후, 오디오 데이터를 가져옴
    audio_feats_path = os.path.join(data_path, 'audio_feats.pkl')
    audio_feats_data_args = {
        'data_path': audio_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    audio_feats_data_args['max_seq_len'] = args.audio_seq_len = bm['max_seq_lengths']['audio_feats']
    audio_feats_data = get_v_a_data(audio_feats_data_args, audio_feats_path)
    

    # MMDataset 클래스를 사용해서 학습, 검증, 테스트 데이터를 각각 준비함
    train_data = MMDataset(train_label_ids, text_data['train'], video_feats_data['train'], audio_feats_data['train'], cons_text_feats['train'], condition_idx['train'])
    dev_data = MMDataset(dev_label_ids, text_data['dev'], video_feats_data['dev'], audio_feats_data['dev'], cons_text_feats['dev'], condition_idx['dev']) 
    test_data = MMDataset(test_label_ids, text_data['test'], video_feats_data['test'], audio_feats_data['test'], cons_text_feats['test'], condition_idx['test'])

    # 학습, 검증, 테스트 데이터를 딕셔너리에 넣어서 반환함
    data = {'train': train_data, 'dev': dev_data, 'test': test_data}     
    
    return data  # 준비된 데이터를 돌려줌
    

# 데이터셋에서 인덱스와 라벨을 가져오는 함수
def get_indexes_annotations(args, bm, label_list, read_file_path, data_mode):

    # 라벨을 숫자로 바꾸기 위한 맵(사전) 생성
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i  # 예: {"greeting": 0, "farewell": 1}

    # 파일을 열고 데이터를 한 줄씩 읽음
    with open(read_file_path, 'r') as f:
        data = csv.reader(f, delimiter="\t")  # 데이터를 탭(\t)으로 구분해서 읽어옴
        indexes = []  # 인덱스를 담을 리스트
        label_ids = []  # 라벨 ID를 담을 리스트

        for i, line in enumerate(data):
            if i == 0:
                continue  # 첫 줄(헤더)은 건너뜀
            
            # 데이터셋 종류에 따라 다르게 처리
            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])  # MIntRec 데이터셋에서는 첫 3개 요소를 합쳐서 인덱스 생성
                indexes.append(index)  # 인덱스 리스트에 추가
                
                label_id = label_map[line[4]]  # 라벨 ID 가져오기

            elif args.dataset in ['MELD']:
                index = '_'.join([line[0], line[1]])  # MELD 데이터셋에서는 첫 2개 요소를 합쳐서 인덱스 생성
                indexes.append(index)  # 인덱스 리스트에 추가
                
                label_id = label_map[bm['label_maps'][line[3]]]  # 라벨 ID 가져오기
            
            label_ids.append(label_id)  # 라벨 ID 리스트에 추가
    
    return indexes, label_ids  # 인덱스와 라벨 ID 리스트를 반환

'''
DataManager 클래스: 데이터를 관리하는 클래스, 필요한 데이터를 준비하고 불러옴
get_data 함수: 학습, 검증, 테스트 데이터를 준비하고 각각 MMDataset에 넣어서 반환
get_indexes_annotations 함수: 파일을 읽고 각 데이터의 인덱스와 라벨 ID를 가져옴
'''