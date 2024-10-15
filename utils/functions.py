import os
import torch
import numpy as np
import pandas as pd
import random
import logging
import copy
from .metrics import Metrics

# 조기 종료를 위한 클래스 정의
class EarlyStopping:
    """검증 손실이 개선되지 않을 경우 학습을 조기에 종료하는 클래스"""
    def __init__(self, args, delta=1e-6):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않았을 때 기다릴 에포크 수.
            delta (float): 모니터링하는 값의 최소 변화량. 개선으로 간주되는 최소 변화량.
        """
        self.patience = args.wait_patience  # 기다릴 에포크 수
        self.logger = logging.getLogger(args.logger_name)  # 로거 설정
        self.monitor = args.eval_monitor  # 모니터할 메트릭 ('loss' 또는 다른 메트릭)
        self.counter = 0  # 개선되지 않은 에포크 수를 카운트
        # 모니터할 값에 따라 초기 최적 점수 설정 (loss인 경우 매우 큰 값, 그 외는 매우 작은 값)
        self.best_score = 1e8 if self.monitor == 'loss' else 1e-6
        self.early_stop = False  # 조기 종료 여부 플래그
        self.delta = delta  # 개선으로 간주될 최소 변화량
        self.best_model = None  # 최적 모델 저장

    # 클래스를 호출했을 때 실행되는 메소드
    def __call__(self, score, model):
        # 개선 여부 판단
        better_flag = score <= (self.best_score - self.delta) if self.monitor == 'loss' else score >= (self.best_score + self.delta)

        if better_flag:  # 성능이 개선된 경우
            self.counter = 0  # 카운터 초기화
            self.best_model = copy.deepcopy(model)  # 모델 저장
            self.best_score = score  # 최적 점수 업데이트

        else:  # 개선되지 않은 경우
            self.counter += 1  # 카운터 증가
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 로그 출력

            if self.counter >= self.patience:  # patience만큼 개선되지 않으면 학습 종료
                self.early_stop = True
         
# 랜덤 시드 설정 함수
def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 파이토치 시드 설정
    torch.cuda.manual_seed(seed)  # CUDA 시드 설정
    torch.cuda.manual_seed_all(seed)  # 여러 GPU 사용 시 모든 GPU의 시드 설정
    torch.backends.cudnn.deterministic = True  # CUDNN 동작을 결정적으로 설정
    torch.backends.cudnn.benchmark = False  # CUDNN 성능 최적화를 비활성화
    os.environ['PYTHONHASHSEED'] = str(seed)  # 파이썬 해시 시드 설정

# 출력 경로 설정 함수
def set_output_path(args, save_model_name):

    if not os.path.exists(args.output_path):  # 출력 경로가 존재하지 않으면 생성
        os.makedirs(args.output_path)

    pred_output_path = os.path.join(args.output_path, save_model_name)  # 모델 저장 경로 설정
    if not os.path.exists(pred_output_path):  # 모델 저장 경로가 없으면 생성
        os.makedirs(pred_output_path)

    model_path = os.path.join(pred_output_path, args.model_path)  # 모델 파일 경로 설정
    if not os.path.exists(model_path):  # 모델 파일 경로가 없으면 생성
        os.makedirs(model_path)

    return pred_output_path, model_path  # 저장 경로 반환

# 넘파이 파일 저장 함수
def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)  # 저장 경로 설정
    np.save(npy_path, npy_file)  # 넘파이 파일 저장

# 넘파이 파일 로드 함수
def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)  # 파일 경로 설정
    npy_file = np.load(npy_path)  # 넘파이 파일 로드
    return npy_file

# 모델 저장 함수
def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  # 모델이 병렬 처리 모드로 되어있다면 원래 모델을 가져옴
    model_file = os.path.join(model_dir, 'pytorch_model.bin')  # 모델 파일 경로 설정

    torch.save(save_model.state_dict(), model_file)  # 모델의 가중치 저장

# 모델 복원 함수
def restore_model(model, model_dir, device):
    output_model_file = os.path.join(model_dir, 'pytorch_model.bin')  # 모델 파일 경로 설정
    m = torch.load(output_model_file, map_location=device)  # 모델 가중치를 로드
    model.load_state_dict(m)  # 모델에 가중치 적용
    return model  # 모델 반환

# 테스트 결과를 저장하는 함수
def save_results(args, test_results, debug_args=None):
    
    save_keys = ['y_pred', 'y_true', 'features', 'scores']  # 저장할 키 목록
    for s_k in save_keys:
        if s_k in test_results.keys():  # 결과에 해당 키가 있으면 저장
            save_path = os.path.join(args.output_path, s_k + '.npy')
            np.save(save_path, test_results[s_k])

    results = {}
    metrics = Metrics(args)  # 메트릭 클래스 인스턴스화
    
    for key in metrics.eval_metrics:  # 평가 메트릭을 테스트 결과에서 가져옴
         if key in test_results.keys():
            results[key] = round(test_results[key] * 100, 2)  # 결과를 백분율로 저장

    if 'best_eval_score' in test_results:  # 최적 평가 점수가 있으면 결과에 추가
        eval_key = 'eval_' + args.eval_monitor
        results.update({eval_key: test_results['best_eval_score']})

    # 기록할 변수 및 이름 설정
    _vars = [args.dataset, args.method, args.text_backbone, args.seed, args.log_id]
    _names = ['dataset', 'method', 'text_backbone', 'seed', 'log_id']
    
    if debug_args is not None:  # 디버그 인자가 있으면 변수와 이름에 추가
        _vars.extend([args[key] for key in debug_args.keys()])
        _names.extend(debug_args.keys())

    vars_dict = {k: v for k, v in zip(_names, _vars)}  # 이름과 변수를 딕셔너리로 변환
    results = dict(results, **vars_dict)  # 결과에 추가

    keys = list(results.keys())  # 키 리스트 생성
    values = list(results.values())  # 값 리스트 생성
    
    if not os.path.exists(args.results_path):  # 결과 경로가 없으면 생성
        os.makedirs(args.results_path)
        
    results_path = os.path.join(args.results_path, args.results_file_name)  # 결과 파일 경로 설정
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:  # 결과 파일이 없거나 크기가 0이면 새로 작성
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)  # 결과를 데이터프레임으로 변환
        df1.to_csv(results_path, index=False)  # CSV 파일로 저장
    else:
        df1 = pd.read_csv(results_path)  # 기존 결과 파일 읽기
        new = pd.DataFrame(results, index=[1])  # 새로운 결과 추가
        df1 = df1.append(new, ignore_index=True)  # 기존 결과에 추가
        df1.to_csv(results_path, index=False)  # 업데이트된 결과를 저장
    data_diagram = pd.read_csv(results_path)  # 결과 파일 읽기
    
    print('test_results', data_diagram)  # 테스트 결과 출력