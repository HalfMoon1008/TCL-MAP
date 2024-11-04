from configs.base import ParamManager, add_config_param  # 파라미터 설정 관리
from data.base import DataManager  # 데이터 관리
from methods import method_map  # 메소드 매핑
from utils.functions import set_torch_seed, save_results, set_output_path  # 유틸리티 함수들

import argparse  # 커맨드라인 인자 파싱
import logging  # 로깅 설정
import os
import datetime
import itertools  # 반복을 위한 모듈
import warnings
import copy 

# 커맨드라인에서 받을 인자들을 정의해주는 함수
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # 로거 이름 설정, 기본값은 'Multimodal Intent Recognition'
    parser.add_argument('--logger_name', type=str, default='Multimodal Intent Recognition', help="Logger name for multimodal intent recognition.")

    # 데이터셋 이름 선택, 기본값은 'MIntRec'
    parser.add_argument('--dataset', type=str, default='MIntRec', help="The selected person id.")

    # 데이터 모드 선택, 기본값은 'multi-class'
    parser.add_argument('--data_mode', type=str, default='multi-class', help="The selected person id.")

    # 사용할 방법(method) 선택, 기본값은 'TCL-MAP'
    parser.add_argument('--method', type=str, default='TCL-MAP', help="which method to use.")

    # 텍스트 백본 설정, 기본값은 'bert-base-uncased'
    parser.add_argument("--text_backbone", type=str, default='bert-base-uncased', help="which backbone to use for text modality")

    # 랜덤 시드 설정, 기본값은 0
    parser.add_argument('--seed', type=int, default=0, help="The selected person id.")

    # 데이터 로딩에 사용할 워커(worker) 수 설정, 기본값은 8
    parser.add_argument('--num_workers', type=int, default=8, help="The number of workers to load data.")

    # 로그 파일의 인덱스 설정, 기본값은 None
    parser.add_argument('--log_id', type=str, default=None, help="The index of each logging file.")
    
    # 사용할 GPU ID 설정, 기본값은 '0'
    parser.add_argument('--gpu_id', type=str, default='0', help="The selected person id.")

    # 데이터 경로 설정
    parser.add_argument("--data_path", default='/Datasets', type=str, help="The input data dir. Should contain text, video and audio data for the task.")

    # 학습 모드 설정
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")

    # 하이퍼 파라미터 튜닝 모드 설정
    parser.add_argument("--tune", action="store_true", help="Whether to tune the model with a series of hyper-parameters.")

    # 학습한 모델을 저장할지 여부 설정
    parser.add_argument("--save_model", action="store_true", help="save trained-model for multimodal intent recognition.")

    # 최종 결과를 저장할지 여부 설정
    parser.add_argument("--save_results", action="store_true", help="save final results for multimodal intent recognition.")

    # 로그 파일 저장 경로 설정
    parser.add_argument('--log_path', type=str, default='logs', help="Logger directory.")
    
    # 캐시 디렉토리 설정
    parser.add_argument('--cache_path', type=str, default='cache', help="The caching directory for pre-trained models.")   

    # 결과 저장 경로 설정
    parser.add_argument('--results_path', type=str, default='results', help="The path to save results.")

    # 출력 경로 설정
    parser.add_argument("--output_path", default='outputs', type=str, help="The output directory where all train data will be written.") 

    # 모델 저장 경로 설정
    parser.add_argument("--model_path", default='models', type=str, help="The output directory where the model predictions and checkpoints will be written.") 

    # 설정 파일 이름
    parser.add_argument("--config_file_name", type=str, default='TCL_MAP_MIntRec.py', help="The name of the config file.")

    # 결과 파일 이름
    parser.add_argument("--results_file_name", type=str, default='results.csv', help="The file name of all the results.")    
    
    # 예측 결과 저장 여부 설정
    parser.add_argument('--save_pred', type=bool, default=False, help="Logger directory.")
    
    # 인자들을 파싱해서 args로 반환
    args = parser.parse_args()
    return args


# 로깅 설정하는 함수
def set_logger(args):
    
    # 로그 파일 경로가 없다면 만들어줌
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    # 시간 포맷 설정
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 로거 이름 설정
    args.logger_name = f"{args.method}_{args.dataset}_{args.data_mode}_{args.seed}"
    # 로그 ID 설정
    args.log_id = f"{args.logger_name}_{time}"
    
    # 로거 가져오기
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러 설정
    log_path = os.path.join(args.log_path, args.log_id + '.log')
    fh = logging.FileHandler(log_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # 콘솔 출력 핸들러 설정
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    return logger


# 모델과 관련된 기본 설정
def set_up(args):
    
    # 모델 이름 설정
    save_model_name = f"{args.method}_{args.dataset}_{args.text_backbone}_{args.data_mode}_{args.seed}"
    
    # 결과 및 모델 출력 경로 설정
    args.pred_output_path, args.model_output_path = set_output_path(args, save_model_name)
    
    # 시드 설정 (모델의 재현성을 위해)
    set_torch_seed(args.seed)
    
    return args


# 모델 학습과 평가를 수행하는 메인 함수
def work(args, data, logger, debug_args=None, ind_args=None):
    
    # 시드 다시 설정
    set_torch_seed(args.seed)
    
    # 메소드 매핑에서 해당 메소드 가져오기
    method_manager = method_map[args.method]
    method = method_manager(args, data)
        
    # 의도 인식 시작 로그
    logger.info('Intent Recognition begins...')

    if args.train:
        # 학습 시작 로그
        logger.info('Training begins...')
        method._train(args)
        logger.info('Training is finished...')  # 학습 완료 로그

    # 테스트 시작 로그
    logger.info('Testing begins...')
    outputs = method._test(args)
    logger.info('Testing is finished...')
    logger.info('Intent recognition is finished...')
    
    if args.save_results:
        # 결과 저장 로그
        logger.info('Results are saved in %s', str(os.path.join(args.results_path, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)


# 하이퍼파라미터 튜닝을 위한 함수
def run(args, data, logger, ind_args=None):
    debug_args = {}

    # 리스트 형태의 인자들만 가져와서 debug_args에 추가
    for k, v in args.items():
        if isinstance(v, list):
            debug_args[k] = v
        
    # 모든 조합을 순서대로 적용하여 모델 학습 및 테스트 실행
    for result in itertools.product(*debug_args.values()):
        for i, key in enumerate(debug_args.keys()):
            args[key] = result[i]         
        
        # 조합마다 work 함수 호출
        work(args, data, logger, debug_args, ind_args)


# 메인 실행 부분
if __name__ == '__main__':
    
    warnings.filterwarnings('ignore')  # 경고 메시지 무시
    
    # 인자 파싱
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # GPU 설정
    
    # 파라미터 매니저 설정 및 인자 추가
    param = ParamManager(args)
    args = param.args
    args = add_config_param(args, args.config_file_name)
    args = set_up(args)
    
    # 데이터 매니저 생성
    data = DataManager(args)
    logger = set_logger(args)
    
    # 로그 설정
    logger = set_logger(args)
    logger.info("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.info(f"{k}: {args[k]}")
    logger.info("="*30+" End Params "+"="*30)
    
    # 모델 실행
    run(args, data, logger)
