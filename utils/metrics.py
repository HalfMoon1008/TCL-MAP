from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score, roc_auc_score, average_precision_score, \
    auc, precision_recall_curve, roc_curve
        
from scipy.optimize import brentq
from scipy.interpolate import interp1d
        
import logging
import numpy as np

# 학습 또는 평가 동안의 평균값을 계산하는 클래스
class AverageMeter(object):
    """ 현재 값과 평균값을 계산 및 저장 """

    def __init__(self):
        self.reset()  # 객체 초기화 시 reset 메소드 호출

    def reset(self):
        """ 모든 값들을 0으로 초기화 """
        self.val = 0  # 현재 값
        self.avg = 0  # 평균값
        self.sum = 0  # 값들의 합계
        self.count = 0  # 값의 개수

    def update(self, val, n=1):
        """ 새로운 값을 받아 평균을 업데이트
        val: 새로 추가되는 값
        n: 해당 값의 개수 (기본값은 1)
        """
        self.val = val  # 현재 값 업데이트
        self.sum += val * n  # 합계에 새로운 값을 반영
        self.count += n  # 총 개수에 n을 더함
        self.avg = float(self.sum) / self.count  # 새로운 평균 계산

# 다양한 성능 지표를 계산하는 클래스
class Metrics(object):
    """
    혼동 행렬(confusion matrix)의 열은 예측된 인덱스, 행은 실제(target) 인덱스를 나타냄
    """
    def __init__(self, args):
        # 로거(logger) 설정 (args에서 로거 이름 가져옴)
        self.logger = logging.getLogger(args.logger_name)
        # 평가에 사용할 지표 목록 설정
        self.eval_metrics = ['acc', 'f1', 'prec', 'rec', 'weighted_f1', 'weighted_prec', 'weighted_rec']

    # 호출 시 성능 지표들을 계산하는 메소드
    def __call__(self, y_true, y_pred, show_results=False):
        """
        y_true: 실제 레이블
        y_pred: 예측된 레이블
        show_results: 결과를 로그에 표시할지 여부 (기본값은 False)
        """

        # 다양한 지표 계산
        acc_score = self._acc_score(y_true, y_pred)  # 정확도
        macro_f1, weighted_f1 = self._f1_score(y_true, y_pred)  # F1-스코어 (macro와 weighted)
        macro_prec, weighted_prec = self._precision_score(y_true, y_pred)  # Precision (macro와 weighted)
        macro_rec, weighted_rec = self._recall_score(y_true, y_pred)  # Recall (macro와 weighted)
        
        # 계산된 지표를 딕셔너리로 저장
        eval_results = {
            'acc': acc_score,  # 정확도
            'f1': macro_f1,  # F1-스코어 (macro)
            'weighted_f1': weighted_f1,  # F1-스코어 (weighted)
            'prec': macro_prec,  # Precision (macro)
            'weighted_prec': weighted_prec,  # Precision (weighted)
            'rec': macro_rec,  # Recall (macro)
            'weighted_rec': weighted_rec  # Recall (weighted)
        }
        
        # 결과를 로그에 표시하는 옵션
        if show_results:
            # confusion matrix행렬을 표시
            self._show_confusion_matrix(y_true, y_pred)

            # 로그에 성능 결과 출력
            self.logger.info("***** In-domain Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

        return eval_results  # 계산된 성능 지표 반환

    # 정확도 계산 함수
    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    # F1-스코어 계산 함수 (macro와 weighted 평균)
    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted')

    # Precision 계산 함수 (macro와 weighted 평균)
    def _precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='weighted')

    # Recall 계산 함수 (macro와 weighted 평균)
    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='weighted')

    # confusion matrix 행렬을 로그에 표시하는 함수
    def _show_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)  # confusion matrix 행렬 계산
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))  # confusion matrix 행렬을 로그에 출력
