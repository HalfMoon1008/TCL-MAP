import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import TCL_MAP
from .loss import SupConLoss
import numpy as np

__all__ = ['TCL_MAP_manager']

# TCL_MAP 모델을 관리하는 클래스 정의
class TCL_MAP_manager:

    def __init__(self, args, data):
             
        self.logger = logging.getLogger(args.logger_name)  # 로거 설정
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU가 있으면 사용, 없으면 CPU 사용
        args.device = self.device
        self.model = TCL_MAP(args)  # 모델 생성
        self.model.to(self.device)  # 모델을 지정된 장치로 이동
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)  # 옵티마이저와 스케줄러 설정

        # 데이터 로더 가져오기
        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        self.criterion = nn.CrossEntropyLoss()  # 분류 작업을 위한 손실 함수 설정
        self.cons_criterion = SupConLoss(temperature=args.temperature)  # 대조 손실 함수 설정
        self.metrics = Metrics(args)  # 성능 측정 메트릭 설정
        
        # 학습 모드일 때는 최상의 평가 점수를 0으로 초기화, 그렇지 않으면 모델을 복원
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)
            
    def _set_optimizer(self, args, model):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        # AdamW 옵티마이저 설정
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
        
        # 학습 스텝과 워밍업 스텝 설정
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        # 선형 스케줄러 설정
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)  # 조기 종료 설정
        
        # 에포크 반복
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()  # 모델을 학습 모드로 설정
            loss_record = AverageMeter()  # 손실 기록을 위한 초기화
            cons_loss_record = AverageMeter()  # 대조 손실 기록 초기화
            cls_loss_record = AverageMeter()  # 분류 손실 기록 초기화
            
            # 배치 반복
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)  # 텍스트 특징
                cons_text_feats = batch['cons_text_feats'].to(self.device)  # 대조 텍스트 특징
                condition_idx = batch['condition_idx'].to(self.device)  # 조건 인덱스
                video_feats = batch['video_feats'].to(self.device)  # 비디오 특징
                audio_feats = batch['audio_feats'].to(self.device)  # 오디오 특징
                label_ids = batch['label_ids'].to(self.device)  # 레이블
                
                with torch.set_grad_enabled(True):  # 그래디언트 계산 활성화

                    # 모델의 출력과 대조 특징, 조건 특징 추출
                    logits, _, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                    
                    # 대조 손실과 분류 손실 계산
                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    cons_loss = self.cons_criterion(cons_feature)
                    cls_loss = self.criterion(logits, label_ids)
                    loss = cls_loss + cons_loss  # 총 손실은 대조 손실과 분류 손실의 합
                    
                    # 그래디언트 초기화
                    self.optimizer.zero_grad()
                    
                    # 역전파 및 그래디언트 계산
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))

                    # 그래디언트 클리핑
                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    # 옵티마이저와 스케줄러 업데이트
                    self.optimizer.step()
                    self.scheduler.step()
            
            # 평가 결과 계산
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            # 결과를 로깅
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            # 조기 종료 여부 확인
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        # 최적의 평가 점수와 모델 저장
        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model  
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    # 모델의 출력 계산 (평가 또는 테스트 시 사용)
    def _get_outputs(self, args, dataloader, show_results=False):

        self.model.eval()  # 평가 모드로 설정

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
        
        # 배치 반복
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)  # 텍스트 특징
            cons_text_feats = batch['cons_text_feats'].to(self.device)  # 대조 텍스트 특징
            condition_idx = batch['condition_idx'].to(self.device)  # 조건 인덱스
            video_feats = batch['video_feats'].to(self.device)  # 비디오 특징
            audio_feats = batch['audio_feats'].to(self.device)  # 오디오 특징
            label_ids = batch['label_ids'].to(self.device)  # 레이블
                
            with torch.set_grad_enabled(False):  # 그래디언트 비활성화
                # 모델의 출력 계산
                logits, features, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                total_logits = torch.cat((total_logits, logits))  # 로짓 저장
                total_labels = torch.cat((total_labels, label_ids))  # 레이블 저장
                total_features = torch.cat((total_features, features))  # 피처 저장

        # 소프트맥스 적용하여 예측 확률 계산
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_logit = total_logits.cpu().numpy()  # 로짓을 넘파이로 변환
        y_pred = total_preds.cpu().numpy()  # 예측을 넘파이로 변환
        y_true = total_labels.cpu().numpy()  # 실제 레이블을 넘파이로 변환
        y_prob = total_maxprobs.cpu().numpy()  # 최대 확률을 넘파이로 변환
        y_feat = total_features.cpu().numpy()  # 피처를 넘파이로 변환
        
        # 메트릭을 사용하여 성능 측정
        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        
        # 예측 결과 저장 옵션
        if args.save_pred and show_results:
            np.save('y_true_' + str(args.seed) + '.npy', y_true)
            np.save('y_pred_' + str(args.seed) + '.npy', y_pred)

        # 출력 결과에 추가 정보 저장
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    # 테스트 단계 실행
    def _test(self, args):
        
        test_results = {}
        
        # 테스트 데이터셋에서 결과 출력
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results=True)
        if args.train:
            ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        
        # 테스트 결과 저장
        test_results.update(ind_outputs)
        
        return test_results
