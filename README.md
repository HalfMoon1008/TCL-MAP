# TCL-MAP

```plaintext
TCL-MAP/
│
├── data/
│ └── utils.py # 데이터 로드 및 전처리와 관련된 유틸리티 함수들
│
├── methods/TCL_MAP/
│ ├── SubNets/
│ │ ├── transformers_encoder/
│ │ │ ├── multihead_attention.py # 멀티헤드 어텐션을 구현한 모듈
│ │ │ ├── position_embedding.py # 포지션 임베딩을 구현한 모듈
│ │ │ └── transformer.py # Transformer 인코더 관련 모듈
│ │ ├── FeatureNets.py # 각 모달리티(텍스트, 오디오, 비디오 등)로부터 피처를 추출하는 네트워크를 정의하기 위한 모듈
│ │ └── AlignNets.py # 모달리티 간의 정렬을 수행하는 네트워크를 정의하기 위한 모듈
│ ├── loss.py # 손실 함수 정의 (ex. Contrastive Loss)
│ ├── manager.py # 모델 학습 관리, 옵티마이저 및 스케줄러 설정, 훈련 및 평가 로직
│ ├── model.py # TCL-MAP 모델의 전체 구조 정의 및 구현 | 모듈들이 어떻게 연결되고 동작하는지를 구현한 핵심 파일
│
├── utils/
│ ├── functions.py # 유틸리티 함수들 (ER | 모델 저장, 복원, 테스트 저장)
│ └── metrics.py # 평가 메트릭스 정의 (정확도, F1 스코어 등)
'''
---

Original source: https://github.com/thuiar/TCL-MAP
```
