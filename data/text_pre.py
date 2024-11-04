import os
import csv
import sys
import pickle
from transformers import BertTokenizer
import numpy as np

# 이 함수는 데이터를 가져오는 역할을 함
# 'get_data'라는 다른 함수를 호출해서 필요한 데이터를 불러옴
def get_t_data(args, data_args):
    
    # 'get_data' 함수에서 데이터 가져오기
    t_data, cons_text_feats, condition_idx = get_data(args, data_args)

    # 가져온 데이터를 다시 돌려줌
    return t_data, cons_text_feats, condition_idx

# 데이터 읽어오는 함수
def get_data(args, data_args):

    # 데이터 처리를 위한 'DatasetProcessor'준비
    processor = DatasetProcessor(args)
    data_path = data_args['data_path']  # 데이터 파일이 저장된 경로를 가져옴

    # 학습용 데이터 가져오기
    # processor.get_examples(data_dir, mode) | mode = 'train', 'dev', 'test', 'all'
    train_examples = processor.get_examples(data_path, 'train') 
    train_feats, train_cons_text_feats, train_condition_idx = get_backbone_feats(args, data_args, train_examples)

    # 검증용 데이터 가져오기
    dev_examples = processor.get_examples(data_path, 'dev')
    dev_feats, dev_cons_text_feats, dev_condition_idx = get_backbone_feats(args, data_args, dev_examples)

    # 테스트용 데이터 가져오기
    test_examples = processor.get_examples(data_path, 'test')
    test_feats, test_cons_text_feats, test_condition_idx = get_backbone_feats(args, data_args, test_examples)

    # 데이터들 모아놓기 (학습용, 검증용, 테스트용 각각 따로)
    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    cons_text_feats = {
        'train': train_cons_text_feats,
        'dev': dev_cons_text_feats,
        'test': test_cons_text_feats
    }

    condition_idx = {
        'train': train_condition_idx,
        'dev': dev_condition_idx,
        'test': test_condition_idx
    }

    return outputs, cons_text_feats, condition_idx

# 'BertTokenizer'를 사용해서 문장을 숫자 형식으로 바꾸는 함수
def get_backbone_feats(args, data_args, examples):

    # BERT 모델에서 단어를 숫자로 바꾸는 도구 준비
    tokenizer = BertTokenizer.from_pretrained(args.text_backbone, do_lower_case=True)   
    
    # 길이 설정하기
    data_args['prompt_len'] = args.prompt_len
    data_args['label_len'] = args.label_len
    
    # 데이터를 BERT가 이해할 수 있는 형식으로 바꿔줌
    features, cons_features, condition_idx, args.max_cons_seq_length = convert_examples_to_features(args, examples, data_args, tokenizer)     
    features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
    cons_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_features]
    return features_list, cons_features_list, condition_idx

# InputExample 클래스는 문장이나 데이터를 저장하기 위한 상자 역할을 함
class InputExample(object):
    """간단한 문장을 담는 상자처럼 사용함"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """상자에 필요한 정보를 넣음.
        Args:
            guid: 예제의 고유 ID
            text_a: 첫 번째 문장
            text_b: 두 번째 문장 (필요할 때만 사용)
            label: 이 문장이 어떤 종류인지 알려주는 라벨
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

# InputFeatures는 숫자로 바뀐 데이터를 담는 상자
class InputFeatures(object):
    """한 문장이 숫자로 변환된 데이터를 담는 상자임"""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

# 데이터를 다루기 위한 기본 클래스
class DataProcessor(object):
    """문장 데이터를 불러오고 읽기 위한 기본 클래스"""

    # pickle 파일에서 데이터 읽어오기
    @classmethod
    def _read_pkl(cls, input_file, quotechar=None):
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        lines = [] 
        for k, v in data.items():
            line = k.split('_')
            line.append(v)
            lines.append(line)
        return lines

    # 탭으로 구분된 파일 읽기
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """탭으로 구분된 파일을 읽는 함수"""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

# 특정 데이터셋을 위한 처리 class
class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()
        
        if args.dataset in ['MIntRec']:
            self.select_id = 3
            self.label_id = 4
        elif args.dataset in ['MELD']:
            self.select_id = 2
            self.label_id = 3
        
    def get_examples(self, data_dir, mode):
        # 데이터 모드에 따라 맞는 파일을 불러옴
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")
        
    # 읽어온 파일에서 데이터를 `InputExample` 상자로 만들어줌
    def _create_examples(self, lines, set_type):
        """훈련과 검증 데이터를 위한 예제를 만듦"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)  # 예제마다 고유한 이름 만들기
            text_a = line[self.select_id]  # 첫 번째 문장 가져오기
            label = line[self.label_id]    # 라벨 가져오기

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# 문장을 BERT 모델에 맞는 형식으로 바꿔주는 함수
def convert_examples_to_features(args, examples, data_args, tokenizer):
    """파일을 BERT가 사용할 수 있는 숫자 형식으로 변환함"""
        
    max_seq_length = data_args['max_seq_len']
    label_len = data_args['label_len']
    features = []
    cons_features = []
    condition_idx = []
    prefix = ['MASK'] * data_args['prompt_len']  # 프롬프트 길이만큼 마스크 추가

    max_cons_seq_length = max_seq_length + len(prefix) + label_len
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if args.dataset in ['MIntRec']:
            condition = tokenizer.tokenize(example.label)
        elif args.dataset in ['MELD']:
            condition = tokenizer.tokenize(data_args['bm']['label_maps'][example.label])

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_cons_seq_length - len(input_ids))
        input_ids += padding
        cons_inputs_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_cons_seq_length
        assert len(cons_inputs_ids) == max_cons_seq_length
        assert len(input_mask) == max_cons_seq_length
        assert len(segment_ids) == max_cons_seq_length
        condition_idx.append(1 + len(tokens_a) + len(prefix))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
        
        cons_features.append(
            InputFeatures(input_ids=cons_inputs_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features, cons_features, condition_idx, max_cons_seq_length

# 두 개의 문장이 너무 길면 적당한 길이로 자르는 함수
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """주어진 최대 길이에 맞춰 문장을 잘라줌"""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # 첫 번째 문장의 앞부분을 자름
        else:
            tokens_b.pop()  # 두 번째 문장의 뒷부분을 자름