import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from .SubNets.transformers_encoder.transformer import TransformerEncoder
from .AlignNets import AlignSubNet

# MAG 클래스 정의: 멀티모달 정보를 통합하는 역할
class MAG(nn.Module):
    def __init__(self,  config, args):
        super(MAG, self).__init__()
        self.args = args

        # 필요에 따라 정렬 네트워크 사용 여부 결정
        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.mag_aligned_method)

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
        # 텍스트와 비디오 특징을 결합하여 텍스트 특징 차원으로 매핑하는 선형층
        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        # 텍스트와 오디오 특징을 결합하여 텍스트 특징 차원으로 매핑하는 선형층
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)

        # 비디오 특징을 텍스트 특징 차원으로 매핑하는 선형층
        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        # 오디오 특징을 텍스트 특징 차원으로 매핑하는 선형층
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)

        # 베타 시프트 값 설정
        self.beta_shift = args.beta_shift

        # 층 정규화와 드롭아웃 레이어 정의
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6  # 작은 값 epsilon 설정

        # 필요에 따라 입력 특징들을 정렬
        if self.args.need_aligned:
            text_embedding, visual, acoustic  = self.alignNet(text_embedding, visual, acoustic)
        
        # 텍스트와 비디오 특징을 결합하여 가중치 계산
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        # 텍스트와 오디오 특징을 결합하여 가중치 계산
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        # 비디오와 오디오 특징에 가중치를 적용하여 멀티모달 표현 생성
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        # 텍스트 임베딩과 멀티모달 표현의 L2 노름 계산
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        # 노름이 0인 경우를 방지하기 위한 처리
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        # 스케일링 계수 계산
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)

        # 알파 값 계산 (최대값을 1로 제한)
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        # 최종 멀티모달 임베딩 생성
        acoustic_vis_embedding = alpha * h_m

        # 멀티모달 임베딩과 텍스트 임베딩을 합하고 정규화 및 드롭아웃 적용
        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output

# MAP 클래스 정의: 멀티모달 정보를 BERT 모델에 통합
class MAP(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config

        # BERT의 기본 구성 요소 로드
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # 멀티모달 융합 레이어 (MAG)
        self.MAG = MAG(
            config, args
        )
        self.args = args

        # MAP 모듈의 구성 설정
        self.alignNet = AlignSubNet(args, args.aligned_method)
        self.embed_dim = args.text_feat_dim
        self.num_heads = args.nheads
        self.layers = args.n_levels
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        # 오디오, 비디오, 텍스트 특징을 임베딩 차원으로 매핑하는 선형층 정의
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(args.audio_feat_dim),
            nn.Linear(args.audio_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(args.video_feat_dim),
            nn.Linear(args.video_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(args.text_feat_dim),
            nn.Linear(args.text_feat_dim, self.embed_dim),
        )

        # 출력 프로젝션 레이어 정의
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, args.text_feat_dim)
        )

        # Transformer 인코더 초기화 (오디오 및 비디오 특징을 통합)
        self.trans_a_with_l = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=self.attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask
        )
        
        # 감마 파라미터 초기화
        self.gamma = nn.Parameter(torch.ones(args.text_feat_dim) * 1e-4)

        self.init_weights()  # 가중치 초기화

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ 모델의 어텐션 헤드를 제거하는 메소드.
            heads_to_prune: {layer_num: 제거할 헤드의 리스트} 형태의 딕셔너리
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        condition_idx,
        ctx,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
    
        r"""
        반환 값:
            다양한 요소로 구성된 튜플을 반환하며, 구성 및 입력에 따라 다름.

            last_hidden_state (torch.FloatTensor, shape: (batch_size, sequence_length, hidden_size)):
                모델의 마지막 레이어에서의 시퀀스 히든 상태.
            pooler_output (torch.FloatTensor, shape: (batch_size, hidden_size)):
                첫 번째 토큰의 마지막 레이어 히든 상태를 Linear 레이어와 Tanh 활성화 함수를 통해 추가 처리한 값.
            hidden_states (tuple(torch.FloatTensor), 선택 사항):
                각 레이어의 출력 및 임베딩의 히든 상태를 포함하는 튜플.
            attentions (tuple(torch.FloatTensor), 선택 사항):
                각 레이어의 어텐션 가중치를 포함하는 튜플.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # 입력 ID와 임베딩을 동시에 지정할 수 없음
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "input_ids와 inputs_embeds를 동시에 지정할 수 없습니다."
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "input_ids 또는 inputs_embeds 중 하나를 지정해야 합니다."
            )

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device
            )

        # 어텐션 마스크 확장
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # 디코더인 경우 인코더의 어텐션 마스크 처리
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # 헤드 마스크 준비
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        # 기본 텍스트 임베딩 획득
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # 모달리티-인식 프롬프트 생성 및 적용
        batch_ctx = ctx.unsqueeze(0).repeat(acoustic.shape[0], 1, 1)
        _, aligned_visual, aligned_acoustic  = self.alignNet(batch_ctx, visual, acoustic)
        aligned_acoustic = self.audio_proj(aligned_acoustic)
        aligned_visual = self.video_proj(aligned_visual)
        batch_ctx = self.text_proj(batch_ctx)
        generated_ctx = self.trans_a_with_l(
            batch_ctx.permute(1, 0, 2),
            aligned_visual.permute(1, 0, 2),
            aligned_acoustic.permute(1, 0, 2)
        ).permute(1, 0, 2)
        generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma
        for i in range(embedding_output.shape[0]):
            # 프롬프트 위치에 생성된 컨텍스트 삽입
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = generated_ctx[i]

        
        # MAG를 사용한 초기 융합
        fused_embedding = self.MAG(embedding_output, visual, acoustic)

        # BERT 인코더를 통해 토큰들을 정제
        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # hidden_states 및 attentions 추가
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs, generated_ctx  # 출력 및 생성된 컨텍스트 반환
    
# MAP 모델 클래스 정의: 분류 작업을 위한 모델
class MAP_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len

        self.bert = MAP(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)

        self.init_weights()

    def forward(
        self,
        text,
        visual,
        acoustic,
        condition_idx,
        ctx,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (torch.LongTensor of shape (batch_size,), 선택 사항):
            시퀀스 분류/회귀 손실 계산을 위한 레이블.
            인덱스는 [0, ..., config.num_labels - 1] 범위 내에 있어야 함.
            config.num_labels == 1 인 경우 회귀 손실(MSE)이 계산됨.
            config.num_labels > 1 인 경우 분류 손실(Cross-Entropy)이 계산됨.
        반환 값:
            다양한 요소로 구성된 튜플을 반환하며, 구성 및 입력에 따라 다름.

            loss (torch.FloatTensor of shape (1,), 선택 사항):
                분류(또는 회귀) 손실.
            logits (torch.FloatTensor of shape (batch_size, config.num_labels)):
                분류(또는 회귀) 점수 (SoftMax 이전 값).
            hidden_states (tuple(torch.FloatTensor), 선택 사항):
                각 레이어의 출력 및 임베딩의 히든 상태를 포함하는 튜플.
            attentions (tuple(torch.FloatTensor), 선택 사항):
                각 레이어의 어텐션 가중치를 포함하는 튜플.
        """
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        # BERT 모델을 통해 피처 추출
        outputs, generated_ctx = self.bert(
            input_ids,
            visual,
            acoustic,
            condition_idx,
            ctx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        # 조건 인덱스를 사용하여 필요한 위치의 토큰 추출
        condition_tuple = tuple(
            sequence_output[torch.arange(sequence_output.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1)
            for i in range(self.label_len)
        )
        condition = torch.cat(condition_tuple, dim=1)
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # hidden_states 및 attentions 추가

        if labels is not None:
            if self.num_labels == 1:
                # 회귀 작업인 경우
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 분류 작업인 경우
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # 손실 추가
            
        return outputs, pooled_output, condition, generated_ctx  # 결과 반환


# Cons_Model 클래스 정의: BERT 모델을 사용하여 텍스트 데이터를 처리
class Cons_Model(BertPreTrainedModel):
    """
    이 모델은 셀프-어텐션만 사용하는 인코더로 동작할 수도 있고, 디코더로 설정되어 교차-어텐션 레이어를 추가할 수도 있음.
    """

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.args = args
        # 가중치 초기화 및 최종 처리 적용
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        모델의 어텐션 헤드를 제거하는 메소드.
        heads_to_prune: {layer_num: 제거할 헤드의 리스트} 형태의 딕셔너리
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        condition_idx,
        ctx,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), 선택 사항):
            인코더의 마지막 레이어 출력 히든 상태. 모델이 디코더로 설정된 경우 교차-어텐션에서 사용됨.
        encoder_attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), 선택 사항):
            인코더 입력의 패딩 토큰 인덱스에 대한 어텐션을 방지하기 위한 마스크.
            모델이 디코더로 설정된 경우 교차-어텐션에서 사용됨.
            마스크 값은 [0, 1]에서 선택됨:
            - 1은 마스킹되지 않은 토큰을 의미함.
            - 0은 마스킹된 토큰을 의미함.
        past_key_values (tuple(tuple(torch.FloatTensor)), 길이 config.n_layers, 각 튜플은 4개의 텐서로 구성됨):
            어텐션 블록의 사전 계산된 키와 값 히든 상태를 포함함. 디코딩 속도를 높이기 위해 사용될 수 있음.
            past_key_values를 사용하는 경우, 마지막 decoder_input_ids만 입력할 수 있음.
        use_cache (bool, 선택 사항):
            True로 설정된 경우, past_key_values 키 값 상태가 반환되어 디코딩 속도를 높이는 데 사용될 수 있음.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # 입력 ID와 임베딩을 동시에 지정할 수 없음
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids와 inputs_embeds를 동시에 지정할 수 없습니다.")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("input_ids 또는 inputs_embeds 중 하나를 지정해야 합니다.")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length 계산
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 어텐션 마스크 확장
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # 디코더인 경우 인코더의 어텐션 마스크 처리
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # 헤드 마스크 준비
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 증강된 샘플의 임베딩 획득
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # 모달리티-인식 프롬프트 적용
        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = ctx[i]

        # BERT 인코더를 통해 토큰들을 정제
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# TCL_MAP 클래스 정의: 전체 모델을 구성하는 클래스
class TCL_MAP(nn.Module):
    def __init__(self, args):
        
        super(TCL_MAP, self).__init__()
        
        # MAP 모델과 Cons 모델 초기화
        self.model = MAP_Model.from_pretrained(
            args.text_backbone, cache_dir=args.cache_path, args=args
        )
        self.cons_model = Cons_Model.from_pretrained(
            args.text_backbone, cache_dir=args.cache_path, args=args
        )
        
        # 컨텍스트 벡터 초기화
        self.ctx_vectors = self._init_ctx(args)
        self.ctx = nn.Parameter(self.ctx_vectors)

        self.label_len = args.label_len
        args.feat_size = args.text_feat_dim
        args.video_feat_size = args.video_feat_dim
        args.audio_feat_size = args.audio_feat_dim

    def _init_ctx(self, args):
        # 컨텍스트 벡터를 무작위로 초기화
        ctx = torch.empty(args.prompt_len, args.text_feat_dim, dtype=torch.float)
        nn.init.trunc_normal_(ctx)
        return ctx

    
    def forward(self, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):
        video_feats = video_feats.float()
        audio_feats = audio_feats.float()

        # 일반 샘플 처리
        outputs, pooled_output, condition, generated_ctx = self.model(
            text=text_feats,
            visual=video_feats,
            acoustic=audio_feats,
            condition_idx=condition_idx, 
            ctx=self.ctx
        )

        # 증강 샘플 처리
        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:, 1], cons_text_feats[:, 2]
        cons_outputs = self.cons_model(
            input_ids=cons_input_ids, 
            condition_idx=condition_idx,
            ctx=generated_ctx,
            token_type_ids=cons_segment_ids, 
            attention_mask=cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state
        cons_condition_tuple = tuple(
            last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1)
            for i in range(self.label_len)
        )
        cons_condition = torch.cat(cons_condition_tuple, dim=1)

        # 분류 피처와 레이블/[MASK] 토큰 표현 반환
        return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1)