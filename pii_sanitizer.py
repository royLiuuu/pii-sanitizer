import re
import threading
import logging
import time
import uuid
from collections import defaultdict
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import boto3
from boto3 import Session
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig

from instance_counter_anonymizer import InstanceCounterAnonymizer
from invalid_param_exception import InvalidParamException

# 确保 logger 已定义
logger = logging.getLogger(__name__)


class RunningMode(Enum):
    BEDROCK_GUARDRAIL = 0
    PRESIDIO = 1
    PRESIDIO_TRANSFORMER = 2


class PiiSanitizer:
    # 单例控制变量
    _instance = None
    _lock = threading.RLock()

    running_mode: RunningMode
    bedrock_client: Optional
    bedrock_guardrail_arn: Optional[str]
    bedrock_guardrail_arn_version: Optional[str]
    engine: Optional[AnalyzerEngine]
    anonymizer: Optional[AnonymizerEngine]

    BEDROCK_TO_PRESIDIO = {
        "NAME": "PERSON",
        "PHONE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "ADDRESS": "LOCATION",
    }

    _TRANSFORMER_CONFIG = [
        {"lang_code": "en",
         "model_name": {
             "spacy": "en_core_web_sm",  # for tokenization, lemmatization
             "transformers": "StanfordAIMI/stanford-deidentifier-base"  # for NER
         }
         }]
    _TRANSFORMER_MAPPINGS = dict(
        PER="PERSON",
        LOC="LOCATION",
        ORG="ORGANIZATION",
        AGE="AGE",
        ID="ID",
        EMAIL="EMAIL",
        DATE="DATE_TIME",
        PHONE="PHONE_NUMBER",
        PERSON="PERSON",
        LOCATION="LOCATION",
        GPE="LOCATION",
        ORGANIZATION="ORGANIZATION",
        NORP="NRP",
        PATIENT="PERSON",
        STAFF="PERSON",
        HOSP="LOCATION",
        PATORG="ORGANIZATION",
        TIME="DATE_TIME",
        HCW="PERSON",
        HOSPITAL="LOCATION",
        FACILITY="LOCATION",
        VENDOR="ORGANIZATION",
    )

    _PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z_]+)\}")
    _RESTORE_PATTERN = re.compile(r'<([A-Z_]+_\d+)>')

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式：确保全局只有一个 PiiSanitizer 实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PiiSanitizer, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        """
        初始化方法。仅在第一次创建实例时执行资源加载。
        """
        # 如果已经初始化过，直接返回，避免重复加载模型导致内存飙升
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            # 双重检查，防止并发初始化
            if getattr(self, "_initialized", False):
                return
            self._load_resources(*args, **kwargs)
            self._initialized = True

    def reload(self, *args, **kwargs):
        """
        线程安全的重新加载方法。
        使用 Copy-on-Write 策略：先在局部变量中加载新资源，最后原子替换。
        """
        # 即使业务逻辑不加锁，reload 自身需要防重入
        with self._lock:
            # 1. 解析新配置 (提取到外层，以便后续更新 self)
            new_running_mode = kwargs.get("running_mode", RunningMode.PRESIDIO)
            # 提取 Bedrock 相关配置，如果有传入则使用，否则保持 None (或从 kwargs 获取)
            # 注意：build_bedrock_client 内部会 pop 这些参数，所以最好先提取出来
            new_guardrail_arn = kwargs.get("bedrock_guardrail_arn")
            new_guardrail_version = kwargs.get("bedrock_guardrail_arn_version")
            
            if self._initialized and new_running_mode == self.running_mode:
                logger.info(f"Reloading PiiSanitizer: RunningMode remains {new_running_mode}")
                
                # 优化：如果是本地模型模式且已经加载好，通常不需要重载，直接返回以节省资源
                if new_running_mode in (RunningMode.PRESIDIO, RunningMode.PRESIDIO_TRANSFORMER):
                    logger.info("Configuration unchanged for local model mode. Skipping reload.")
                    return
                
                # 注意：对于 BEDROCK_GUARDRAIL，即使 mode 没变，
                # 也可能通过 kwargs 更新了 access_key 或 arn，所以这里不 return，继续往下走重建 client
            else:
                 logger.info(f"Reloading PiiSanitizer: Switching from {getattr(self, 'running_mode', 'None')} to {new_running_mode}")

            logger.info("Starting resource reload...")
            
            new_engine = None
            new_anonymizer = None
            new_bedrock_client = None
            
            # 2. 预加载新资源 (耗时操作放在这里)
            match new_running_mode:
                case RunningMode.PRESIDIO:
                    new_engine = AnalyzerEngine()
                    new_anonymizer = AnonymizerEngine()
                    new_anonymizer.add_anonymizer(InstanceCounterAnonymizer)
                case RunningMode.PRESIDIO_TRANSFORMER:
                    try:
                        labels_to_ignore = ["O"]
                        ner_model_configuration = NerModelConfiguration(
                            model_to_presidio_entity_mapping=self._TRANSFORMER_MAPPINGS,
                            alignment_mode="expand",
                            aggregation_strategy="max",
                            labels_to_ignore=labels_to_ignore)

                        engine = TransformersNlpEngine(
                            models=self._TRANSFORMER_CONFIG,
                            ner_model_configuration=ner_model_configuration)

                        new_engine = AnalyzerEngine(
                            nlp_engine=engine,
                            supported_languages=["en"]
                        )
                    except Exception as e:
                        logger.error(f"Failed to load Transformer model, falling back to default Presidio: {e}")
                        new_engine = AnalyzerEngine()

                    new_anonymizer = AnonymizerEngine()
                    new_anonymizer.add_anonymizer(InstanceCounterAnonymizer)
                case RunningMode.BEDROCK_GUARDRAIL:
                    # 注意：如果只是 mode 没变但 key 变了，build_bedrock_client 会使用新的 kwargs
                    new_bedrock_client = self.build_bedrock_client(**kwargs)
                case _:
                    raise InvalidParamException("running mode is not valid")

            # 3. 原子替换 (Atomic Swap)
            self.running_mode = new_running_mode
            self.engine = new_engine
            self.anonymizer = new_anonymizer
            self.bedrock_client = new_bedrock_client
            
            # 关键：更新 Bedrock 配置参数
            if new_guardrail_arn:
                self.bedrock_guardrail_arn = new_guardrail_arn
            if new_guardrail_version:
                self.bedrock_guardrail_arn_version = new_guardrail_version

            self._initialized = True
            logger.info("Reload complete.")

    def _load_resources(self, *args, **kwargs):
        """
        初始化加载，直接复用 reload 逻辑即可，或者保持原样。
        """
        # 为了代码复用，初始化可以直接调用 reload
        self.reload(*args, **kwargs)

    def build_bedrock_client(self, **kwargs):
        region = kwargs.pop("region", None)
        self.bedrock_guardrail_arn = kwargs.pop("bedrock_guardrail_arn", None)
        self.bedrock_guardrail_arn_version = kwargs.pop("bedrock_guardrail_arn_version", None)
        profile = kwargs.pop("profile", None)
        access_key = kwargs.pop("access_key", None)
        secret_key = kwargs.pop("secret_key", None)
        session_token = kwargs.pop("session_token", None)
        if region is None:
            raise InvalidParamException("region is not provided")
        if profile is not None:
            return Session(profile_name=profile).client(service_name="bedrock-runtime", **kwargs)
        if access_key and secret_key:
            return boto3.client(service_name="bedrock-runtime",
                                aws_access_key_id=access_key,
                                aws_secret_access_key=secret_key,
                                aws_session_token=session_token,
                                region_name=region,
                                **kwargs
                                )
        return boto3.client(service_name="bedrock-runtime",
                            region_name=region)

    def wrap_rephrase(self):
        def decorator(f):
            def inner(message: str) -> str:
                # 注意：这里不需要显式加锁，因为 anonymize_message 和 restore_pii 内部会加锁。
                # 但为了保证“脱敏->LLM调用->还原”整个流程期间配置不发生变化（原子性），
                # 可以在这里加锁。但LLM调用通常很慢，锁住会导致全系统阻塞。
                # 权衡：通常只锁“脱敏”和“还原”这两个瞬间操作即可。如果reload发生了，
                # 可能导致脱敏用了旧配置，还原用了新配置。
                # 建议：仅锁方法内部调用，或者在这里显式获取锁（如果接受 LLM 调用期间无法 reload）。
                
                # 方案 A: 只锁关键方法（推荐，高并发友好）
                # reload 可能会在 LLM 调用期间发生，但只要 restore 逻辑兼容即可。
                
                if message is None:
                    return None
                # 增加类型检查，确保 message 是字符串
                if not isinstance(message, str):
                    raise InvalidParamException(f"Input message must be a string, got {type(message)}")
                
                if message == '':
                    return f(message)
                
                # 1. 脱敏 (会获取锁)
                anonymized_message, pii_result = self.anonymize_message(message)
                
                # 2. LLM 调用 (无锁，允许耗时操作)
                llm_result = f(anonymized_message)
                
                # 3. 还原 (会获取锁)
                restore_pii_message = self.restore_pii(llm_result, pii_result)
                return restore_pii_message

            return inner

        return decorator

    def anonymize_message(self, message: str):
        # [新增] 开始计时 & 记录输入元数据
        start_time = time.time()
        msg_len = len(message) if message else 0
        logger.debug(f"Starting anonymization. Input length: {msg_len} chars")

        if message is None or message == '':
            return message, None

        # [建议] 配合之前的输入验证
        if not isinstance(message, str):
            logger.error(f"Invalid input type: {type(message)}")
            raise InvalidParamException(f"Input message must be a string")

        # [新增] 生成本次请求的随机 Token (8位大写十六进制)
        # 例如: 'A1B2C3D4'，这确保了每次请求的占位符后缀都是独一无二的
        mask_token = uuid.uuid4().hex[:8].upper()

        try:
            match self.running_mode:
                case RunningMode.PRESIDIO:
                    results = self.engine.analyze(text=message, language='en')
                    entity_mapping = dict()
                    anonymized_text = self.anonymizer.anonymize(message, results, {
                        "DEFAULT": OperatorConfig(
                            "entity_counter", {
                                "entity_mapping": entity_mapping, 
                                "mask_token": mask_token  # [新增] 传入 token
                            }
                        )
                    })
                    
                    # [新增] 记录统计信息 (无 PII)
                    self._log_anonymization_stats(entity_mapping, start_time)
                    
                    return anonymized_text.text, entity_mapping

                case RunningMode.PRESIDIO_TRANSFORMER:
                    results = self.engine.analyze(text=message, language='en')
                    entity_mapping = dict()
                    anonymized_text = self.anonymizer.anonymize(message, results, {
                        "DEFAULT": OperatorConfig(
                            "entity_counter", {
                                "entity_mapping": entity_mapping, 
                                "mask_token": mask_token  # [新增] 传入 token
                            }
                        )
                    })
                    self._log_anonymization_stats(entity_mapping, start_time)
                    return anonymized_text.text, entity_mapping

                case RunningMode.BEDROCK_GUARDRAIL:
                    guardrail_response = self.bedrock_client.apply_guardrail(
                        guardrailIdentifier=self.bedrock_guardrail_arn,
                        guardrailVersion=self.bedrock_guardrail_arn_version,
                        source="INPUT",
                        content=[{"text": {"text": message}}]
                    )
                    
                    # [新增] 记录 Bedrock 请求元数据
                    request_id = guardrail_response.get("ResponseMetadata", {}).get("RequestId", "unknown")
                    action = guardrail_response.get('action')
                    logger.info(f"Bedrock Guardrail called. RequestId: {request_id}, Action: {action}")

                    if 'NONE' == action:
                        logger.debug("No PII detected by Bedrock.")
                        return message, dict()

                    # [修改] 调用转换函数时，传入 mask_token
                    anonymized_text, pii_result = self.bedrock_guardrail_to_presidio(
                        guardrail_response, 
                        mask_token=mask_token 
                    )
                    
                    # [新增] 统计 Bedrock 返回的实体
                    self._log_anonymization_stats(pii_result, start_time)
                    
                    return anonymized_text, pii_result

        except Exception as e:
            # [新增] 错误日志
            logger.error(f"Anonymization failed: {str(e)}", exc_info=True)
            raise e

    def bedrock_guardrail_to_presidio(self,
                                      bedrock_response: Dict[str, Any],
                                      pii_type_map: Dict[str, str] = None,
                                      mask_token: str = ""  # <--- [新增] 接收 mask_token 参数
                                      ) -> Tuple[str, Dict[str, Dict[str, str]]]:

        # 基础输入验证
        if not bedrock_response or not isinstance(bedrock_response, dict):
            return "", {}

        if pii_type_map is None:
            pii_type_map = self.BEDROCK_TO_PRESIDIO

        outputs = bedrock_response.get("outputs", [])
        # 确保 outputs 列表非空且第一个元素包含 text
        if not outputs or not isinstance(outputs[0], dict):
            # 如果没有 outputs，尝试直接返回原始文本或空字符串
            # 这里假设如果 guardrail 拦截了，outputs 应该有修正后的文本
            return "", {}

        text = outputs[0].get("text", "")

        assessments = bedrock_response.get("assessments", [])
        if not assessments:
            return text, {}

        # 安全获取嵌套属性
        policy_assessment = assessments[0].get("sensitiveInformationPolicy", {})
        pii_entities = policy_assessment.get("piiEntities", [])
        
        if not pii_entities:
            return text, {}

        # 将实体按类型收集
        entities_by_type: Dict[str, list] = defaultdict(list)
        for ent in pii_entities:
            if not isinstance(ent, dict):
                continue
            t = ent.get("type")
            match_val = ent.get("match")
            if t and match_val:
                entities_by_type[t].append(match_val)

        # 对每种类型去重，确定 placeholder 编号
        # 例如 ["roy","roy","ben","roy"] → ["roy","ben"]
        unique_entities_by_type = {
            t: list(dict.fromkeys(vals))  # 去重但保持顺序
            for t, vals in entities_by_type.items()
        }

        # [新增] 准备后缀
        suffix = f"_{mask_token}" if mask_token else ""

        # 生成 placeholder 映射
        # 例如 "<PERSON_0_A1B2C3D4>": "Roy"
        pii_mapping: Dict[str, Dict[str, str]] = defaultdict(dict)

        # 反向映射：raw_value → placeholder_token
        reverse_lookup: Dict[str, Dict[str, str]] = defaultdict(dict)

        for bedrock_type, vals in unique_entities_by_type.items():
            presidio_type = pii_type_map.get(bedrock_type, bedrock_type)
            for idx, raw in enumerate(vals):
                # [修改] 拼接 mask_token 到占位符核心部分
                placeholder_core = f"{presidio_type}_{idx}{suffix}"
                placeholder_token = f"<{placeholder_core}>"

                # 保持与 InstanceCounterAnonymizer 结构一致：{ raw: placeholder }
                pii_mapping[presidio_type][raw] = placeholder_token
                reverse_lookup[presidio_type][raw] = placeholder_token

        # 替换文本逻辑保持不变，因为它是基于 entities_by_type 的顺序来查找值的
        # 而用于替换的具体字符串是直接从 reverse_lookup 拿的，所以这里不需要改动。
        used_count_by_type = {t: 0 for t in entities_by_type}

        def replace_placeholder(m: re.Match) -> str:
            bedrock_type = m.group(1)
            if bedrock_type not in entities_by_type:
                return m.group(0)

            vals = entities_by_type[bedrock_type]
            idx = used_count_by_type[bedrock_type]

            if idx >= len(vals):
                return m.group(0)

            used_count_by_type[bedrock_type] += 1

            raw_value = vals[idx]
            presidio_type = pii_type_map.get(bedrock_type, bedrock_type)

            # 防御 reverse_lookup 查找失败
            type_lookup = reverse_lookup.get(presidio_type)
            if type_lookup and raw_value in type_lookup:
                return type_lookup[raw_value] # 这里会返回带后缀的新 Token
            return m.group(0)

        new_text = self._PLACEHOLDER_PATTERN.sub(replace_placeholder, text)

        return new_text, reverse_lookup


    def deanonymize_message(self, message: str, entity_mapping):
        if message is None:
            return None
            
        # 类型检查
        if not isinstance(message, str):
             raise InvalidParamException(f"Input message must be a string, got {type(message)}")

        if message == '':
            return message
        return self.restore_pii(message, entity_mapping)

    def build_placeholder_map(self, pii_mapping: dict) -> dict:
        """
        把嵌套的 mapping 展平，得到:
        {'{PERSON_0}': 'Don', '{PERSON_1}': 'roy', ...}
        """
        flat = {}
        # 防御性检查
        if not pii_mapping or not isinstance(pii_mapping, dict):
            return flat
            
        for pii_type, inner in pii_mapping.items():
            if not isinstance(inner, dict): 
                continue
            for real_value, placeholder in inner.items():
                flat[placeholder] = real_value
        return flat

    def restore_pii(self, masked_message: str, pii_mapping: dict) -> str:
        """
        根据 mapping 恢复 message 中的 {PERSON_0}、{PHONE_NUMBER_0} 等占位符
        """
        # 边界检查
        if not masked_message:
            return masked_message
        if not pii_mapping:
            return masked_message

        flat_map = self.build_placeholder_map(pii_mapping)
        if not flat_map:
            return masked_message

        # 匹配 {PERSON_0} / {PHONE_NUMBER_0} / {EMAIL_ADDRESS_0} 等
        # [说明]
        # 原有的正则 r'[<{]([A-Z_0-9]+)[>}]' 已经足够匹配如 <PERSON_0_1A2B3C> 这样的格式
        # 只要 mask_token 保持是大写字母+数字即可。
        # 如果你希望支持更复杂的字符（如小写），需要修改正则：
        # pattern = re.compile(r'[<{]([A-Za-z0-9_]+)[>}]') 
        
        pattern = re.compile(r'[<{]([A-Z_0-9]+)[>}]')

        def replace(match: re.Match) -> str:
            placeholder = match.group(0)  # 比如 "{PERSON_0}"
            return flat_map.get(placeholder, placeholder)

        return pattern.sub(replace, masked_message)

    def _log_anonymization_stats(self, entity_mapping: Dict, start_time: float):
        """
        安全地记录脱敏统计信息，确保不包含具体 PII 值。
        entity_mapping 结构通常为: {'PERSON': {'Roy': '<PERSON_0>'}, ...}
        """
        duration = (time.time() - start_time) * 1000
        stats = []
        total_entities = 0
        
        if entity_mapping:
            for entity_type, mapping in entity_mapping.items():
                count = len(mapping)
                total_entities += count
                stats.append(f"{entity_type}:{count}")
        
        stat_str = ", ".join(stats)
        logger.info(f"Anonymization completed in {duration:.2f}ms. "
                    f"Total entities: {total_entities}. Details: [{stat_str}]")
