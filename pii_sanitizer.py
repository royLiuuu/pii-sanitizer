import re
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


class RunningMode(Enum):
    BEDROCK_GUARDRAIL = 0
    PRESIDIO = 1
    PRESIDIO_TRANSFORMER = 2


class PiiSanitizer:
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

    def __init__(self, *args, **kwargs):
        self.running_mode = kwargs.pop("running_mode", RunningMode.PRESIDIO)
        match self.running_mode:
            case RunningMode.PRESIDIO:
                self.engine = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                self.anonymizer.add_anonymizer(InstanceCounterAnonymizer)
            case RunningMode.PRESIDIO_TRANSFORMER:
                labels_to_ignore = ["O"]
                ner_model_configuration = NerModelConfiguration(
                    model_to_presidio_entity_mapping=self._TRANSFORMER_MAPPINGS,
                    alignment_mode="expand",  # "strict", "contract", "expand"
                    aggregation_strategy="max",  # "simple", "first", "average", "max"
                    labels_to_ignore=labels_to_ignore)

                engine = TransformersNlpEngine(
                    models=self._TRANSFORMER_CONFIG,
                    ner_model_configuration=ner_model_configuration)

                # Transformer-based analyzer
                self.engine = AnalyzerEngine(
                    nlp_engine=engine,
                    supported_languages=["en"]
                )
                self.anonymizer = AnonymizerEngine()
                self.anonymizer.add_anonymizer(InstanceCounterAnonymizer)
            case RunningMode.BEDROCK_GUARDRAIL:
                self.bedrock_client = self.build_bedrock_client(**kwargs)
            case _:
                raise InvalidParamException("running mode is not valid")

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
                if message is None:
                    return None
                if message == '':
                    return f(message)
                anonymized_message, pii_result = self.anonymize_message(message)
                llm_result = f(anonymized_message)
                restore_pii_message = self.restore_pii(llm_result, pii_result)
                return restore_pii_message

            return inner

        return decorator

    def anonymize_message(self, message: str):
        if message is None or message == '':
            return message, None

        match self.running_mode:
            case RunningMode.PRESIDIO:
                results = self.engine.analyze(text=message,
                                              language='en')
                entity_mapping = dict()
                anonymized_text = self.anonymizer.anonymize(message, results, {
                    "DEFAULT": OperatorConfig(
                        "entity_counter", {"entity_mapping": entity_mapping}
                    )
                })
                return anonymized_text.text, entity_mapping
            case RunningMode.PRESIDIO_TRANSFORMER:
                results = self.engine.analyze(text=message,
                                              language='en')
                entity_mapping = dict()
                anonymized_text = self.anonymizer.anonymize(message, results, {
                    "DEFAULT": OperatorConfig(
                        "entity_counter", {"entity_mapping": entity_mapping}
                    )
                })
                return anonymized_text.text, entity_mapping
            case RunningMode.BEDROCK_GUARDRAIL:
                guardrail_response = self.bedrock_client.apply_guardrail(
                    guardrailIdentifier=self.bedrock_guardrail_arn,
                    guardrailVersion=self.bedrock_guardrail_arn_version,
                    source="INPUT",  # Evaluate the input prompt
                    content=[
                        {
                            "text": {
                                "text": message
                            }
                        }
                    ]
                )
                if 'NONE' == guardrail_response['action']:
                    return message, dict()

                anonymized_text, pii_result = self.bedrock_guardrail_to_presidio(guardrail_response)
                return anonymized_text, pii_result

    def bedrock_guardrail_to_presidio(self,
                                      bedrock_response: Dict[str, Any],
                                      pii_type_map: Dict[str, str] = None,
                                      ) -> Tuple[str, Dict[str, Dict[str, str]]]:

        if pii_type_map is None:
            pii_type_map = self.BEDROCK_TO_PRESIDIO

        outputs = bedrock_response.get("outputs", [])
        if not outputs:
            return "", {}

        text = outputs[0].get("text", "")

        assessments = bedrock_response.get("assessments", [])
        if not assessments:
            return text, {}

        pii_entities = (
            assessments[0]
            .get("sensitiveInformationPolicy", {})
            .get("piiEntities", [])
        )

        # 将实体按类型收集
        entities_by_type: Dict[str, list] = defaultdict(list)
        for ent in pii_entities:
            t = ent.get("type")
            match = ent.get("match")
            if t and match:
                entities_by_type[t].append(match)

        # 对每种类型去重，确定 placeholder 编号
        # 例如 ["roy","roy","ben","roy"] → ["roy","ben"]
        unique_entities_by_type = {
            t: list(dict.fromkeys(vals))  # 去重但保持顺序
            for t, vals in entities_by_type.items()
        }

        # 生成 placeholder 映射
        # 例如 "<PERSON_0>": "Don"
        pii_mapping: Dict[str, Dict[str, str]] = defaultdict(dict)

        # 反向映射：raw_value → placeholder_token
        reverse_lookup: Dict[str, Dict[str, str]] = defaultdict(dict)

        for bedrock_type, vals in unique_entities_by_type.items():
            presidio_type = pii_type_map.get(bedrock_type, bedrock_type)
            for idx, raw in enumerate(vals):
                placeholder_core = f"{presidio_type}_{idx}"
                placeholder_token = f"{{{placeholder_core}}}"
                mapping_key = f"<{placeholder_core}>"

                pii_mapping[presidio_type][mapping_key] = raw
                reverse_lookup[presidio_type][raw] = placeholder_token

        # 替换文本：按出现顺序扫描 {NAME} 等占位符，每次取下一个实体值
        used_count_by_type = {t: 0 for t in entities_by_type}

        pattern = re.compile(r"\{([A-Z_]+)\}")

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

            return reverse_lookup[presidio_type][raw_value]

        new_text = pattern.sub(replace_placeholder, text)

        return new_text, reverse_lookup

    def deanonymize_message(self, message: str, entity_mapping):
        if message is None or message == '':
            return message
        return self.restore_pii(message, entity_mapping)

    def build_placeholder_map(self, pii_mapping: dict) -> dict:
        """
        把嵌套的 mapping 展平，得到:
        {'{PERSON_0}': 'Don', '{PERSON_1}': 'roy', ...}
        """
        flat = {}
        for pii_type, inner in pii_mapping.items():
            for real_value, placeholder in inner.items():
                flat[placeholder] = real_value
        return flat

    def restore_pii(self, masked_message: str, pii_mapping: dict) -> str:
        """
        根据 mapping 恢复 message 中的 {PERSON_0}、{PHONE_NUMBER_0} 等占位符
        """
        flat_map = self.build_placeholder_map(pii_mapping)

        # 匹配 {PERSON_0} / {PHONE_NUMBER_0} / {EMAIL_ADDRESS_0} 等
        pattern = re.compile(r'[<{]([A-Z_0-9]+)[>}]')

        def replace(match: re.Match) -> str:
            placeholder = match.group(0)  # 比如 "{PERSON_0}"
            return flat_map.get(placeholder, placeholder)

        return pattern.sub(replace, masked_message)
