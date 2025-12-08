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

from .instance_counter_anonymizer import InstanceCounterAnonymizer
from .invalid_param_exception import InvalidParamException

# Ensure logger is defined
logger = logging.getLogger(__name__)


class RunningMode(Enum):
    BEDROCK_GUARDRAIL = 0
    PRESIDIO = 1
    PRESIDIO_TRANSFORMER = 2


class PiiSanitizer:
    # Singleton control
    _instance = None
    _lock = threading.RLock()
    _initialized = False

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
        Singleton implementation.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PiiSanitizer, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        """
        Initialize resources only once.
        """
        if self._initialized:
            return

        with self._lock:
            # Double-check locking
            if self._initialized:
                return
            self._load_resources(*args, **kwargs)
            self._initialized = True

    def reload(self, *args, **kwargs):
        """
        Thread-safe resource reload using Copy-on-Write.
        """
        # Prevent re-entry during reload
        with self._lock:
            # 1. Parse new config
            new_running_mode = kwargs.get("running_mode", RunningMode.PRESIDIO)
            # Extract Bedrock config (build_bedrock_client pops args)
            new_guardrail_arn = kwargs.get("bedrock_guardrail_arn")
            new_guardrail_version = kwargs.get("bedrock_guardrail_arn_version")
            
            if self._initialized and new_running_mode == getattr(self, "running_mode", None):
                logger.info(f"Reloading PiiSanitizer: RunningMode remains {new_running_mode}")
                
                # Skip reload for local models if config is unchanged
                if new_running_mode in (RunningMode.PRESIDIO, RunningMode.PRESIDIO_TRANSFORMER):
                    logger.info("Configuration unchanged for local model mode. Skipping reload.")
                    return
                
                # Bedrock mode might need client update even if mode matches
            else:
                 logger.info(f"Reloading PiiSanitizer: Switching from {getattr(self, 'running_mode', 'None')} to {new_running_mode}")

            logger.info("Starting resource reload...")
            
            new_engine = None
            new_anonymizer = None
            new_bedrock_client = None
            
            # 2. Preload resources (heavy lifting)
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
                    # Rebuild client if keys changed
                    new_bedrock_client = self.build_bedrock_client(**kwargs)
                case _:
                    raise InvalidParamException("running mode is not valid")

            # 3. Atomic swap
            self.running_mode = new_running_mode
            self.engine = new_engine
            self.anonymizer = new_anonymizer
            self.bedrock_client = new_bedrock_client
            
            # Update Bedrock params
            if new_guardrail_arn:
                self.bedrock_guardrail_arn = new_guardrail_arn
            if new_guardrail_version:
                self.bedrock_guardrail_arn_version = new_guardrail_version

            self._initialized = True
            logger.info("Reload complete.")

    def _load_resources(self, *args, **kwargs):
        """
        Reuse reload logic.
        """
        # Reuse reload
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
                # Lock critical methods
                
                if message is None:
                    return None
                # Ensure string input
                if not isinstance(message, str):
                    raise InvalidParamException(f"Input message must be a string, got {type(message)}")
                
                if message == '':
                    return f(message)
                
                # 1. Anonymize (locks)
                anonymized_message, pii_result = self.anonymize_message(message)
                
                # 2. LLM call (unlocked, potentially slow)
                llm_result = f(anonymized_message)
                
                # 3. Restore (locks)
                restore_pii_message = self.restore_pii(llm_result, pii_result)
                return restore_pii_message

            return inner

        return decorator

    def anonymize_message(self, message: str):
        # Track stats
        start_time = time.time()
        msg_len = len(message) if message else 0
        logger.debug(f"Starting anonymization. Input length: {msg_len} chars")

        if message is None or message == '':
            return message, None

        # Validate input
        if not isinstance(message, str):
            logger.error(f"Invalid input type: {type(message)}")
            raise InvalidParamException(f"Input message must be a string")

        # Generate unique request token
        # e.g. 'A1B2C3D4'
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
                                "mask_token": mask_token  # Pass token
                            }
                        )
                    })
                    
                    # Log stats (no PII)
                    self._log_anonymization_stats(entity_mapping, start_time)
                    
                    return anonymized_text.text, entity_mapping

                case RunningMode.PRESIDIO_TRANSFORMER:
                    results = self.engine.analyze(text=message, language='en')
                    entity_mapping = dict()
                    anonymized_text = self.anonymizer.anonymize(message, results, {
                        "DEFAULT": OperatorConfig(
                            "entity_counter", {
                                "entity_mapping": entity_mapping, 
                                "mask_token": mask_token  # Pass token
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
                    
                    # Log Bedrock metadata
                    request_id = guardrail_response.get("ResponseMetadata", {}).get("RequestId", "unknown")
                    action = guardrail_response.get('action')
                    logger.info(f"Bedrock Guardrail called. RequestId: {request_id}, Action: {action}")

                    if 'NONE' == action:
                        logger.debug("No PII detected by Bedrock.")
                        return message, dict()

                    # Pass mask_token
                    anonymized_text, pii_result = self.bedrock_guardrail_to_presidio(
                        guardrail_response, 
                        mask_token=mask_token 
                    )
                    
                    # Log Bedrock entities
                    self._log_anonymization_stats(pii_result, start_time)
                    
                    return anonymized_text, pii_result

        except Exception as e:
            # Log error
            logger.error(f"Anonymization failed: {str(e)}", exc_info=True)
            raise e

    def bedrock_guardrail_to_presidio(self,
                                      bedrock_response: Dict[str, Any],
                                      pii_type_map: Dict[str, str] = None,
                                      mask_token: str = ""  # Accepts mask_token
                                      ) -> Tuple[str, Dict[str, Dict[str, str]]]:

        # Basic validation
        if not bedrock_response or not isinstance(bedrock_response, dict):
            return "", {}

        if pii_type_map is None:
            pii_type_map = self.BEDROCK_TO_PRESIDIO

        outputs = bedrock_response.get("outputs", [])
        # Ensure outputs exist
        if not outputs or not isinstance(outputs[0], dict):
            # Return empty if no output
            return "", {}

        text = outputs[0].get("text", "")

        assessments = bedrock_response.get("assessments", [])
        if not assessments:
            return text, {}

        # Safe property access
        policy_assessment = assessments[0].get("sensitiveInformationPolicy", {})
        pii_entities = policy_assessment.get("piiEntities", [])
        
        if not pii_entities:
            return text, {}

        # Group entities by type
        entities_by_type: Dict[str, list] = defaultdict(list)
        for ent in pii_entities:
            if not isinstance(ent, dict):
                continue
            t = ent.get("type")
            match_val = ent.get("match")
            if t and match_val:
                entities_by_type[t].append(match_val)

        # Deduplicate per type
        unique_entities_by_type = {
            t: list(dict.fromkeys(vals))  # Preserve order
            for t, vals in entities_by_type.items()
        }

        # Prepare suffix
        suffix = f"_{mask_token}" if mask_token else ""

        # Generate placeholder map
        # e.g. "<PERSON_0_A1B2C3D4>": "Roy"
        pii_mapping: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Reverse map: raw -> placeholder
        reverse_lookup: Dict[str, Dict[str, str]] = defaultdict(dict)

        for bedrock_type, vals in unique_entities_by_type.items():
            presidio_type = pii_type_map.get(bedrock_type, bedrock_type)
            for idx, raw in enumerate(vals):
                # Append mask_token
                placeholder_core = f"{presidio_type}_{idx}{suffix}"
                placeholder_token = f"<{placeholder_core}>"

                # Match InstanceCounterAnonymizer format
                pii_mapping[presidio_type][raw] = placeholder_token
                reverse_lookup[presidio_type][raw] = placeholder_token

        # Replace text using reverse_lookup
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

            # Guard against missing lookup
            type_lookup = reverse_lookup.get(presidio_type)
            if type_lookup and raw_value in type_lookup:
                return type_lookup[raw_value] # Returns token with suffix
            return m.group(0)

        new_text = self._PLACEHOLDER_PATTERN.sub(replace_placeholder, text)

        return new_text, reverse_lookup


    def deanonymize_message(self, message: str, entity_mapping):
        if message is None:
            return None
            
        # Type check
        if not isinstance(message, str):
             raise InvalidParamException(f"Input message must be a string, got {type(message)}")

        if message == '':
            return message
        return self.restore_pii(message, entity_mapping)

    def build_placeholder_map(self, pii_mapping: dict) -> dict:
        """
        Flatten nested mapping.
        """
        flat = {}
        # Defensive check
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
        Restore placeholders.
        """
        # Boundary checks
        if not masked_message:
            return masked_message
        if not pii_mapping:
            return masked_message

        flat_map = self.build_placeholder_map(pii_mapping)
        if not flat_map:
            return masked_message

        # Match placeholders like <PERSON_0_TOKEN>
        pattern = re.compile(r'[<{]([A-Z_0-9]+)[>}]')

        def replace(match: re.Match) -> str:
            placeholder = match.group(0)  # e.g. "{PERSON_0}"
            return flat_map.get(placeholder, placeholder)

        return pattern.sub(replace, masked_message)

    def _log_anonymization_stats(self, entity_mapping: Dict, start_time: float):
        """
        Log anonymization stats safely (no PII).
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
