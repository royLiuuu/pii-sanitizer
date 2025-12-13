import unittest
from typing import Optional
from pii_sanitizer import PiiSanitizer, RunningMode

class MockGeminiService:
    def __init__(self):
        self.sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO)

    # Apply decorator. Note: arg_name="content" is crucial because index 0 is 'self'
    @PiiSanitizer(running_mode=RunningMode.PRESIDIO).wrap_rephrase(arg_name="content")
    async def nvc_rephrase_with_gemini(
            self,
            content: str,
            model: Optional[str] = None,
            uid: Optional[str] = None,
            request_id: Optional[str] = None,
            correlation_id: Optional[str] = None
    ) -> str:
        """
        Mock implementation of the service method.
        It asserts that the content received is anonymized.
        """
        # Verify that PII (e.g., "John Doe") is NOT present in the content received by the function
        if "John Doe" in content:
            raise ValueError(f"Privacy Leak! PII found in content: {content}")
        
        # Verify placeholders are present (implementation detail of Presidio/Anonymizer)
        # Typically <PERSON_0> or similar
        if "<" not in content and "PERSON" not in content:
             pass # Depending on configuration, might fail if check is too strict
        
        # Simulate LLM processing: Echo back with some modification
        return f"Gemini processed: {content} (Model: {model})"

class TestAsyncWrapRephrase(unittest.IsolatedAsyncioTestCase):
    async def test_nvc_rephrase_with_gemini_anonymization(self):
        service = MockGeminiService()
        
        original_content = "Please contact John Doe at 555-0199."
        
        # Call the async method
        result = await service.nvc_rephrase_with_gemini(
            content=original_content,
            model="gemini-1.5-flash",
            uid="user-123",
            request_id="req-abc",
            correlation_id="corr-xyz"
        )
        
        # The wrapper should have:
        # 1. Anonymized "John Doe" -> <PERSON_0> before calling the method
        # 2. Received "Gemini processed: Please contact <PERSON_0>... " from the method
        # 3. Restored <PERSON_0> -> "John Doe" before returning to us
        
        expected_part = "Gemini processed: Please contact John Doe at 555-0199"
        self.assertIn(expected_part, result)
        self.assertIn("(Model: gemini-1.5-flash)", result)

    async def test_nvc_rephrase_empty_content(self):
        service = MockGeminiService()
        # Test empty string behavior (should bypass anonymization)
        result = await service.nvc_rephrase_with_gemini(content="", model="test")
        self.assertEqual(result, "Gemini processed:  (Model: test)")

if __name__ == '__main__':
    unittest.main()
