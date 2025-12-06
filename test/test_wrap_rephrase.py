import unittest
from pii_sanitizer import PiiSanitizer, RunningMode

class TestWrapRephraseIntegration(unittest.TestCase):
    def setUp(self):
        # Use PRESIDIO mode for integration testing to verify real logic
        # independent of AWS Bedrock availability
        self.sanitizer = PiiSanitizer(running_mode=RunningMode.PRESIDIO)

    def test_wrap_rephrase_basic_integration(self):
        """
        Test that wrap_rephrase correctly anonymizes input before passing to the decorated function,
        and restores it afterwards using the real PiiSanitizer logic.
        """
        original_message = "My name is John Doe and my email is john.doe@example.com"
        
        @self.sanitizer.wrap_rephrase()
        def mock_llm_echo(anonymized_text: str) -> str:
            # Assert that PII has been replaced with placeholders inside the function
            # Note: The exact placeholder format depends on InstanceCounterAnonymizer
            # Typically it might be <PERSON_0>, <EMAIL_ADDRESS_0> etc.
            self.assertNotIn("John Doe", anonymized_text)
            self.assertNotIn("john.doe@example.com", anonymized_text)
            self.assertIn("PERSON", anonymized_text)
            self.assertIn("EMAIL_ADDRESS", anonymized_text)
            return anonymized_text

        # The result should be the original message because the mock function just returns the anonymized text,
        # and wrap_rephrase restores the original values into the placeholders.
        restored_message = mock_llm_echo(original_message)
        
        self.assertEqual(restored_message, original_message)

    def test_wrap_rephrase_with_text_modification(self):
        """
        Test that wrap_rephrase works when the decorated function modifies the text
        (simulating an LLM generating a response that includes the entities).
        """
        original_message = "Contact support at 555-0199."
        
        @self.sanitizer.wrap_rephrase()
        def mock_llm_response(anonymized_text: str) -> str:
            # Simulate LLM understanding the context and replying
            # It should receive something like "Contact support at <PHONE_NUMBER_0>."
            return f"I received your number: {anonymized_text}"

        result = mock_llm_response(original_message)
        
        # The placeholders in the LLM response should be restored to real values
        expected_result = "I received your number: Contact support at 555-0199."
        self.assertEqual(result, expected_result)

    def test_wrap_rephrase_empty_string(self):
        @self.sanitizer.wrap_rephrase()
        def simple_func(text):
            return "Processed " + text

        self.assertEqual(simple_func(""), "Processed ")
        self.assertEqual(simple_func(None), None) # Depending on how anonymize_message handles None

if __name__ == '__main__':
    unittest.main()

